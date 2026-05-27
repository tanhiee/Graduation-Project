import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, render_template, request

# Tắt log TensorFlow để tránh rác terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers

app = Flask(__name__)

# Thư mục lưu trữ mô hình
MODEL_DIR = 'deploy_models'
scaler_path = os.path.join(MODEL_DIR, 'minmax_scaler.pkl')
if_path = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
encoder_path = os.path.join(MODEL_DIR, 'vae_encoder.keras')
decoder_path = os.path.join(MODEL_DIR, 'vae_decoder.keras')
config_path = os.path.join(MODEL_DIR, 'soar_config.json')

# Biến toàn cục lưu trữ trạng thái hệ thống
scaler = None
iso_forest = None
encoder = None
decoder = None
vae_threshold = 0.0
feature_cols = []
contamination_ratio = 0.0

stream_df = None
stream_index = 0
blocklist = {} # IP -> Timestamp blocked
playbook_logs = []
history_logs = []
total_scanned = 0
anomalies_detected = 0
is_running = True

# ---- Định nghĩa lại Lớp Sampling Custom cho Keras Deserialization ----
class Sampling(layers.Layer):
    """Lớp lấy mẫu z = mu + sigma * epsilon (Reparameterization trick)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Nạp mô hình khi khởi động
def init_models():
    global scaler, iso_forest, encoder, decoder, vae_threshold, feature_cols, contamination_ratio, stream_df
    
    if not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
        print("[⚠️ WARNING] Chưa tìm thấy mô hình. Hãy đảm bảo đã chạy Notebook trước.")
        return False
        
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(if_path, 'rb') as f:
            iso_forest = pickle.load(f)
        with open(config_path, 'r', encoding='utf-8') as f:
            soar_config = json.load(f)
            
        # Nạp VAE Encoder kèm theo lớp custom Sampling
        encoder = tf.keras.models.load_model(
            encoder_path, 
            custom_objects={'Sampling': Sampling}, 
            compile=False
        )
        decoder = tf.keras.models.load_model(
            decoder_path, 
            compile=False
        )
        
        vae_threshold = soar_config['vae_threshold']
        feature_cols = soar_config['feature_cols']
        contamination_ratio = soar_config['contamination_ratio']
        
        # Đọc dữ liệu log để chuẩn bị stream
        df = pd.read_csv('lms_training_dataset_final.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['ip', 'timestamp']).reset_index(drop=True)
        df['time_gap'] = df.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(300.0)

        df_idx = df.set_index('timestamp')
        mean_gap = df_idx.groupby('ip')['time_gap'].rolling('5min', min_periods=1).mean().reset_index()
        std_gap = df_idx.groupby('ip')['time_gap'].rolling('5min', min_periods=1).std().reset_index()

        df['time_gap_per_IP'] = mean_gap['time_gap'].values
        df['req_regularity_per_IP'] = std_gap['time_gap'].fillna(300.0).values
        df['hour_of_day'] = df['timestamp'].dt.hour
        
        # Trích xuất tập giả lập (Validation + Test)
        normal_df = df[df['Label'] == 0]
        malicious_df = df[df['Label'] == 1]
        _, val_normal_df = np.split(normal_df, [int(0.8*len(normal_df))])
        
        # Trộn ngẫu nhiên
        stream_df = pd.concat([val_normal_df, malicious_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print("[✓] Mô hình AI và dữ liệu log đã được nạp thành công vào SOAR!")
        return True
    except Exception as e:
        print(f"[❌ ERROR] Lỗi nạp mô hình: {e}")
        return False

# Tính lỗi giải nén MAE VAE
def get_vae_mae(x_scaled):
    if encoder is None or decoder is None:
        return 0.0
    z_mean, _, _ = encoder(x_scaled, training=False)
    reconstruction = decoder(z_mean, training=False)
    mae = np.mean(np.abs(x_scaled - reconstruction.numpy()), axis=1)
    return float(mae[0])

# Thêm log playbook
def add_playbook_log(message):
    global playbook_logs
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    playbook_logs.insert(0, log_msg)
    if len(playbook_logs) > 40:
        playbook_logs.pop()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global stream_index, total_scanned, anomalies_detected, is_running
    
    # Giả lập xử lý log mới nếu đang chạy
    new_events = []
    if is_running and stream_df is not None and stream_index < len(stream_df):
        # Lấy 1 log tiếp theo để xử lý
        row = stream_df.iloc[stream_index]
        stream_index += 1
        total_scanned += 1
        
        ip = row['ip']
        timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        actual_label = int(row['Label'])
        
        # Trích xuất vector đặc trưng
        feature_vector = row[feature_cols].values.reshape(1, -1).astype(np.float32)
        feature_vector_scaled = scaler.transform(feature_vector)
        
        is_blocked = ip in blocklist
        vae_error = 0.0
        is_vae_anomaly = False
        is_if_anomaly = False
        
        # Nếu chưa bị block, chạy AI kiểm tra
        if not is_blocked:
            vae_error = get_vae_mae(feature_vector_scaled)
            is_vae_anomaly = vae_error > vae_threshold
            
            # isolation forest baseline
            if_pred = iso_forest.predict(feature_vector_scaled)[0]
            is_if_anomaly = if_pred == -1
            
            # Kích hoạt hành động của SOAR
            if is_vae_anomaly:
                anomalies_detected += 1
                blocklist[ip] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                is_blocked = True
                
                # Chi tiết Playbook phản ứng tự động
                add_playbook_log(f"🚨 CRITICAL ALERT: Phát hiện Credential Stuffing từ IP {ip}!")
                add_playbook_log(f"   [Chi tiết] VAE MAE Error: {vae_error:.5f} (Ngưỡng: {vae_threshold:.5f})")
                add_playbook_log(f"   🛡️ SOAR PLAYBOOK: Kích hoạt Kịch bản ngăn chặn khẩn cấp...")
                add_playbook_log(f"   ↳ [FIREWALL] Đã nạp quy tắc tự động chặn IP {ip} vào Firewall.")
                add_playbook_log(f"   ↳ [MOODLE API] Gửi API khóa tạm thời 3 tài khoản đăng nhập sai từ IP này.")
                add_playbook_log(f"   ↳ [ALERT] Gửi thẻ báo động SOC đến Slack channel #soc-incident-alerts.")
                add_playbook_log(f"   ↳ [LOG] Đã ghi nhận sự kiện an toàn vào soar_security_events.log.")
                
                # Ghi log sự kiện ra file
                with open('soar_security_events.log', 'a', encoding='utf-8') as lf:
                    lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SOAR BLOCK: IP={ip}, MAE={vae_error:.6f}, Threshold={vae_threshold:.6f}\n")
            
            elif is_if_anomaly:
                # Cảnh báo Isolation Forest
                add_playbook_log(f"⚠️ WARNING: Isolation Forest phát hiện hành vi bất thường từ IP {ip}.")
                add_playbook_log(f"   🛡️ SOAR PLAYBOOK: Đang tăng mức giám sát log (Verbose mode) cho IP này.")
                add_playbook_log(f"   ↳ IP {ip} được đưa vào hàng đợi đánh giá SOC.")
        
        event_data = {
            "timestamp": timestamp_str,
            "ip": ip,
            "vae_error": vae_error,
            "is_vae_anomaly": is_vae_anomaly,
            "is_if_anomaly": is_if_anomaly,
            "is_blocked": is_blocked,
            "label": actual_label,
            "features": {col: float(row[col]) for col in feature_cols}
        }
        new_events.append(event_data)
        history_logs.insert(0, event_data)
        if len(history_logs) > 100:
            history_logs.pop()
            
    # Format danh sách blocklist để trả về
    blocklist_list = [{"ip": k, "time": v} for k, v in blocklist.items()]
    
    return jsonify({
        "is_running": is_running,
        "metrics": {
            "total_scanned": total_scanned,
            "anomalies_detected": anomalies_detected,
            "active_blocked_ips": len(blocklist),
            "vae_threshold": vae_threshold,
            "contamination_ratio": contamination_ratio
        },
        "new_events": new_events,
        "blocklist": blocklist_list,
        "playbook_logs": playbook_logs
    })

@app.route('/api/toggle', methods=['POST'])
def toggle_stream():
    global is_running
    is_running = not is_running
    add_playbook_log(f"⚙️ SYSTEM: Đã {'BẮT ĐẦU' if is_running else 'TẠM DỪNG'} kịch bản quét live logs.")
    return jsonify({"is_running": is_running})

@app.route('/api/reset-blocklist', methods=['POST'])
def reset_blocklist():
    global blocklist, anomalies_detected
    blocklist.clear()
    add_playbook_log("⚙️ SYSTEM: Đã xóa toàn bộ IP khỏi Blocklist và khôi phục cài đặt gốc Firewall.")
    return jsonify({"success": True})

@app.route('/api/manual-block', methods=['POST'])
def manual_block():
    global blocklist
    data = request.get_json()
    ip = data.get('ip')
    if ip:
        blocklist[ip] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        add_playbook_log(f"🛡️ MANUAL ACTION: Quản trị viên chủ động thêm IP {ip} vào Blocklist.")
        add_playbook_log(f"   ↳ [FIREWALL] Cập nhật Firewall chặn IP {ip} lập tức.")
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "IP is required"}), 400

@app.route('/api/manual-unblock', methods=['POST'])
def manual_unblock():
    global blocklist
    data = request.get_json()
    ip = data.get('ip')
    if ip in blocklist:
        del blocklist[ip]
        add_playbook_log(f"🛡️ MANUAL ACTION: Quản trị viên gỡ bỏ IP {ip} khỏi danh sách hạn chế.")
        add_playbook_log(f"   ↳ [FIREWALL] Gỡ bỏ luật cấm IP {ip} khỏi Firewall.")
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "IP not in blocklist"}), 400

if __name__ == '__main__':
    # Nạp mô hình
    has_models = init_models()
    if not has_models:
        print("\n[⚠️ WARNING] Không thể chạy chế độ AI vì thiếu file mô hình!")
        print("Vui lòng huấn luyện mô hình trong file Jupyter Notebook trước nhé.")
        sys.exit(1)
        
    print("\n=======================================================")
    print(" HỆ THỐNG SOAR TRỰC QUAN ĐÃ SẴN SÀNG KHỞI CHẠY!")
    print(" 👉 Mở trình duyệt truy cập: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
