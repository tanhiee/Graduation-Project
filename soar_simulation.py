import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Định nghĩa màu sắc ANSI để hiển thị terminal đẹp mắt
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

print(f"{Colors.BLUE}{Colors.BOLD}==========================================================================")
print("     HỆ THỐNG SOAR PHÁT HIỆN & PHẢN ỨNG TỰ ĐỘNG CREDENTIAL STUFFING")
print(f"=========================================================================={Colors.RESET}\n")

# 1. Kiểm tra sự tồn tại của các mô hình đã train
MODEL_DIR = 'deploy_models'
scaler_path = os.path.join(MODEL_DIR, 'minmax_scaler.pkl')
if_path = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
encoder_path = os.path.join(MODEL_DIR, 'vae_encoder.keras')
decoder_path = os.path.join(MODEL_DIR, 'vae_decoder.keras')
config_path = os.path.join(MODEL_DIR, 'soar_config.json')

missing_files = []
for p in [scaler_path, if_path, encoder_path, decoder_path, config_path]:
    if not os.path.exists(p):
        missing_files.append(p)

if missing_files:
    print(f"{Colors.RED}{Colors.BOLD}[❌ LỖI] Không tìm thấy các file mô hình đã huấn luyện:{Colors.RESET}")
    for mf in missing_files:
        print(f"   - {mf}")
    print(f"\n{Colors.YELLOW}👉 Vui lòng chạy toàn bộ file Jupyter Notebook 'anomaly_detection_pipeline.ipynb' trước")
    print(f"   để huấn luyện và tự động lưu các mô hình vào thư mục '{MODEL_DIR}'.{Colors.RESET}")
    sys.exit(1)

# 2. Nạp các mô hình và cấu hình
print(f"{Colors.GREEN}[*] Đang nạp các mô hình học máy và cấu hình SOAR...{Colors.RESET}")

# Tắt log TensorFlow để tránh rác màn hình
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(if_path, 'rb') as f:
    iso_forest = pickle.load(f)
with open(config_path, 'r', encoding='utf-8') as f:
    soar_config = json.load(f)

# Nạp VAE Encoder & Decoder
encoder = tf.keras.models.load_model(encoder_path, compile=False)
decoder = tf.keras.models.load_model(decoder_path, compile=False)

vae_threshold = soar_config['vae_threshold']
feature_cols = soar_config['feature_cols']

print(f"{Colors.GREEN}[✓] Nạp mô hình thành công!{Colors.RESET}")
print(f"    - Ngưỡng VAE Threshold: {Colors.BOLD}{vae_threshold:.6f}{Colors.RESET}")
print(f"    - Các đặc trưng giám sát: {Colors.BOLD}{feature_cols}{Colors.RESET}\n")

# 3. Đọc dữ liệu log để giả lập live log stream
print(f"{Colors.GREEN}[*] Đang nạp tập dữ liệu log để giả lập live stream...{Colors.RESET}")
df = pd.read_csv('lms_training_dataset_final.csv')

# Áp dụng Feature Engineering giống hệt trong notebook
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['ip', 'timestamp']).reset_index(drop=True)
df['time_gap'] = df.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(300.0)

df_idx = df.set_index('timestamp')
mean_gap = df_idx.groupby('ip')['time_gap'].rolling('5min', min_periods=1).mean().reset_index()
std_gap = df_idx.groupby('ip')['time_gap'].rolling('5min', min_periods=1).std().reset_index()

df['time_gap_per_IP'] = mean_gap['time_gap'].values
df['req_regularity_per_IP'] = std_gap['time_gap'].fillna(300.0).values
df['hour_of_day'] = df['timestamp'].dt.hour

# Trích xuất tập Evaluation để mô phỏng (Val + Test)
# Lọc lấy tập Malicious và 20% Normal làm tập giả lập live stream
normal_df = df[df['Label'] == 0]
malicious_df = df[df['Label'] == 1]
# Lấy 20% Normal cuối làm tập Validation giả lập
_, val_normal_df = np.split(normal_df, [int(0.8*len(normal_df))])

stream_df = pd.concat([val_normal_df, malicious_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"{Colors.GREEN}[✓] Đã tạo tập dữ liệu live stream gồm {Colors.BOLD}{len(stream_df)} logs{Colors.RESET} (bao gồm cả Bình thường & Tấn công).")
print(f"{Colors.CYAN}>>> HỆ THỐNG SOAR ĐÃ BẮT ĐẦU HOẠT ĐỘNG. ĐANG QUÉT LIVE TRAFFIC LOGS... <<<{Colors.RESET}\n")
time.sleep(2.0)

# Khởi tạo Blocklist lưu trữ các IP đã bị SOAR tự động khóa
blocklist = set()

# Hàm tính lỗi giải nén MAE cho VAE
def get_vae_mae_error(x_input):
    z_mean, _, _ = encoder(x_input, training=False)
    reconstruction = decoder(z_mean, training=False)
    # Lỗi MAE trung bình theo chiều đặc trưng
    mae = np.mean(np.abs(x_input - reconstruction.numpy()), axis=1)
    return mae[0]

# 4. Vòng lặp mô phỏng Real-Time Event Loop của SOAR
try:
    for idx, row in stream_df.iterrows():
        ip = row['ip']
        timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        actual_label = row['Label']
        
        # Trích xuất vector đặc trưng và chuẩn hóa MinMaxScaler
        feature_vector = row[feature_cols].values.reshape(1, -1).astype(np.float32)
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # --- BƯỚC A: Kiểm tra xem IP này đã nằm trong Blocklist của SOAR chưa ---
        if ip in blocklist:
            print(f"[{timestamp_str}] IP {Colors.RED}{ip:<15}{Colors.RESET} | STATUS: {Colors.RED}{Colors.BOLD}BLOCKED{Colors.RESET} | {Colors.RED}Log bị Drop tự động bởi SOAR Firewall Playbook.{Colors.RESET}")
            time.sleep(0.2)
            continue
            
        # --- BƯỚC B: Chạy qua mô hình học sâu VAE để tính lỗi giải nén ---
        vae_error = get_vae_mae_error(feature_vector_scaled)
        
        # --- BƯỚC C: Chạy qua mô hình Isolation Forest làm baseline ---
        if_pred = iso_forest.predict(feature_vector_scaled)[0] # 1 là normal, -1 là anomaly
        
        is_vae_anomaly = vae_error > vae_threshold
        is_if_anomaly = if_pred == -1
        
        # --- BƯỚC D: Xử lý và Phản ứng tự động theo Kịch bản SOAR (Playbook Execution) ---
        if is_vae_anomaly:
            # 🚨 ĐÃ PHÁT HIỆN TẤN CÔNG CREDENTIAL STUFFING BẰNG VAE
            print(f"[{timestamp_str}] IP {Colors.YELLOW}{ip:<15}{Colors.RESET} | VAE MAE: {Colors.RED}{vae_error:.6f}{Colors.RESET} | STATUS: {Colors.RED}{Colors.BOLD}🚨 CRITICAL ANOMALY!{Colors.RESET}")
            print(f"  {Colors.RED}↳ [!] Phát hiện hành vi Credential Stuffing (Low & Slow) với độ đều đặn cao.{Colors.RESET}")
            
            # Kích hoạt Playbook tự động chặn đứng cuộc tấn công (Incident Containment Playbook)
            print(f"  {Colors.BOLD}{Colors.CYAN}[🛡️ SOAR PLAYBOOK ACTIVATED - INCIDENT CONTAINMENT]{Colors.RESET}")
            print(f"    - Step 1: Cấu hình Firewall rule tự động chặn IP {Colors.BOLD}{ip}{Colors.RESET}")
            blocklist.add(ip)
            print(f"    - Step 2: Gọi Moodle API khóa tạm thời các tài khoản liên quan đến IP này")
            print(f"    - Step 3: Gửi báo động khẩn cấp tới kênh Slack {Colors.BOLD}#soc-critical-alerts{Colors.RESET}")
            print(f"    - Step 4: Ghi nhận sự kiện vào file {Colors.BOLD}soar_security_events.log{Colors.RESET}")
            
            # Ghi log sự kiện
            with open('soar_security_events.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT: Credential Stuffing blocked! IP={ip}, MAE={vae_error:.6f}, Threshold={vae_threshold:.6f}\n")
            
            print(f"  {Colors.GREEN}[✓] Playbook kết thúc thành công. IP {ip} đã bị cách ly!{Colors.RESET}\n")
            time.sleep(1.0) # Pause lâu hơn một chút khi có sự kiện tấn công để dễ quan sát
            
        elif is_if_anomaly:
            # ⚠️ Isolation Forest báo động nhưng VAE không báo động (Cảnh báo nghi ngờ)
            print(f"[{timestamp_str}] IP {Colors.CYAN}{ip:<15}{Colors.RESET} | VAE MAE: {vae_error:.6f} | STATUS: {Colors.YELLOW}{Colors.BOLD}⚠️ WARNING (IF Anomaly){Colors.RESET}")
            print(f"  {Colors.YELLOW}↳ [!] Isolation Forest phát hiện dấu hiệu bất thường nhẹ. VAE đánh giá an toàn.{Colors.RESET}")
            print(f"  {Colors.BOLD}{Colors.CYAN}[🔍 SOAR PLAYBOOK ACTIVATED - SUSPICIOUS INVESTIGATION]{Colors.RESET}")
            print(f"    - Step 1: Nâng cao cấp độ giám sát log (Verbose logging) cho IP {ip}")
            print(f"    - Step 2: Đánh dấu IP này vào hàng đợi xem xét của Chuyên viên SOC")
            print(f"  {Colors.GREEN}[✓] Playbook kết thúc. IP đang được theo dõi chặt chẽ.{Colors.RESET}\n")
            time.sleep(0.5)
            
        else:
            # ✅ LOG HOÀN TOÀN BÌNH THƯỜNG
            print(f"[{timestamp_str}] IP {Colors.GREEN}{ip:<15}{Colors.RESET} | VAE MAE: {vae_error:.6f} | STATUS: {Colors.GREEN}✅ Normal Request{Colors.RESET}")
            time.sleep(0.1) # Logs bình thường chạy nhanh

except KeyboardInterrupt:
    print(f"\n{Colors.YELLOW}[!] Đang dừng hệ thống SOAR...{Colors.RESET}")

print(f"\n{Colors.BLUE}{Colors.BOLD}==========================================================================")
print(f"               HỆ THỐNG SOAR ĐÃ DỪNG HOẠT ĐỘNG AN TOÀN")
print(f"   Tổng số IP bị chặn tự động trong phiên: {len(blocklist)}")
print(f"   Blocklist IP: {list(blocklist)}")
print(f"=========================================================================={Colors.RESET}")
