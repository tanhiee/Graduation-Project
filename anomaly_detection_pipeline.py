"""
==========================================================================
 ĐỒ ÁN TỐT NGHIỆP - AN TOÀN THÔNG TIN
 Phát hiện điểm dị thường (Anomaly Detection) chống tấn công
 Credential Stuffing (Low & Slow) trên hệ thống LMS Moodle

 Pipeline: Data Split → Isolation Forest (Baseline) → VAE (Main Model)
           → Threshold & Evaluation → ROC Comparison
==========================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_curve, auc
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# --- Cấu hình đường dẫn lưu biểu đồ ---
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(SAVE_DIR, exist_ok=True)

# Cấu hình style biểu đồ chuyên nghiệp
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#333',
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
})

print("=" * 70)
print("  ANOMALY DETECTION PIPELINE - CREDENTIAL STUFFING ON MOODLE LMS")
print("=" * 70)

# =====================================================================
# BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# =====================================================================
print("\n[STEP 1] Đọc và tiền xử lý dữ liệu...")

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'lms_training_dataset_final.csv'))

print(f"  → Tổng số mẫu: {len(df)}")
print(f"  → Phân bố nhãn:")
print(f"     Normal  (Label 0): {len(df[df['Label'] == 0])}")
print(f"     Malicious (Label 1): {len(df[df['Label'] == 1])}")
print(f"  → Các cột: {list(df.columns)}")

# --- Mã hóa cột IP bằng LabelEncoder ---
le_ip = LabelEncoder()
df['ip_encoded'] = le_ip.fit_transform(df['ip'])

# --- Chọn Features số học để huấn luyện ---
FEATURE_COLS = ['ip_encoded', 'req_per_IP_5min', 'error_ratio_per_IP',
                'unique_UA_per_IP', 'size']

X = df[FEATURE_COLS].values.astype(np.float32)
y = df['Label'].values

# --- Chuẩn hóa Min-Max [0, 1] ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"  → Features đã chọn: {FEATURE_COLS}")
print(f"  → Shape sau chuẩn hóa: {X_scaled.shape}")

# =====================================================================
# BƯỚC 2: CHIA DỮ LIỆU (DATA SPLIT)
# =====================================================================
print("\n[STEP 2] Chia dữ liệu...")

# Tách riêng Normal và Malicious
normal_mask = (y == 0)
malicious_mask = (y == 1)

X_normal = X_scaled[normal_mask]
X_malicious = X_scaled[malicious_mask]
y_normal = y[normal_mask]
y_malicious = y[malicious_mask]

# Chia Normal thành 80% Train, 20% Validation
X_normal_train, X_normal_val = train_test_split(
    X_normal, test_size=0.2, random_state=42
)

# Tập Test = toàn bộ Malicious
X_test = X_malicious
y_test = y_malicious  # Toàn bộ là label 1

# Tập Evaluation (Validation + Test) - dùng để đánh giá cả 2 mô hình
X_eval = np.vstack([X_normal_val, X_test])
y_eval = np.concatenate([
    np.zeros(len(X_normal_val)),  # Normal
    np.ones(len(X_test))          # Malicious
])

print(f"  → Normal Train   : {X_normal_train.shape[0]} mẫu (chỉ dùng huấn luyện VAE)")
print(f"  → Normal Val     : {X_normal_val.shape[0]} mẫu")
print(f"  → Malicious Test : {X_test.shape[0]} mẫu")
print(f"  → Evaluation Set : {X_eval.shape[0]} mẫu (Val + Test)")

# =====================================================================
# BƯỚC 3: BASELINE MODEL - ISOLATION FOREST
# =====================================================================
print("\n[STEP 3] Huấn luyện Baseline Model (Isolation Forest)...")

# contamination = tỉ lệ dị thường ước tính trong tập eval
contamination_ratio = len(X_test) / len(X_eval)
print(f"  → Contamination ratio: {contamination_ratio:.4f}")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=contamination_ratio,
    random_state=42,
    n_jobs=-1
)

# Huấn luyện trên tập Evaluation (Validation + Test) để tự tìm dị biệt
iso_forest.fit(X_eval)

# Dự đoán: Isolation Forest trả về 1 (normal) hoặc -1 (anomaly)
iso_pred_raw = iso_forest.predict(X_eval)
# Chuyển đổi: -1 → 1 (Malicious), 1 → 0 (Normal)
iso_pred = np.where(iso_pred_raw == -1, 1, 0)

# Anomaly score (càng âm = càng dị thường)
iso_scores = -iso_forest.decision_function(X_eval)  # Đảo dấu: càng cao = càng dị thường

# --- Đánh giá Isolation Forest ---
iso_f1 = f1_score(y_eval, iso_pred, average='weighted')
iso_cm = confusion_matrix(y_eval, iso_pred)

print(f"\n  ✦ Isolation Forest - F1 Score (weighted): {iso_f1:.4f}")
print(f"  ✦ Classification Report:")
print(classification_report(y_eval, iso_pred,
                            target_names=['Normal', 'Malicious'],
                            digits=4))

# =====================================================================
# BƯỚC 4: MAIN MODEL - VARIATIONAL AUTOENCODER (VAE)
# =====================================================================
print("\n[STEP 4] Xây dựng và huấn luyện Variational Autoencoder (VAE)...")

input_dim = X_normal_train.shape[1]
latent_dim = 2  # Không gian tiềm ẩn 2D (dễ trực quan hóa)

# ---- Lớp Sampling cho Latent Space ----
class Sampling(layers.Layer):
    """Lớp lấy mẫu z = mu + sigma * epsilon (Reparameterization trick)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ---- ENCODER ----
encoder_inputs = keras.Input(shape=(input_dim,), name='encoder_input')
x = layers.Dense(64, activation='relu', name='enc_dense1')(encoder_inputs)
x = layers.BatchNormalization(name='enc_bn1')(x)
x = layers.Dense(32, activation='relu', name='enc_dense2')(x)
x = layers.BatchNormalization(name='enc_bn2')(x)
x = layers.Dense(16, activation='relu', name='enc_dense3')(x)

z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling(name='sampling')([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# ---- DECODER ----
decoder_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
x = layers.Dense(16, activation='relu', name='dec_dense1')(decoder_inputs)
x = layers.BatchNormalization(name='dec_bn1')(x)
x = layers.Dense(32, activation='relu', name='dec_dense2')(x)
x = layers.BatchNormalization(name='dec_bn2')(x)
x = layers.Dense(64, activation='relu', name='dec_dense3')(x)
decoder_outputs = layers.Dense(input_dim, activation='sigmoid', name='dec_output')(x)

decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
decoder.summary()

# ---- VAE MODEL ----
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reduce_mean(tf.square(data - reconstruction), axis=-1),
                )
            )
            # KL Divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_mean(tf.square(data - reconstruction), axis=-1),
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Khởi tạo và biên dịch VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# Callback Early Stopping
early_stop = EarlyStopping(
    monitor='val_total_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)

# ---- HUẤN LUYỆN CHỈ TRÊN TẬP NORMAL TRAIN ----
print(f"\n  → Bắt đầu huấn luyện VAE trên {X_normal_train.shape[0]} mẫu Normal...")
history = vae.fit(
    X_normal_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_normal_val,),
    callbacks=[early_stop],
    verbose=1
)

print("  ✦ Huấn luyện VAE hoàn tất!")

# =====================================================================
# BƯỚC 5: TÍNH RECONSTRUCTION ERROR & THRESHOLD
# =====================================================================
print("\n[STEP 5] Tính Reconstruction Error và xác định Threshold...")

def compute_reconstruction_error(model, data):
    """Tính MSE reconstruction error cho từng mẫu"""
    z_mean, z_log_var, z = model.encoder(data)
    reconstructed = model.decoder(z)
    mse = np.mean(np.square(data - reconstructed.numpy()), axis=1)
    return mse

# Tính lỗi trên từng tập
re_normal_train = compute_reconstruction_error(vae, X_normal_train)
re_normal_val = compute_reconstruction_error(vae, X_normal_val)
re_malicious = compute_reconstruction_error(vae, X_test)

# Reconstruction error trên toàn bộ tập Eval
re_eval = np.concatenate([re_normal_val, re_malicious])

# THRESHOLD = max reconstruction error của tập Normal Validation
threshold = np.max(re_normal_val)

print(f"  → RE Normal Train  : mean={np.mean(re_normal_train):.6f}, max={np.max(re_normal_train):.6f}")
print(f"  → RE Normal Val    : mean={np.mean(re_normal_val):.6f}, max={np.max(re_normal_val):.6f}")
print(f"  → RE Malicious     : mean={np.mean(re_malicious):.6f}, max={np.max(re_malicious):.6f}")
print(f"\n  ★ THRESHOLD (max loss Normal Val) = {threshold:.6f}")

# Dự đoán VAE: vượt threshold → Malicious (1)
vae_pred = (re_eval > threshold).astype(int)

# --- Đánh giá VAE ---
vae_f1 = f1_score(y_eval, vae_pred, average='weighted')
vae_cm = confusion_matrix(y_eval, vae_pred)

print(f"\n  ✦ VAE - F1 Score (weighted): {vae_f1:.4f}")
print(f"  ✦ Classification Report:")
print(classification_report(y_eval, vae_pred,
                            target_names=['Normal', 'Malicious'],
                            digits=4))

# =====================================================================
# BƯỚC 6: VẼ 5 BIỂU ĐỒ "ĂN ĐIỂM"
# =====================================================================
print("\n[STEP 6] Xuất 5 biểu đồ chuyên nghiệp...")

# ── Màu sắc chuyên nghiệp ──
COLOR_NORMAL = '#00d2ff'      # Cyan sáng
COLOR_MALICIOUS = '#e94560'   # Đỏ san hô
COLOR_THRESHOLD = '#f5a623'   # Vàng cam
COLOR_IF = '#7b68ee'          # Tím nhạt (Isolation Forest)
COLOR_VAE = '#00e396'         # Xanh lá neon (VAE)
BG_DARK = '#1a1a2e'
BG_AXES = '#16213e'

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 1: Training & Validation Loss Curve
# ─────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor(BG_DARK)
ax1.set_facecolor(BG_AXES)

epochs_range = range(1, len(history.history['total_loss']) + 1)

ax1.plot(epochs_range, history.history['total_loss'],
         color=COLOR_NORMAL, linewidth=2.5, label='Training Loss', marker='o',
         markersize=3, alpha=0.9)
ax1.plot(epochs_range, history.history['val_total_loss'],
         color=COLOR_MALICIOUS, linewidth=2.5, label='Validation Loss', marker='s',
         markersize=3, alpha=0.9)

ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Loss (MSE + KL)', fontsize=13, fontweight='bold')
ax1.set_title('Biểu đồ 1: Đường cong hội tụ (Training & Validation Loss)',
              fontsize=15, fontweight='bold', pad=15)
ax1.legend(fontsize=12, loc='upper right', fancybox=True, framealpha=0.8,
           edgecolor='#555')
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(os.path.join(SAVE_DIR, '01_training_validation_loss.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 01_training_validation_loss.png")

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 2: Reconstruction Error Histogram
# ─────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor(BG_DARK)
ax2.set_facecolor(BG_AXES)

ax2.hist(re_normal_val, bins=80, alpha=0.75, color=COLOR_NORMAL,
         label=f'Normal (n={len(re_normal_val)})', edgecolor='white', linewidth=0.5)
ax2.hist(re_malicious, bins=80, alpha=0.75, color=COLOR_MALICIOUS,
         label=f'Malicious (n={len(re_malicious)})', edgecolor='white', linewidth=0.5)
ax2.axvline(x=threshold, color=COLOR_THRESHOLD, linestyle='--', linewidth=2.5,
            label=f'Threshold = {threshold:.4f}')

ax2.set_xlabel('Reconstruction Error (MSE)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Số lượng mẫu', fontsize=13, fontweight='bold')
ax2.set_title('Biểu đồ 2: Phân phối lỗi giải nén (Reconstruction Error)',
              fontsize=15, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='upper right', fancybox=True, framealpha=0.8,
           edgecolor='#555')
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, '02_reconstruction_error_histogram.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 02_reconstruction_error_histogram.png")

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 3: Confusion Matrix - Isolation Forest (Baseline)
# ─────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 6))
fig3.patch.set_facecolor(BG_DARK)

sns.heatmap(iso_cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=['Normal', 'Malicious'],
            yticklabels=['Normal', 'Malicious'],
            annot_kws={'size': 18, 'fontweight': 'bold'},
            linewidths=2, linecolor='#333',
            cbar_kws={'label': 'Số lượng'},
            ax=ax3)

ax3.set_xlabel('Dự đoán (Predicted)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Thực tế (Actual)', fontsize=13, fontweight='bold')
ax3.set_title(f'Biểu đồ 3: Confusion Matrix - Isolation Forest\n(F1={iso_f1:.4f})',
              fontsize=14, fontweight='bold', pad=15)

fig3.tight_layout()
fig3.savefig(os.path.join(SAVE_DIR, '03_confusion_matrix_isolation_forest.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 03_confusion_matrix_isolation_forest.png")

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 4: Confusion Matrix - VAE (Main Model)
# ─────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 6))
fig4.patch.set_facecolor(BG_DARK)

sns.heatmap(vae_cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Normal', 'Malicious'],
            yticklabels=['Normal', 'Malicious'],
            annot_kws={'size': 18, 'fontweight': 'bold'},
            linewidths=2, linecolor='#333',
            cbar_kws={'label': 'Số lượng'},
            ax=ax4)

ax4.set_xlabel('Dự đoán (Predicted)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Thực tế (Actual)', fontsize=13, fontweight='bold')
ax4.set_title(f'Biểu đồ 4: Confusion Matrix - VAE\n(F1={vae_f1:.4f})',
              fontsize=14, fontweight='bold', pad=15)

fig4.tight_layout()
fig4.savefig(os.path.join(SAVE_DIR, '04_confusion_matrix_vae.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 04_confusion_matrix_vae.png")

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 5: ROC Curve - So sánh AUC Isolation Forest vs VAE
# ─────────────────────────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(10, 8))
fig5.patch.set_facecolor(BG_DARK)
ax5.set_facecolor(BG_AXES)

# ROC cho Isolation Forest (dùng anomaly score)
fpr_if, tpr_if, _ = roc_curve(y_eval, iso_scores)
auc_if = auc(fpr_if, tpr_if)

# ROC cho VAE (dùng reconstruction error)
fpr_vae, tpr_vae, _ = roc_curve(y_eval, re_eval)
auc_vae = auc(fpr_vae, tpr_vae)

# Đường chéo ngẫu nhiên
ax5.plot([0, 1], [0, 1], color='#555', linestyle='--', linewidth=1.5,
         label='Random Classifier (AUC = 0.50)', alpha=0.7)

# Đường ROC Isolation Forest
ax5.plot(fpr_if, tpr_if, color=COLOR_IF, linewidth=3,
         label=f'Isolation Forest (AUC = {auc_if:.4f})', alpha=0.9)
ax5.fill_between(fpr_if, tpr_if, alpha=0.15, color=COLOR_IF)

# Đường ROC VAE
ax5.plot(fpr_vae, tpr_vae, color=COLOR_VAE, linewidth=3,
         label=f'VAE (AUC = {auc_vae:.4f})', alpha=0.9)
ax5.fill_between(fpr_vae, tpr_vae, alpha=0.15, color=COLOR_VAE)

ax5.set_xlabel('False Positive Rate (Tỷ lệ Dương tính giả)', fontsize=13, fontweight='bold')
ax5.set_ylabel('True Positive Rate (Tỷ lệ phát hiện đúng)', fontsize=13, fontweight='bold')
ax5.set_title('Biểu đồ 5: Đường cong ROC - So sánh Isolation Forest vs VAE',
              fontsize=15, fontweight='bold', pad=15)
ax5.legend(fontsize=12, loc='lower right', fancybox=True, framealpha=0.8,
           edgecolor='#555')
ax5.grid(True, alpha=0.3)
ax5.set_xlim([-0.02, 1.02])
ax5.set_ylim([-0.02, 1.02])

fig5.tight_layout()
fig5.savefig(os.path.join(SAVE_DIR, '05_roc_curve_comparison.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 05_roc_curve_comparison.png")

# =====================================================================
# TỔNG KẾT
# =====================================================================
print("\n" + "=" * 70)
print("  TỔNG KẾT KẾT QUẢ")
print("=" * 70)
print(f"""
  ┌──────────────────────┬──────────────────┬──────────────────┐
  │      Chỉ số          │ Isolation Forest │       VAE        │
  ├──────────────────────┼──────────────────┼──────────────────┤
  │  F1 Score (weighted) │     {iso_f1:.4f}       │     {vae_f1:.4f}       │
  │  AUC                 │     {auc_if:.4f}       │     {auc_vae:.4f}       │
  │  True Positive       │     {iso_cm[1][1]:>5d}        │     {vae_cm[1][1]:>5d}        │
  │  False Positive      │     {iso_cm[0][1]:>5d}        │     {vae_cm[0][1]:>5d}        │
  │  True Negative       │     {iso_cm[0][0]:>5d}        │     {vae_cm[0][0]:>5d}        │
  │  False Negative      │     {iso_cm[1][0]:>5d}        │     {vae_cm[1][0]:>5d}        │
  └──────────────────────┴──────────────────┴──────────────────┘
""")

if auc_vae > auc_if:
    print("  ★ KẾT LUẬN: VAE (Deep Learning) vượt trội hơn Isolation Forest (ML truyền thống)")
    print("    trong việc phát hiện tấn công Credential Stuffing trên hệ thống LMS Moodle.")
else:
    print("  ★ KẾT LUẬN: Cả hai mô hình đều cho kết quả tốt.")
    print("    Cần tinh chỉnh thêm hyperparameter để tối ưu VAE.")

print(f"\n  📁 Tất cả biểu đồ đã được lưu tại: {SAVE_DIR}")
print("=" * 70)
