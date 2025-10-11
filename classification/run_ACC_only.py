import numpy as np
import os
from scipy.stats import zscore

# === Đường dẫn đến file ACS bạn vừa sinh ===
prefix = "./mpnet_acs_SetFit_sst2/"
train_path = os.path.join(prefix, "concept_labels_train.npy")
val_path = os.path.join(prefix, "concept_labels_val.npy")

train_scores = np.load(train_path)
val_scores = np.load(val_path)

# === Bước 1: Chuẩn hóa giá trị (z-score normalization) ===
train_corr = zscore(train_scores, axis=0)
val_corr = zscore(val_scores, axis=0)

# === Bước 2: Cắt giá trị ngoài [-3, 3] (optional) ===
train_corr = np.clip(train_corr, -3, 3)
val_corr = np.clip(val_corr, -3, 3)

# === Bước 3: Chuẩn hóa về [0, 1] để interpret dễ hơn ===
train_corr = (train_corr - train_corr.min()) / (train_corr.max() - train_corr.min())
val_corr = (val_corr - val_corr.min()) / (val_corr.max() - val_corr.min())

# === Lưu file ACC (corrected) ===
os.makedirs("./acc_outputs", exist_ok=True)
np.save("./acc_outputs/concept_labels_train_ACC.npy", train_corr)
np.save("./acc_outputs/concept_labels_val_ACC.npy", val_corr)

print("✅ Saved corrected ACC files to ./acc_outputs/")
