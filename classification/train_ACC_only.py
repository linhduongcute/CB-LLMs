import os
import time
import numpy as np
from datasets import load_dataset
import config as CFG
from utils import get_labels

# === Cấu hình cơ bản ===
dataset = "SetFit/sst2"
labeling = "mpnet"  # mpnet, angle, simcse, llm

train_dataset = load_dataset(dataset, split='train')
val_dataset = load_dataset(dataset, split='validation')

concept_set = CFG.concept_set[dataset]
d_name = dataset.replace('/', '_')

prefix = "./"
if labeling == 'mpnet':
    prefix += "mpnet_acs"
elif labeling == 'simcse':
    prefix += "simcse_acs"
elif labeling == 'angle':
    prefix += "angle_acs"
elif labeling == 'llm':
    prefix += "llm_labeling"
prefix += "/" + d_name + "/"

train_similarity = np.load(prefix + "concept_labels_train.npy")
val_similarity = np.load(prefix + "concept_labels_val.npy")

# === Automatic Concept Correction ===
start = time.time()
print("running automatic concept correction...")

for i in range(train_similarity.shape[0]):
    for j in range(len(concept_set)):
        if get_labels(j, dataset) != train_dataset["label"][i]:
            train_similarity[i][j] = 0.0
        else:
            if train_similarity[i][j] < 0.0:
                train_similarity[i][j] = 0.0

for i in range(val_similarity.shape[0]):
    for j in range(len(concept_set)):
        if get_labels(j, dataset) != val_dataset["label"][i]:
            val_similarity[i][j] = 0.0
        else:
            if val_similarity[i][j] < 0.0:
                val_similarity[i][j] = 0.0

end = time.time()
print("time of ACC:", (end - start) / 3600, "hours")

# === Lưu kết quả ACC ===
os.makedirs(prefix + "acc_outputs", exist_ok=True)
np.save(prefix + "acc_outputs/concept_labels_train_ACC.npy", train_similarity)
np.save(prefix + "acc_outputs/concept_labels_val_ACC.npy", val_similarity)

print("✅ ACC done. Files saved to:", prefix + "acc_outputs/")
