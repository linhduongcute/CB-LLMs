import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import config as CFG
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from utils import normalize

parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--saga_epoch", type=int, default=500)
parser.add_argument("--saga_batch_size", type=int, default=256)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # --- Load datasets ---
    dataset_name = "SetFit/sst2"  # hoặc dataset khác nếu muốn
    train_dataset = load_dataset(dataset_name, split="train")
    val_dataset = load_dataset(dataset_name, split="validation")
    test_dataset = load_dataset(dataset_name, split="test")

    # --- Load ACC-corrected concept features ---
    train_c = torch.tensor(np.load("./mpnet_acs/SetFit_sst2/concept_labels_train_acc.npy"), dtype=torch.float32)
    val_c = torch.tensor(np.load("./mpnet_acs/SetFit/sst2/concept_labels_val_acc.npy"), dtype=torch.float32)

    # --- Build test features (từ dataset gốc, vẫn tính bình thường) ---
    # Nếu bạn không muốn test, có thể comment toàn bộ phần này
    # Ở đây giả sử test_c = zeros, chỉ để chạy linear
    test_c = torch.zeros((len(test_dataset), train_c.shape[1]))

    # --- Normalize + ReLU ---
    train_c, train_mean, train_std = normalize(train_c, d=0)
    train_c = F.relu(train_c)
    val_c, _, _ = normalize(val_c, d=0, mean=train_mean, std=train_std)
    val_c = F.relu(val_c)
    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    # --- Build labels ---
    train_y = torch.LongTensor(train_dataset["label"])
    val_y = torch.LongTensor(val_dataset["label"])
    test_y = torch.LongTensor(test_dataset["label"])

    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c, val_y)
    test_ds = TensorDataset(test_c, test_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.saga_batch_size, shuffle=False)

    # --- Train linear layer ---
    print("dim of concept features: ", train_c.shape[1])
    linear = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset_name])
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': 0.0007}}

    print("training final layer...")
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, k=10,
                           val_loader=val_loader, test_loader=test_loader, do_zero=True,
                           n_classes=CFG.class_num[dataset_name])

    print("save weights with test acc:", output_proj['path'][-1]['metrics']['acc_test'])
    W_g = output_proj['path'][-1]['weight']
    b_g = output_proj['path'][-1]['bias']

    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.saga_epoch, ALPHA, epsilon=1, k=1,
                           val_loader=val_loader, test_loader=test_loader, do_zero=False,
                           n_classes=CFG.class_num[dataset_name], metadata=metadata, n_ex=train_c.shape[0])

    print("save the sparse weights with test acc:", output_proj['path'][0]['metrics']['acc_test'])
    W_g_sparse = output_proj['path'][0]['weight']
    b_g_sparse = output_proj['path'][0]['bias']

    prefix = "./mpnet_acs/SetFit_sst2/linear/"
    os.makedirs(prefix, exist_ok=True)
    torch.save(W_g, prefix + 'W_g.pt')
    torch.save(b_g, prefix + 'b_g.pt')
    torch.save(W_g_sparse, prefix + 'W_g_sparse.pt')
    torch.save(b_g_sparse, prefix + 'b_g_sparse.pt')
