import torch
from models.transformer import TransformerBinaryClassifier
from dataset import dataset1
from utils.trainer import train
from utils.tester import test
from utils.attention import save_attention_weights
import logging
import os

# 日志配置
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(filename='outputs/train.log', level=logging.INFO)

def log_fn(msg):
    print(msg)
    logging.info(msg)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_fn(f"Using device: {device}")

    # 加载数据
    trainset, valset, testset, feature_names = dataset1.load_data("data/processed")
    X_train, y_train = torch.tensor(trainset[0].values, dtype=torch.float32), torch.tensor(trainset[1].values, dtype=torch.float32)
    X_val, y_val = torch.tensor(valset[0].values, dtype=torch.float32), torch.tensor(valset[1].values, dtype=torch.float32)
    X_test, y_test = torch.tensor(testset[0].values, dtype=torch.float32), torch.tensor(testset[1].values, dtype=torch.float32)
    AA_mut = testset[2]

    from torch.utils.data import TensorDataset, DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = TransformerBinaryClassifier(input_dim=X_train.shape[1])
    model = model.to(device)

    from torch.nn import BCELoss
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    best_state = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, log_fn=log_fn)
    model.load_state_dict(best_state)

    predictions, labels, attentions, auc = test(model, test_loader, device)
    log_fn(f"Test AUC: {auc:.4f}")

    save_attention_weights(attentions, feature_names, 'outputs/attention_weights.csv')
    log_fn("Attention weights saved.")

if __name__ == "__main__":
    main()

