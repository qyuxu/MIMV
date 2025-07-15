import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 示例结构，按你原来的模型补充
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        out = torch.sigmoid(self.output_layer(x))
        attention_weights = [x]  # mock attention
        return out, attention_weights

