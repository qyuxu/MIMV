from sklearn.metrics import roc_auc_score
import torch

def test(model, test_loader, device):
    model.eval()
    predictions, labels, attentions = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, attention_weights = model(inputs)
            prob = torch.sigmoid(outputs).squeeze().cpu().numpy()
            predictions.extend(prob)
            labels.extend(targets.cpu().numpy())
            attentions.append(attention_weights)
    auc = roc_auc_score(labels, predictions)
    return predictions, labels, attentions, auc

