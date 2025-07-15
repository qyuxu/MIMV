import csv
import numpy as np

def save_attention_weights(attention_weights_list, feature_names, filename):
    flattened = []
    for i, weights in enumerate(attention_weights_list):
        if isinstance(weights, list):
            weights = [w.detach().cpu().numpy() for w in weights]
            for j, head_weights in enumerate(weights):
                if isinstance(head_weights, np.ndarray):
                    flat = head_weights.flatten().tolist()
                    flattened.append([i, j] + flat)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_Index', 'Head_Index'] + feature_names)
        writer.writerows(flattened)

