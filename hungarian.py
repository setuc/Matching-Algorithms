import numpy as np
import difflib
from scipy.optimize import linear_sum_assignment

# Define the same predicted_data and ground_truth as in cost.py
predicted_data = [
    {"field": "Invoice Num", "text": "INV-0001", "confidence": 0.95},
    {"field": "Date", "text": "2021-01-02", "confidence": 0.80},
    {"field": "Document Type", "text": "Invoice", "confidence": 0.70}
]

ground_truth = [
    {"field": "Invoice Number", "text": "INV-0001"},
    {"field": "Date", "text": "2021-01-01"},
    {"field": "Total", "text": "$100.00"}
]

# Helper to compute similarity between two strings (0 to 1)
def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# Compute cost matrix (identical to cost.py implementation)
pred_fields = predicted_data
gt_fields = ground_truth

cost_matrix = []  # will be a list of lists
for pred in pred_fields:
    pred_label = pred["field"]
    pred_text = pred["text"]
    pred_conf = pred["confidence"]
    row_costs = []
    for gt in gt_fields:
        gt_label = gt["field"]
        gt_text = gt["text"]
        # Field name similarity and value similarity
        name_sim = similarity(pred_label.lower(), gt_label.lower())
        text_sim = similarity(pred_text, gt_text)
        sim_score = 0.5 * name_sim + 0.5 * text_sim  # weigh them equally
        cost = (1 - sim_score) + (1 - pred_conf)     # base cost + confidence penalty
        row_costs.append(cost)
    cost_matrix.append(row_costs)

# ANSI color codes
GREEN = "\033[92m"    # For predicted field names
BLUE = "\033[94m"     # For ground truth field names
YELLOW = "\033[93m"   # For headings or cost values if needed
RESET = "\033[0m"     # Reset to default

# Prepare cost matrix with a dummy column to allow leaving some predictions unmatched
cost_matrix_np = np.array(cost_matrix)
n_pred, n_gt = cost_matrix_np.shape
if n_pred == n_gt:
    dummy_col = np.full((n_pred, 1), 0.5)  # cost 0.5 for leaving a pred unmatched
    cost_matrix_np = np.hstack([cost_matrix_np, dummy_col])
    gt_labels = [gt["field"] for gt in ground_truth] + ["<No Match>"]
elif n_pred > n_gt:
    dummy_cols = np.full((n_pred, n_pred - n_gt), 0.5)
    cost_matrix_np = np.hstack([cost_matrix_np, dummy_cols])
    gt_labels = [gt["field"] for gt in ground_truth] + ["<No Match>"] * (n_pred - n_gt)
elif n_gt > n_pred:
    dummy_rows = np.full((n_gt - n_pred, n_gt), 0.5)
    cost_matrix_np = np.vstack([cost_matrix_np, dummy_rows])
    gt_labels = [gt["field"] for gt in ground_truth]
    pred_fields += [{"field": "<No Prediction>", "text": "", "confidence": 1.0}] * (n_gt - n_pred)

# Solve assignment using the Hungarian algorithm
row_idx, col_idx = linear_sum_assignment(cost_matrix_np)
assigned_pairs = list(zip(row_idx, col_idx))

# Neatly print the colored assignments
print(f"{YELLOW}Optimal assignment:{RESET}")
for pred_index, gt_index in assigned_pairs:
    pred_field = pred_fields[pred_index]["field"]
    gt_field = gt_labels[gt_index]
    print(f"Pred {pred_index} ({GREEN}\"{pred_field}\"{RESET}) -> GT ({BLUE}{gt_field}{RESET})")