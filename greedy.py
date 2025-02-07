import numpy as np
import difflib

# Azure Document Intelligence predictions JSON, simplified for brevity
predicted_data = [
    {"field": "Invoice Num", "text": "INV-0001", "confidence": 0.95},
    {"field": "Date", "text": "2021-01-02", "confidence": 0.80},
    {"field": "Document Type", "text": "Invoice", "confidence": 0.70}
]

# "Ground truth" (could be human-labeled, CRM data, or from another model)
ground_truth = [
    {"field": "Invoice Number", "text": "INV-0001"},
    {"field": "Date", "text": "2021-01-01"},
    {"field": "Total", "text": "$100.00"}
]

# Helper to compute similarity between two strings (0 to 1)
def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# Compute cost matrix (same as in cost.py and hungarian.py)
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
YELLOW = "\033[93m"   # For cost values or headers
RESET = "\033[0m"     # Reset to default

# Remember the original number of ground truth fields
orig_n_gt = len(ground_truth)

# Prepare numpy cost matrix with dummy columns/rows if needed
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

# Greedy matching based on all possible pairs sorted by cost ascending,
# but excluding dummy columns (i.e. only for j < orig_n_gt)
matched_pred = set()
matched_gt = set()
greedy_pairs = []

# Create sorted list of candidate pairs using only real GT indices
# In the hungarian method, we had the dummy columns to allow leaving some predictions unmatched
# Here, we only consider the original number of GT fields
pairs = [(cost_matrix_np[i, j], i, j)
         for i in range(cost_matrix_np.shape[0])
         for j in range(orig_n_gt)]
pairs.sort(key=lambda x: x[0])  # sort by cost ascending

for cost, i, j in pairs:
    if i not in matched_pred and j not in matched_gt:
        # If both this prediction and this ground truth are unmatched, assign them
        greedy_pairs.append((i, j, cost))
        matched_pred.add(i)
        matched_gt.add(j)

print("Greedy matched pairs (pred_index, gt_index, cost):", greedy_pairs)
# Print the greedy matching results with colored output
print(f"{YELLOW}Greedy matched pairs:{RESET}")
for i, j, cost in greedy_pairs:
    pred_field = pred_fields[i]["field"]
    gt_field = gt_labels[j]
    print(f"Pred {i} ({GREEN}\"{pred_field}\"{RESET}) -> GT ({BLUE}\"{gt_field}\"{RESET}) with cost: {YELLOW}{cost:.2f}{RESET}")