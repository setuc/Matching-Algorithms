import difflib

# ANSI color codes
GREEN  = "\033[92m"    # For predicted field names
BLUE   = "\033[94m"    # For ground truth field names
YELLOW = "\033[93m"    # For headings and cost values
RESET  = "\033[0m"     # Reset to default

# Example data (same as used in hungarian.py)
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

# Compute the cost matrix (using same cost logic as hungarian.py)
cost_matrix = []
for pred in predicted_data:
    pred_label = pred["field"]
    pred_text = pred["text"]
    pred_conf = pred["confidence"]
    row_costs = []
    for gt in ground_truth:
        gt_label = gt["field"]
        gt_text = gt["text"]
        name_sim = similarity(pred_label.lower(), gt_label.lower())
        text_sim = similarity(pred_text, gt_text)
        sim_score = 0.5 * name_sim + 0.5 * text_sim
        cost = (1 - sim_score) + (1 - pred_conf)
        row_costs.append(cost)
    cost_matrix.append(row_costs)

# Print the cost matrix with colored output (without any dummy columns)
print(f"{YELLOW}Cost Matrix (rows: predictions, columns: ground truths):{RESET}")
for i, row in enumerate(cost_matrix):
    row_string = "  ".join(f"{cost:.2f}" for cost in row)
    print(f"Pred {i} ({GREEN}{predicted_data[i]['field']}{RESET}): {row_string}")

###########################################################################
# Gale–Shapley stable matching algorithm
def gale_shapley(pred_prefs, gt_prefs):
    # pred_prefs: dict of {pred_index: [gt_index preference list]}
    # gt_prefs: dict of {gt_index: [pred_index preference list]}
    free_preds = list(pred_prefs.keys())
    current_match = {}  # gt_index -> pred_index
    # Precompute ranking for quick comparison: rank[gt][pred] = position in gt's list (lower = more preferred)
    rank = {j: {pred: r for r, pred in enumerate(gt_prefs[j])} for j in gt_prefs}
    next_proposal = {i: 0 for i in pred_prefs}  # index in pref list each pred will propose next

    while free_preds:
        i = free_preds[0]  # take a free prediction
        if next_proposal[i] >= len(pred_prefs[i]):
            free_preds.pop(0)  # no one left to propose
            continue
        j = pred_prefs[i][next_proposal[i]]
        next_proposal[i] += 1
        if j not in current_match:
            current_match[j] = i
            free_preds.pop(0)
        else:
            k = current_match[j]
            if rank[j][i] < rank[j][k]:
                current_match[j] = i
                free_preds.pop(0)
                free_preds.append(k)
            # Otherwise, j rejects i (i remains free)
    return current_match

###########################################################################
# Build preference lists from cost_matrix (not including dummy for stable matching logic)
pred_preferences = {}
for i, row in enumerate(cost_matrix):
    sorted_gt = sorted(range(len(row)), key=lambda j: row[j])
    pred_preferences[i] = sorted_gt

gt_preferences = {}
for j in range(len(cost_matrix[0])):
    sorted_pred = sorted(range(len(cost_matrix)), key=lambda i: cost_matrix[i][j])
    gt_preferences[j] = sorted_pred

stable_match = gale_shapley(pred_preferences, gt_preferences)
print(f"\n{YELLOW}Stable matches (gt_index -> pred_index):{RESET}", stable_match)
