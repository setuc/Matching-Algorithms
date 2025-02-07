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

# Compute cost matrix
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

# Display the cost matrix
gt_field_names = [gt["field"] for gt in gt_fields]

# ANSI color codes
GREEN = "\033[92m"    # For predicted field names
BLUE = "\033[94m"     # For ground truth field names
YELLOW = "\033[93m"   # For cost values
RESET = "\033[0m"     # Reset to default

for i, (pred, row) in enumerate(zip(pred_fields, cost_matrix)):
    pred_label = pred["field"]
    # Format costs to two decimals
    formatted_costs = [f"{cost:.2f}" for cost in row]
    print(
        f"Pred {i} ({GREEN}\"{pred_label}\"{RESET}) vs GT {BLUE}{gt_field_names}{RESET} costs: {YELLOW}{formatted_costs}{RESET}"
    )