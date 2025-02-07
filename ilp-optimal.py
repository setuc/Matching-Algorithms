import difflib
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus

# ANSI color codes for colorful printing
GREEN  = "\033[92m"    # For predicted field names
BLUE   = "\033[94m"    # For ground truth field names
YELLOW = "\033[93m"    # For headings or emphasis
RESET  = "\033[0m"     # Reset to default

# Example data for predictions and ground truth
predictions = [
    {"field": "Invoice Num", "text": "INV-0001", "confidence": 0.95},
    {"field": "Date", "text": "2021-01-02", "confidence": 0.80},
    {"field": "Document Type", "text": "Invoice", "confidence": 0.70}
]

ground_truth = [
    {"field": "Invoice Number", "text": "INV-0001"},
    {"field": "Date", "text": "2021-01-01"},
    {"field": "Total", "text": "$100.00"}
]

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# --- Add dummy ground truth options so that every prediction is forced to match ---
n_pred = len(predictions)
n_gt   = len(ground_truth)
# If numbers are equal, add one dummy to allow leaving unmatched.
dummy_count = 1 if n_pred == n_gt else (n_pred - n_gt)
dummy_option = {"field": "<No Match>", "text": "", "confidence": 0.0}
dummy_ground_truth = [dummy_option for _ in range(dummy_count)]
new_ground_truth = ground_truth + dummy_ground_truth
n_total_gt = len(new_ground_truth)

# Compute cost parameters for ILP.
# For a real ground truth (j < n_gt):
#     cost = (1 - sim_score) + (1 - prediction confidence)
# where sim_score is the average similarity (field names and texts).
# For dummy options (j >= n_gt), we use a fixed dummy cost.
cost = {}
for i in range(n_pred):
    for j in range(n_total_gt):
        if j < n_gt:
            pred = predictions[i]
            gt = ground_truth[j]
            name_sim = similarity(pred["field"].lower(), gt["field"].lower())
            text_sim = similarity(pred["text"], gt["text"])
            sim_score = 0.5 * name_sim + 0.5 * text_sim
            cost[(i, j)] = (1 - sim_score) + (1 - pred["confidence"])
        else:
            cost[(i, j)] = 0.5  # fixed dummy cost

# Create the ILP model as a minimization problem.
prob = LpProblem('Assignment', LpMinimize)

# Create binary variables for each prediction - ground_truth (including dummy) pair (i,j)
x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat='Binary')
     for i in range(n_pred) for j in range(n_total_gt)}

# Objective: minimize total matching cost.
prob += lpSum(cost[(i, j)] * x[(i, j)]
              for i in range(n_pred)
              for j in range(n_total_gt))

# Constraints: Force every prediction to be assigned exactly one option (real or dummy)
for i in range(n_pred):
    prob += lpSum(x[(i, j)] for j in range(n_total_gt)) == 1

# For real ground truth options, ensure at most one prediction is matched.
for j in range(n_gt):
    prob += lpSum(x[(i, j)] for i in range(n_pred)) <= 1

# (No constraint on dummy options so they can be assigned to multiple predictions)

# Solve the ILP
prob.solve()

# Gather matched pairs based on binary variable values
matches = [(i, j) for (i, j) in x if x[(i, j)].value() == 1]

# Colorful output of results similar to the Hungarian code
print(f"{YELLOW}ILP Optimal Assignment (Status: {LpStatus[prob.status]}):{RESET}")
for i, j in matches:
    pred_field = predictions[i]["field"]
    if j < n_gt:
        gt_field = ground_truth[j]["field"]
        print(f"Pred {i} ({GREEN}\"{pred_field}\"{RESET}) -> GT {j} ({BLUE}\"{gt_field}\"{RESET})")
    else:
        print(f"Pred {i} ({GREEN}\"{pred_field}\"{RESET}) -> GT (dummy) ({BLUE}\"<No Match>\"{RESET})")