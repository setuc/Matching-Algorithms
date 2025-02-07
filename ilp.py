from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus
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

# Create the ILP model
prob = LpProblem('Assignment', LpMaximize)

# Create binary variables for each prediction-ground_truth pair (i,j)
x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat='Binary')
     for i in range(len(predictions)) for j in range(len(ground_truth))}

# Objective: maximize total confidence of matches
prob += lpSum(predictions[i]["confidence"] * x[(i, j)] 
              for i in range(len(predictions)) 
              for j in range(len(ground_truth)))

# Constraints: each prediction and each ground truth can be matched at most once
for i in range(len(predictions)):
    prob += lpSum(x[(i, j)] for j in range(len(ground_truth))) <= 1
for j in range(len(ground_truth)):
    prob += lpSum(x[(i, j)] for i in range(len(predictions))) <= 1

# Solve the ILP
prob.solve()

# Gather matched pairs based on binary variable values
matches = [(i, j) for (i, j) in x if x[(i, j)].value() == 1]

# Colorful output of results similar to the Hungarian code
print(f"{YELLOW}ILP Optimal Assignment (Status: {LpStatus[prob.status]}):{RESET}")
for i, j in matches:
    pred_field = predictions[i]["field"]
    gt_field = ground_truth[j]["field"]
    print(f"Pred {i} ({GREEN}\"{pred_field}\"{RESET}) -> GT {j} ({BLUE}\"{gt_field}\"{RESET})")
