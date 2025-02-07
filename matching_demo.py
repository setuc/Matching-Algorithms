import os
import time
import numpy as np
import random
import difflib
import string
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import pulp
import logging

########################################################################
## Logging Setup with Color
########################################################################

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',     # blue
        'INFO': '\033[92m',      # green
        'WARNING': '\033[93m',   # yellow
        'ERROR': '\033[91m',     # red
        'CRITICAL': '\033[95m',  # magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        level_color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{level_color}{message}{self.RESET}"

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Avoid adding multiple handlers if already set
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

setup_logging()
log = logging.getLogger(__name__)

########################################################################
## 1. GENERATE SYNTHETIC DATASET
########################################################################

def generate_synthetic_data(num_fields=10, seed=42):
    """
    Generate a synthetic dataset of predicted fields vs ground truth fields.
    Each field has (field_name, text_value) in ground truth.
    Predictions have the same or slightly modified text plus a confidence score.
    Some predicted fields won't exist in ground truth, and some GT fields may be missed.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate some "ground truth" fields
    ground_truth = []
    for i in range(num_fields):
        gt_field = f"Field_{i}"
        gt_value = f"Value_{i}"
        ground_truth.append({"field": gt_field, "text": gt_value})
        
    # We'll produce a certain number of predictions (some correct, some missing, some extra)
    num_predictions = num_fields + random.randint(-2, 3)  # slightly vary
    predicted_data = []
    
    for i in range(num_predictions):
        if i < num_fields and random.random() < 0.8:
            gt_field = ground_truth[i]["field"]
            gt_value = ground_truth[i]["text"]
            if random.random() < 0.2:
                gt_value += random.choice(string.ascii_lowercase)
            conf = round(random.uniform(0.7, 1.0), 2)
            predicted_data.append({"field": gt_field, "text": gt_value, "confidence": conf})
        else:
            rand_field_num = random.randint(0, num_fields * 2)
            rand_value_num = random.randint(0, num_fields * 2)
            field_str = f"Field_{rand_field_num}"
            text_str = f"Value_{rand_value_num}"
            conf = round(random.uniform(0.3, 0.95), 2)
            predicted_data.append({"field": field_str, "text": text_str, "confidence": conf})
    
    random.shuffle(predicted_data)
    log.debug("Synthetic data generated")
    return ground_truth, predicted_data


########################################################################
## 2. COST FUNCTION + CONFIDENCE FILTER
########################################################################

def string_similarity(a, b):
    """Compute normalized string similarity in [0,1]."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def compute_cost_matrix(ground_truth, predicted_data):
    """
    Builds a cost matrix: cost[i, j] = (1 - similarity) + (1 - confidence).
    Similarity is text-based, combining field name and text value similarity.
    """
    cost_matrix = []
    for pred in predicted_data:
        pred_label = pred["field"]
        pred_text  = pred["text"]
        pred_conf  = pred["confidence"]
        
        row_costs = []
        for gt in ground_truth:
            gt_label = gt["field"]
            gt_text  = gt["text"]
            name_sim = string_similarity(pred_label.lower(), gt_label.lower())
            text_sim = string_similarity(pred_text, gt_text)
            sim_score = 0.5 * name_sim + 0.5 * text_sim
            cost = (1 - sim_score) + (1 - pred_conf)
            row_costs.append(cost)
        cost_matrix.append(row_costs)
    log.debug("Cost matrix computed")
    return np.array(cost_matrix)

def filter_predictions_by_confidence(predictions, threshold=0.6):
    """Filters out predictions below a given confidence threshold."""
    filtered = [p for p in predictions if p["confidence"] >= threshold]
    log.debug(f"Filtered predictions: {len(filtered)} out of {len(predictions)}")
    return filtered


########################################################################
## 3. ALGORITHM IMPLEMENTATIONS
########################################################################

### 3A. Hungarian Algorithm ###
def match_hungarian(ground_truth, predicted_data, cost_matrix):
    n_pred, n_gt = cost_matrix.shape
    dummy_col = np.full((n_pred, 1), 0.6)
    cost_with_dummy = np.hstack([cost_matrix, dummy_col])
    row_idx, col_idx = linear_sum_assignment(cost_with_dummy)
    matches = []
    for r, c in zip(row_idx, col_idx):
        if c < n_gt:
            matches.append((r, c))
    log.debug("Hungarian matching completed")
    return matches

### 3B. Greedy Matching ###
def match_greedy(ground_truth, predicted_data, cost_matrix):
    n_pred, n_gt = cost_matrix.shape
    all_pairs = []
    for i in range(n_pred):
        for j in range(n_gt):
            all_pairs.append((cost_matrix[i, j], i, j))
    
    all_pairs.sort(key=lambda x: x[0])
    matched_pred = set()
    matched_gt = set()
    matches = []
    skip_threshold = 1.0
    
    for cost, i, j in all_pairs:
        if i not in matched_pred and j not in matched_gt:
            if cost < skip_threshold:
                matched_pred.add(i)
                matched_gt.add(j)
                matches.append((i, j))
    log.debug("Greedy matching completed")
    return matches

### 3C. Stable Marriage (Gale-Shapley) ###
def match_stable_marriage(ground_truth, predicted_data, cost_matrix):
    n_pred, n_gt = cost_matrix.shape
    pred_preferences = {}
    for i in range(n_pred):
        sorted_gt = sorted(range(n_gt), key=lambda j: cost_matrix[i, j])
        pred_preferences[i] = sorted_gt
    
    gt_preferences = {}
    for j in range(n_gt):
        sorted_pred = sorted(range(n_pred), key=lambda i: cost_matrix[i, j])
        gt_preferences[j] = sorted_pred
    
    gt_rank = {}
    for j in range(n_gt):
        gt_rank[j] = {}
        for rank, i in enumerate(gt_preferences[j]):
            gt_rank[j][i] = rank
    
    free_preds = list(range(n_pred))
    next_proposal = {i: 0 for i in range(n_pred)}
    current_match = {}
    
    while free_preds:
        i = free_preds[0]
        if next_proposal[i] >= n_gt:
            free_preds.pop(0)
            continue
        j = pred_preferences[i][next_proposal[i]]
        next_proposal[i] += 1
        
        if j not in current_match:
            current_match[j] = i
            free_preds.pop(0)
        else:
            k = current_match[j]
            if gt_rank[j][i] < gt_rank[j][k]:
                current_match[j] = i
                free_preds.pop(0)
                free_preds.append(k)
    
    matches = []
    for j, i in current_match.items():
        matches.append((i, j))
    log.debug("Stable marriage matching completed")
    return matches

### 3D. Linear Programming ###
def match_linear_programming(ground_truth, predicted_data, cost_matrix):
    n_pred, n_gt = cost_matrix.shape
    dummy_col = np.full((n_pred, 1), 0.6)
    cost_extended = np.hstack([cost_matrix, dummy_col])
    n_gt_extended = n_gt + 1
    
    problem = pulp.LpProblem("Matching", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(n_pred), range(n_gt_extended)), 0, 1, pulp.LpBinary)
    
    problem += pulp.lpSum(cost_extended[i][j] * x[i][j] for i in range(n_pred) for j in range(n_gt_extended))
    
    for i in range(n_pred):
        problem += pulp.lpSum([x[i][j] for j in range(n_gt_extended)]) <= 1
    for j in range(n_gt_extended):
        problem += pulp.lpSum([x[i][j] for i in range(n_pred)]) <= 1
    
    solver = pulp.PULP_CBC_CMD(msg=0)
    problem.solve(solver)
    
    matches = []
    for i in range(n_pred):
        for j in range(n_gt_extended):
            if pulp.value(x[i][j]) == 1:
                if j < n_gt:
                    matches.append((i, j))
    log.debug("Linear programming matching completed")
    return matches

### 3E. Approx Nearest Neighbor ###
def match_approx_nn(ground_truth, predicted_data, cost_matrix):
    n_pred, n_gt = cost_matrix.shape
    rng = np.random.RandomState(42)
    gt_positions = rng.rand(n_gt, 2) * 100
    pred_positions = rng.rand(n_pred, 2) * 100
    
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gt_positions)
    dists, indices = nn.kneighbors(pred_positions)
    
    match_threshold = 50.0
    matched_pred = set()
    matched_gt = set()
    matches = []
    
    sorted_preds = sorted(range(n_pred), key=lambda i: dists[i][0])
    
    for i in sorted_preds:
        dist = dists[i][0]
        j = indices[i][0]
        if dist < match_threshold and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            matches.append((i, j))
    log.debug("Approx NN matching completed")
    return matches


########################################################################
## 4. EVALUATION UTILITIES
########################################################################

def compute_tp_fp_fn(matches, ground_truth, predicted_data):
    matched_gt_ids = set([m[1] for m in matches])
    tp = 0
    mismatch = 0
    for (i, j) in matches:
        p_field = predicted_data[i]["field"]
        p_text  = predicted_data[i]["text"]
        g_field = ground_truth[j]["field"]
        g_text  = ground_truth[j]["text"]
        if p_field == g_field and string_similarity(p_text, g_text) > 0.8:
            tp += 1
        else:
            mismatch += 1
    fp = mismatch
    fn = len(ground_truth) - tp
    return tp, fp, fn

def measure_execution_time(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, (end - start)


########################################################################
## 5. INTEGRATED DEMO
########################################################################

def save_summary_table_as_image(metrics, output_folder):
    """
    Create and save a summary table image from the given metrics with improved fonts & column sizes.
    """
    headers = ["Algorithm", "TP", "FP", "FN", "Precision", "Recall", "F1", "Time(ms)"]
    table_data = []
    for algo, m in metrics.items():
        table_data.append([
            algo,
            m["TP"],
            m["FP"],
            m["FN"],
            f"{m['Precision']*100:.0f}",
            f"{m['Recall']*100:.0f}",
            f"{m['F1']*100:.0f}",
            f"{m['ExecTime']*1000:.0f}"
        ])
    
    fig, ax = plt.subplots(figsize=(10, len(metrics)*0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # Increase column widths and row heights
    plt.title("Summary of Matching Algorithms", pad=20, fontsize=14)
    
    table_path = os.path.join(output_folder, "summary_table.png")
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()
    log.debug(f"Summary table image saved to {table_path}")

def run_all_algorithms_demo(num_fields=10, confidence_threshold=0.6):
    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)
    log.info("Generating synthetic data...")
    ground_truth, predictions = generate_synthetic_data(num_fields=num_fields)
    
    log.info("Filtering predictions by confidence...")
    filtered_preds = filter_predictions_by_confidence(predictions, threshold=confidence_threshold)
    
    log.info("Computing cost matrix...")
    cost_matrix = compute_cost_matrix(ground_truth, filtered_preds)
    
    log.info("Running matching algorithms...")
    results = {}
    
    hungarian_matches, hungarian_time = measure_execution_time(match_hungarian, ground_truth, filtered_preds, cost_matrix)
    results["Hungarian"] = (hungarian_matches, hungarian_time)
    
    greedy_matches, greedy_time = measure_execution_time(match_greedy, ground_truth, filtered_preds, cost_matrix)
    results["Greedy"] = (greedy_matches, greedy_time)
    
    stable_matches, stable_time = measure_execution_time(match_stable_marriage, ground_truth, filtered_preds, cost_matrix)
    results["StableMarriage"] = (stable_matches, stable_time)
    
    lp_matches, lp_time = measure_execution_time(match_linear_programming, ground_truth, filtered_preds, cost_matrix)
    results["LinearProgramming"] = (lp_matches, lp_time)
    
    ann_matches, ann_time = measure_execution_time(match_approx_nn, ground_truth, filtered_preds, cost_matrix)
    results["ApproxNN"] = (ann_matches, ann_time)
    
    metrics = {}
    for algo, (matches, t) in results.items():
        tp, fp, fn = compute_tp_fp_fn(matches, ground_truth, filtered_preds)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[algo] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ExecTime": t,
            "Matches": matches
        }
    
    log.info("Generating and saving plots...")
    # 6. Cost Matrix Heatmap
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6,5))
    sns.heatmap(cost_matrix, annot=False, cmap="flare")
    plt.title("Cost Matrix Heatmap (Filtered Predictions)")
    plt.xlabel("Ground Truth Index")
    plt.ylabel("Prediction Index")
    cost_matrix_path = os.path.join(output_folder, "cost_matrix_heatmap.png")
    plt.savefig(cost_matrix_path)
    plt.close()
    log.debug(f"Cost matrix heatmap saved to {cost_matrix_path}")
    
    # 7. Bar Chart of TPs/FPs/FNs for each Algorithm
    algos = list(metrics.keys())
    TPs = [metrics[a]["TP"] for a in algos]
    FPs = [metrics[a]["FP"] for a in algos]
    FNs = [metrics[a]["FN"] for a in algos]
    
    x = np.arange(len(algos))
    width = 0.25
    
    plt.figure(figsize=(10,5))
    plt.bar(x - width, TPs, width, label='TP', color='#2ca02c')
    plt.bar(x, FPs, width, label='FP', color='#d62728')
    plt.bar(x + width, FNs, width, label='FN', color='#9467bd')
    plt.xticks(x, algos, rotation=15)
    plt.ylabel("Count")
    plt.title("TP / FP / FN by Algorithm")
    plt.legend()
    plt.tight_layout()
    bar_chart_path = os.path.join(output_folder, "tp_fp_fn_bar_chart.png")
    plt.savefig(bar_chart_path)
    plt.close()
    log.debug(f"Bar chart saved to {bar_chart_path}")
    
    # 8. Confidence Distribution Plot
    conf_scores = [p["confidence"] for p in filtered_preds]
    plt.figure(figsize=(6,4))
    sns.histplot(conf_scores, bins=10, kde=True, color='#4c72b0')
    plt.title("Confidence Distribution (Filtered)")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    conf_dist_path = os.path.join(output_folder, "confidence_distribution.png")
    plt.savefig(conf_dist_path)
    plt.close()
    log.debug(f"Confidence distribution plot saved to {conf_dist_path}")
    
    # 9. Print a summary table
    log.info("Summary of Results (Threshold = {:.2f})".format(confidence_threshold))
    header = "{:<20} {:>3} {:>3} {:>3} {:>9} {:>9} {:>9} {:>9}".format(
        "Algorithm", "TP", "FP", "FN", "Precision", "Recall", "F1", "Time(ms)"
    )
    print(header)
    for algo, m in metrics.items():
        line = "{:<20} {:>3} {:>3} {:>3} {:>9} {:>9} {:>9} {:>9}".format(
            algo,
            m["TP"],
            m["FP"],
            m["FN"],
            f"{m['Precision']*100:.0f}",
            f"{m['Recall']*100:.0f}",
            f"{m['F1']*100:.0f}",
            f"{m['ExecTime']*1000:.0f}"
        )
        print(line)
    
    # Save the summary table as an image
    save_summary_table_as_image(metrics, output_folder)
    
    log.info("All plots have been saved. Check the '{}' folder.".format(output_folder))
    return ground_truth, filtered_preds, metrics


########################################################################
## 6. RUN
########################################################################

if __name__ == "__main__":
    CONF_THRESHOLD = 0.6
    log.info("Starting Matching Algorithms Demo")
    ground_truth, predictions_metrics, results = run_all_algorithms_demo(num_fields=10, confidence_threshold=CONF_THRESHOLD)
    log.info("Demo finished successfully")