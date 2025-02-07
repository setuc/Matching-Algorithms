# Mastering Document Field Matching

**A Comprehensive Repository for Evaluating Document Field Extraction**

This repository accompanies the LinkedIn blog post:  
**Mastering Document Field Matching: A Complete Guide**

Modern AI models (e.g., GPT-4 Vision, Qwen-2-VL) can extract structured data directly from documents. When models extract fields from documents using services such as Azure AI Document Intelligence or other vision-based models, it is crucial to compare the predicted outputs against human-labeled ground truth (or another reference) to evaluate accuracy and guide review strategies.

In this repository, you will find multiple implementations of field matching algorithms along with utilities for:
- **Cost computation:** Build a cost matrix based on string similarity and confidence scores.
- **Synthetic data generation:** Create artificial datasets for testing.
- **Algorithm evaluation:** Implement and compare various matching approaches.
- **Visualization:** Plot heatmaps, bar charts, and summary tables of evaluation metrics.

---

## Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Comparison Table of Matching Algorithms](#comparison-table-of-matching-algorithms)
- [Repository Structure](#repository-structure)
- [How to Run the Demo](#how-to-run-the-demo)
- [Development Container Setup](#development-container-setup)
- [About & Repository Description](#about--repository-description)
- [License](#license)

---

## Overview

This repository demonstrates and compares several algorithms for matching predicted document fields with ground truth fields. It includes implementations of:
- **Hungarian (Kuhn-Munkres) Algorithm**
- **Greedy Nearest-Neighbor Matching**
- **Gale-Shapley (Stable Marriage) Matching**
- **Linear Programming (ILP) Approaches**
- **Approximate Nearest Neighbor (Vector Matching)**

Each algorithm is provided as a standalone Python script with color-coded output for easy visualization of the matching process. The integrated demo script (`matching_demo.py`) ties everything together by generating synthetic data, computing the cost matrix, running the matching algorithms, and visualizing the results.

---

## Algorithms Implemented

1. **Hungarian (Kuhn-Munkres) Algorithm**  
   Computes an optimal assignment by minimizing the overall matching cost.

2. **Greedy Matching**  
   Quickly evaluates all possible pairs and selects the lowest-cost matches; very fast but may be suboptimal in edge cases.

3. **Stable Marriage (Gale-Shapley) Algorithm**  
   Produces a stable matching based on mutually ranked preferences (not necessarily cost-optimal).

4. **Linear Programming (ILP) Approaches**  
   Uses ILP formulations to either minimize cost or maximize overall confidence; highly flexible for incorporating additional constraints.

5. **Approximate Nearest Neighbor (Vector Matching)**  
   Leverages vector representations for rapid matching, ideal for large-scale or high-dimensional data.

---

## Comparison Table of Matching Algorithms

| **Algorithm**                  | **Complexity**                                      | **Matching Optimality**                                       | **Pros**                                                                                                                                         | **Cons / Trade-offs**                                                                                                    | **Best Use Cases**                                                                                                                   |
|--------------------------------|-----------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **Hungarian (Kuhn-Munkres)**   | O(n³)                                             | Optimal (maximizes total correct matches)                      | Guarantees the best overall assignment; well-studied and reliable                                                                                 | Can be slow for extremely large n; less flexible for custom constraints                                                   | Standard evaluation (e.g., object detection, tracking) when accuracy is critical and problem size is moderate                        |
| **Greedy Matching**            | ~O(n²) (can be optimized)                           | Near-optimal (heuristic)                                         | Very fast in practice; simple to implement                                                                                                       | May produce suboptimal matches in edge cases; no guarantee of best solution                                                 | Large-scale or time-critical evaluations where a slight drop in accuracy is acceptable                                               |
| **Stable Marriage (Gale-Shapley)** | O(n²)                                          | Stable (preference-optimal) but not necessarily cost-optimal     | Ensures no pair would prefer a different match (fair/stable outcome)                                                                               | Not designed to maximize overall correctness; requires constructing artificial preference lists                           | Niche cases where two-sided preferences are critical; less common in standard prediction vs. ground truth matching                    |
| **Linear Programming (ILP)**   | Varies (polynomial for assignment; worst-case exponential) | Optimal (for defined objective)                                  | Highly flexible – can incorporate complex constraints and custom objectives; yields an exact solution                                             | Slower than specialized algorithms; requires an LP solver; may be overkill for simple cases                                | When evaluation requires additional constraints or custom matching criteria, even at smaller scales                                   |
| **Approx. Nearest Neighbor**   | ~O(n log n) build + ~O(n log n) query               | Approximate (high accuracy but not guaranteed)                   | Extremely fast for large or high-dimensional data; memory-efficient; tunable for speed vs. accuracy                                              | Small chance of suboptimal matches; one-to-one matching is not inherently guaranteed (requires additional steps)             | Real-time matching on massive datasets or matching millions of features (e.g., embedding vectors) where exact methods are too slow     |

## How to Run the Demo

1. **Install Dependencies:**  
   Make sure the following Python libraries are installed:
   - numpy
   - matplotlib
   - seaborn
   - pulp
   - scipy
   - scikit-learn

   You can install these by running:

   ```sh
   pip install -r requirements.txt
   ```
2. **Run the Demo Script:**
   Execute the demo to generate synthetic data, run the matching algorithms, and create visualizations:
   ```sh
   python matching_demo.py
   ```
   The demo will create a plots folder with images such as cost_matrix_heatmap.png, tp_fp_fn_bar_chart.png, summary_table.png, and confidence_distribution.png.

