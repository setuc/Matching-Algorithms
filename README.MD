# Mastering Document Field Matching

This repository accompanies the LinkedIn blog post:  
**Mastering Document Field Matching: A Complete Guide**

Modern AI models like GPT-4 Vision and Qwen-2-VL can analyze images and generate structured JSON outputs directly. These models understand both visual content and text, making them ideal for document extraction tasks. When AI models extract structured data from documents—using services such as Azure AI Document Intelligence (formerly Form Recognizer) or other vision-based foundation models—it becomes necessary to compare one model’s output against either human-labeled ground truth or an alternative data source. This matching process is crucial for evaluating accuracy and determining review strategies.

In this repository, you'll find multiple implementations of field matching algorithms along with utilities for cost computation, synthetic data generation, result evaluation, and visualization.

## Table of Contents

- [Constructing a Cost Matrix for Field Matching](#constructing-a-cost-matrix-for-field-matching)
- [Algorithm 1: Kuhn-Munkres (Hungarian) Algorithm](#algorithm-1-kuhn-munkres-hungarian-algorithm)
- [Algorithm 2: Greedy Nearest-Neighbor Matching](#algorithm-2-greedy-nearest-neighbor-matching)
- [Algorithm 3: Gale-Shapley (Stable Marriage) Matching](#algorithm-3-gale-shapley-stable-marriage-matching)
- [Algorithm 4: Linear Programming (ILP) Approach](#algorithm-4-linear-programming-ilp-approach)
- [Algorithm 5: Approximate Nearest Neighbor (Vector Matching)](#algorithm-5-approximate-nearest-neighbor-vector-matching)
- [Putting It All Together](#putting-it-all-together)
- [Production Guidelines](#production-guidelines)
- [How to Run the Demo](#how-to-run-the-demo)
- [Repository Structure](#repository-structure)


## Constructing a Cost Matrix for Field Matching

The foundation for matching involves computing a cost matrix. Each cell represents the cost of pairing a predicted field with a ground truth field. Costs are determined based on string similarity and confidence penalties. Lower costs indicate better matches.


## Algorithm 1: Kuhn-Munkres (Hungarian) Algorithm

This algorithm computes an optimal assignment that minimizes the overall matching cost. The repository includes this implementation along with a detailed explanation of how it operates.


## Algorithm 2: Greedy Nearest-Neighbor Matching

A simpler and faster approach that evaluates all possible prediction–ground truth pairs, ranks them by cost, and selects the lowest ones. Its simplicity makes it straightforward to implement and test.


## Algorithm 3: Gale-Shapley (Stable Marriage) Matching

By building preference lists from both the predicted and ground truth fields, this algorithm computes a stable matching. It is particularly useful when mutual preferences are important.


## Algorithm 4: Linear Programming (ILP) Approach

Two ILP-based formulations provide alternative solutions:
- A formulation that minimizes the overall matching cost.
- An approach that maximizes the overall confidence.
Both strategies are included in the repository to illustrate different optimization paths.


## Algorithm 5: Approximate Nearest Neighbor (Vector Matching)

This method leverages vector representations to quickly identify potential field matches. It is designed for efficiency, especially when dealing with large datasets.


## Putting It All Together

The demo script integrates all of the above algorithms. It is designed to:
- Generate synthetic data.
- Compute the cost matrix.
- Apply each matching algorithm.
- Evaluate the results using metrics such as true positives, false positives, and false negatives.
- Visualize the outcomes through heatmaps and bar charts.


## Production Guidelines

The accompanying blog post discusses best practices for deploying these algorithms in production. Key considerations include:
- Using confidence scores and thresholds appropriately.
- Evaluating trade-offs between speed, accuracy, and scalability.



## How to Run the Demo

1. **Install Dependencies:**  
   Ensure all required libraries are installed (for example: numpy, matplotlib, seaborn, pulp, scipy, and sklearn).

2. **Run the Demo Script:**  
   Execute the main demo script from the command line:
   ```sh
   python Users/admin/Matching%20Algorithms/matching_demo.py
   ```