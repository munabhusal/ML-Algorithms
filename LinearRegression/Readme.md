# Linear Regression from Scratch: Sklearn vs. OLS vs. Batch Gradient Descent

An end-to-end mathematical validation and optimization study comparing a production baseline framework against custom scratch-built implementations using the Diabetes dataset.

### Quick Performance Benchmark

| Model Implementation | R2 Score | RMSE | Key Architecture Detail |
| :--- | :--- | :--- | :--- |
| *Scikit-Learn Baseline* | 0.5028 | 54.43 | Uses Singular Value Decomposition (SVD) |
| *Custom OLS Matrix* | 0.5028 | 54.43 | Analytical global minimum ($\beta = (X^T X)^{-1} X^T y$) |
| Custom Batch GD (Scaled) | 0.5060 | 54.26 | Optimized via feature standardization; 500 epochs |

### Key Insights & Engineering Breakthroughs

* **The Scaling Trap:** Running Batch Gradient Descent on raw, unscaled features required over 16,000 epochs due to an elongated, distorted error contour valley.
* **The Optimization:** I used scratch-built Gradient Descent model by applying `StandardScaler` to normalize the feature spaces. With that, I got thess $R^2$ score ($0.5060$) in just **500 iterations**. Which helped in high computational overhead reduction.

### Mathematical Concepts and Derivation
For complete, handwritten step-by-step mathematical breakdowns of the calculus and linear algebra matrices used in this codebase, check out my articles on Medium:
* [OLS Linear Regression Mathematical Derivation](https://munabhusal.medium.com/ols-handwritten-notes-d7dff6402066)
* [Matrix Calculus & Derivatives for Gradient Descent](https://munabhusal.medium.com/derivation-of-gradient-decent-for-linear-b879a847ba90)

### Project Structure
* `linear_regression.ipynb` - Core Jupyter Notebook which containing data pipelines, scratch code for OLS and gradient descent, training loops, and interactive Plotly charts.

* `Handwritten Notes` - Consist of handwritten notes for the better understanding of formula derivation and its concepts for both OLS and Gradient Descent