# Logistic Regression from Scratch: Scikit-Learn vs. Batch Gradient Descent (`LogiGD`)

An end-to-end mathematical validation and optimization study comparing a production baseline framework against custom scratch-built implementations using a synthetic binary classification dataset.

### Quick Performance Benchmark

| Model Implementation | Testing Accuracy | F1-Score (Class 1) | Key Architecture Detail |
| :--- | :--- | :--- | :--- |
| *Scikit-Learn Baseline* | 93.33% | 0.93 | Production-grade `liblinear`/`lbfgs` solver footprint |
| **Custom `LogiGD`** | 93.33% | 0.93 | Iterative Vectorized Gradient Descent with Sigmoid Activation |

### Key Insights & Engineering Breakthroughs

* **The Numerical Underflow Guard:** Vectorizing the Binary Cross-Entropy loss computation natively in NumPy can cause unstable execution due to $\log(0)$ errors. To solve this, I injected a custom safety buffer (`+ 1e-15`), keeping the logarithmic contours perfectly stable.
* **Algorithmic Convergence:** The custom-built `LogiGD` optimizer successfully minimizes loss in just **249 epochs** at a learning rate of $0.1$. This matches the exact validation matrix, precision coefficients, and decision boundaries of Scikit-Learn's highly optimized solver.

### Mathematical Understanding
For complete, step-by-step mathematical breakdowns of the classification costs, probability mappings, and formulas used to build this codebase, check out my article on Medium:
* [Binary Cost Function](https://munabhusal.medium.com/binary-cost-function-c38bb9bfb23)
* [Logistic Regression Concepts](https://munabhusal.medium.com/logistic-regression-concepts-8471af4a7a5)

### Project Structure
* `Logistic Regression From Scratch vs. Scikit-Learn.ipynb` - Core Jupyter Notebook containing the data pipeline, raw NumPy implementation of the Sigmoid function, manual weight update loops, cost function convergence tracking, and side-by-side decision boundary subplots.