# Decision Tree From Scratch (Information Metrics & Vectorization)

A clean, standalone Python implementation demonstrating how a Decision Tree determines its optimal splitting criteria using **Entropy**, **Gini Impurity**, and **Information Gain**. 

This project bypasses slow, traditional row-by-row looping structures in favor of **highly optimized, vectorized Pandas operations** (`.value_counts(normalize=True)`) to perform calculations instantly.

---

### Core Concepts Implemented

1. **Entropy ($H(S)$):** Measures the impurity or uncertainty in a group of observations. It ranges between 0 (perfectly pure) and 1 (completely random split).
   $$H(S) = -\sum p_i \log_2(p_i)$$

2. **Gini Impurity:** An alternative metric to Entropy that measures how often a randomly chosen element from the set would be incorrectly labeled. It ranges between 0 and 0.5.
   $$\text{Gini}(S) = 1 - \sum p_i^2$$

3. **Information Gain ($IG$):** Measures the reduction in entropy or impurity after a dataset is split on a specific feature. The feature with the **highest** Information Gain is selected as the root/splitting node.
   $$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

---

### Features & Optimizations

* **100% Vectorized Calculations:** Uses NumPy and Pandas vector methods instead of Python nested `for` loops to process node distributions efficiently.
* **Dynamic Feature Parsing:** Automatically evaluates and scores every independent feature in your dataset regardless of column order or position.
* **Zero Machine Learning Dependencies:** Implemented using only core data science libraries (`pandas` and `numpy`) without relying on `scikit-learn`.

---

### Code Architecture

The main script contains three primary core functions:

* `entropy(y_data)`: Returns the total baseline entropy of a target array.
* `gini_impurity(y_data)`: Returns the baseline Gini Impurity of a target array.
* `information_gain(X_column, y)`: Computes the precise mathematical information gain when splitting target $y$ across an independent feature vector `X_column`.

---

### Sample Dataset Output (`play_tennis.csv`)

When evaluated against the standard **Play Tennis** classification dataset, the pipeline dynamically calculates the following results:

```text
Information Gains:
{
 'outlook': 0.2467, 
 'temp': 0.0292, 
 'humidity': 0.1518, 
 'wind': 0.0481
}
____________________
Highest Information Gain: outlook