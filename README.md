# Linear Classifiers — Perceptron, Logistic Regression, Linear SVM, SGD

**Problem.** Build fast, interpretable linear models for classification and compare training dynamics, regularization, and decision boundaries.

**Data.** Tabular features with a binary or multiclass label (`dataset/Movies Dataset.zip`).

**Approach.**
- Preprocessing: imputation (if needed) + **StandardScaler**.
- Models: **Perceptron**, **LogisticRegression**, **LinearSVC**, **SGDClassifier**.
- Regularization: L2 (C / alpha sweep), early stopping where applicable.
- Evaluation artifacts: decision boundary plots, confusion matrix, learning curves.
- Options: `class_weight='balanced'` for imbalance; feature importance via coefficients.

**What I Learned.**
- How scaling affects margin-based linear models.
- Trade-offs between Logistic (probabilities) vs Linear SVM (margins).
- Regularization strength (C/alpha) vs generalization.
- When SGD is preferable for larger datasets.

## Quick Start
```bash
# clone
git clone https://github.com/Joe-Naz01/linear_classifiers.git
cd linear_classifiers

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
