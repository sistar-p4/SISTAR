# Model
This folder contains implementations of machine learning models and algorithms. These models are designed for various tasks such as classification and decision-making.

## Files

### `DT-CTS.py`
This file contains the implementation of a `DecisionTreeClassifier`. The classifier supports the following features:
- Customizable maximum depth (`max_depth`) and minimum samples required to split (`min_samples_split`).
- Tracks thresholds used for splitting to avoid redundant splits.
- Implements Gini impurity for evaluating splits.

### Usage

To use the `DecisionTreeClassifier`, import the class and fit it to your dataset:

```python
from DT-CTS import DecisionTreeClassifier
import numpy as np

# Example dataset
X = np.array([[2.3, 4.5], [1.1, 3.3], [3.3, 2.2], [4.4, 1.1]])
y = np.array([0, 1, 0, 1])

# Initialize and fit the model
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
clf.fit(X, y)

# Access the trained tree
print(clf.tree)
```

### Notes

- The classifier avoids using more than three thresholds per feature to prevent overfitting.
- The implementation is designed for educational and experimental purposes and may not be optimized for large datasets.
