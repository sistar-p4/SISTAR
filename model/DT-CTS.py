import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_thresholds = {}  # Record used thresholds for each feature

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stopping conditions: max depth reached / not enough samples / pure node
        if (depth == self.max_depth or 
            len(y) < self.min_samples_split or 
            self._gini(y) == 0):
            return self._create_leaf_node(y)
        
        # Find the best feature and threshold to split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:  # Cannot split further
            return self._create_leaf_node(y)
        
        # Update record of used thresholds
        if best_feature not in self.feature_thresholds:
            self.feature_thresholds[best_feature] = set()
        if best_threshold not in self.feature_thresholds[best_feature]:
            self.feature_thresholds[best_feature].add(best_threshold)
        
        # Recursively build subtrees
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth+1)
        
        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y):
        best_gain = -float('inf')
        best_feature, best_threshold = None, None
        
        for feature in range(X.shape[1]):
            # Skip features that have already used 3 thresholds
            if len(self.feature_thresholds.get(feature, set())) >= 3:
                continue
            
            # Generate candidate thresholds and filter out used ones
            thresholds = self._generate_candidate_thresholds(X[:, feature])
            used_thresholds = self.feature_thresholds.get(feature, set())
            valid_thresholds = [t for t in thresholds if t not in used_thresholds]
            
            for threshold in valid_thresholds:
                gain = self._calculate_gini_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature, threshold
        
        return best_feature, best_threshold

    def _generate_candidate_thresholds(self, feature_values):
        # For numerical features: generate midpoints between sorted unique values
        sorted_values = np.unique(np.sort(feature_values))
        if len(sorted_values) < 2:
            return []
        thresholds = [(sorted_values[i-1] + sorted_values[i]) / 2 
                      for i in range(1, len(sorted_values))]
        return thresholds

    def _gini(self, y):
        # Compute Gini impurity
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _calculate_gini_gain(self, y, feature, threshold):
        # Calculate Gini gain after split
        mask = feature <= threshold
        if np.sum(mask) == 0 or np.sum(~mask) == 0:
            return 0  # Invalid split
        left_gini = self._gini(y[mask])
        right_gini = self._gini(y[~mask])
        total = len(y)
        weighted_gini = (np.sum(mask)/total * left_gini + 
                         np.sum(~mask)/total * right_gini)
        return self._gini(y) - weighted_gini

    def _create_leaf_node(self, y):
        # Create a leaf node (return the majority class)
        classes, counts = np.unique(y, return_counts=True)
        return {'class': classes[np.argmax(counts)]}

    def predict(self, X):
        # Predict for each sample
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, node):
        # Recursively traverse the tree to make a prediction
        if 'class' in node:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

# Test code ------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create decision tree model
    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=10)
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Test set accuracy: {accuracy:.4f}")

    # View feature threshold usage
    print("\nFeature threshold usage record:")
    for feature, thresholds in tree.feature_thresholds.items():
        print(f"Feature {feature} used threshold count: {len(thresholds)}")
