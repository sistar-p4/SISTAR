import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        # Globally record features and the thresholds they have used (format:{feature_index: set(thresholds)}）
        self.feature_thresholds = {}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stop condition: Purity reached/maximum depth reached/insufficient samples
        if (depth == self.max_depth or 
            len(y) < self.min_samples_split or 
            self._gini(y) == 0):
            return self._create_leaf_node(y)
        
        # Find the best splitting feature and threshold
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:  # Unable to find valid split
            return self._create_leaf_node(y)
        
        # Update used threshold records (only if the threshold is new)
        if best_feature not in self.feature_thresholds:
            self.feature_thresholds[best_feature] = set()
        if best_threshold not in self.feature_thresholds[best_feature]:
            self.feature_thresholds[best_feature].add(best_threshold)
        
        # Build the subtree recursively
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth+1)
        
        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            # # If this feature has used 3 thresholds, skip it
            if len(self.feature_thresholds.get(feature, set())) >= 3:
                continue
            
            # Candidate thresholds are generated and used values are filtered
            thresholds = self._generate_candidate_thresholds(X[:, feature])
            used_thresholds = self.feature_thresholds.get(feature, set())
            valid_thresholds = [t for t in thresholds if t not in used_thresholds]
            
            for threshold in valid_thresholds:
                gain = self._calculate_gini_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _generate_candidate_thresholds(self, feature_values):
        # Generate candidate thresholds for numeric features (midpoints of adjacent values)
        sorted_values = np.unique(np.sort(feature_values))
        thresholds = []
        for i in range(1, len(sorted_values)):
            thresholds.append((sorted_values[i-1] + sorted_values[i]) / 2)
        return thresholds

    def _gini(self, y):
        # Calculate Gini impurity
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini -= p ** 2
        return gini

    def _calculate_gini_gain(self, y, feature, threshold):
        # Calculate the Gini gain after splitting
        left_mask = feature <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0  # invalid split
        
        left_gini = self._gini(y[left_mask])
        right_gini = self._gini(y[right_mask])
        total = len(y)
        weighted_gini = (len(y[left_mask])/total)*left_gini + (len(y[right_mask])/total)*right_gini
        return self._gini(y) - weighted_gini