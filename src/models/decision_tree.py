import numpy as np
import pandas as pd

'''
-----> TreeNode class <-----
'''
class TreeNode:

    # constructor used for initialising a TreeNode
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.data_points = 0
        self.leaf = (left is None and right is None)

'''
-----> Decision Tree Classifier <-----

Hyperparameters:
- criterion: The function to measure the quality of a split [gini, entropy]
- max_depth: The maximum depth of questions to ask
- min_samples_split: The minimum number of samples required to split an internal node
- min_samples_leaf: The minimum number of samples required for a given leaf node to be created
- max_leaf_nodes: The maximum number of leaf nodes the tree can have
'''

class DecisionTreeClassifier:

    # Initialization method to set hyperparameters
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.root = None
        return
    
    # Method to print the contents of the tree including training data-points at each node
    def __str__(self):
        if not self.root:
            return "DecisionTreeClassifier has not been fit to training data"
        lines = []
        self._print_tree(self.root, lines)
        return "\n".join(lines)

    def _print_tree(self, node, lines, spacing=""):
        if node.leaf:
            lines.append(spacing + f"Predict: {node.value}")
            return
        lines.append(spacing + f"[Feature {node.feature} >= {node.threshold}]?")
        self._print_tree(node.left, lines, spacing + "--")
        self._print_tree(node.right, lines, spacing + "--")

    def fit(self, rows):
        self.root = self.build_tree(rows)

    def predict_one(self, row):
        node = self.root
        while not node.leaf:
            if isinstance(row[node.feature], str):
                if row[node.feature] == node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                if row[node.feature] >= node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node.value

    def predict(self, rows):
        return [self.predict_one(row) for row in rows]

    # Method to fit the decision tree classifier to the training data
    def build_tree(self, rows, depth=0):

        # base case if no data is passed
        if len(rows) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return TreeNode(value=self._majority_class(rows))

        # calculate the information gain, split feature, and threshold value
        info_gain, feature, threshold = self._find_best_split(rows)

        if info_gain == 0:
            return TreeNode(value=self._majority_class(rows))

        candidate_thresholds, categorical = self._get_candidates(rows, feature)
        true_rows, false_rows = self._partition(rows, feature, threshold, categorical)

        left_branch = self.build_tree(true_rows, depth + 1)
        right_branch = self.build_tree(false_rows, depth + 1)

        return TreeNode(feature=feature, threshold=threshold, left=left_branch, right=right_branch)

    def _class_counts(self, rows):
        counts = {}
        for row in rows:
            label = row[-1]  # assuming last column is the label
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def _majority_class(self, rows):
        counts = self._class_counts(rows)
        return max(counts, key=counts.get)

    # Method for calculating Gini impurity
    def _gini_impurity(self, rows):
        counts = self._class_counts(rows)
        impurity = 1
        total = len(rows)

        for lbl in counts:
            prob_of_lbl = counts[lbl] / total
            impurity -= prob_of_lbl ** 2
        return impurity

    # Method for calculating entropy
    def _entropy(self, rows):
        counts = self._class_counts(rows)
        total = len(rows)
        entropy = 0

        for lbl in counts:
            prob_of_lbl = counts[lbl] / total
            entropy -= prob_of_lbl * np.log2(prob_of_lbl)
        return entropy

    # Method for finding the most information gain in a candidate split on feature at threshold
    def _find_best_split(self, rows):

        best_gain = 0
        best_feature = None
        best_threshold = None
        feature_count = len(rows[0]) - 1

        # calculate current uncertainty
        current_uncertainty = self._gini_impurity(rows) if self.criterion == 'gini' else self._entropy(rows)

        for feature in range(feature_count):

            candidate_thresholds, categorical = self._get_candidates(rows, feature)

            for candidate in candidate_thresholds:

                true_rows, false_rows = self._partition(rows, feature, candidate, categorical)

                if len(true_rows)  == 0 or len(false_rows) == 0:
                    continue

                p = float(len(true_rows))/len(rows)
                if self.criterion == 'gini':
                    gain = current_uncertainty - (p*self._gini_impurity(true_rows) + (1-p)*self._gini_impurity(false_rows))
                else:
                    gain = current_uncertainty - (p*self._entropy(true_rows) + (1-p)*self._entropy(false_rows))

                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, candidate

        return best_gain, best_feature, best_threshold

    def _partition(self, rows, feature, candidate, categorical):
        
        true_rows, false_rows = [], []
        for row in rows:

            # partition based on categorical feature
            if categorical:
                if row[feature] == candidate:
                    true_rows.append(row)
                else:
                    false_rows.append(row)

            else:
                if row[feature] >= candidate:
                    true_rows.append(row)
                else:
                    false_rows.append(row)
        
        return true_rows, false_rows


    def _get_candidates(self, rows, feature):

        if type(rows[0][feature]) == str: # Categorical feature
            return set(row[feature] for row in rows), True

        else: # Continuous feature
            sorted_elements = sorted(set(row[feature] for row in rows))
            candidates = []
            for i in range(len(sorted_elements) - 1):
                midpoint = (sorted_elements[i] + sorted_elements[i+1])/2
                candidates.append(midpoint)

            return candidates, False