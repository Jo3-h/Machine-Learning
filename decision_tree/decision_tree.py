import numpy as np
import pandas as pd

'''
-----> TreeNode class <-----
'''
class TreeNode:

    # constructor used for initialising a leaf node
    def __init__(self, rows):
        self.rows = rows
        self.leaf = True

    # constructor used for initialising a decision node
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.data_points = 0
        self.leaf = False

'''
-----> Decision Tree Classifier <-----

Hyperparameters:
- criterion: The function to measure the quality of a split [gini, entropy]
- max_depth: The maximum depth of questions to ask
- min_samples_split: The minimum number of samples required to split an internal node
- min_samples_leaf: The minimum number of samples required for a given leaf node to be created
- max_leaf_nodes: The maximum number of leaf nodes the tree can have
'''

class DecisionTree:

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
        return

    # Method to fit the decision tree classifier to the training data
    def build_tree(self, data, depth=0):

        # base case if no data is passed
        if len(data) == 0 or depth > self.max_depth:
            return None

        # calculate the information gain, split feature, and threshold value
        info_gain, feature, threshold = _find_best_split(rows)

        if info_gain == 0:
            return TreeNode(rows)

        return

    # Method for finding the most information gain in a candidate split on feature at threshold
    def _find_best_split(rows):



        best_gain = 0
        best_feature = None
        feature_count = len(rows[0]) - 1

        for feature in range(feature_count):

            candidate_thresholds, categorical = self._get_candidates(rows, feature)

            for candidate in candidate_thresholds:

                true_rows, false_rows = self._partition(rows, feature, candidate, categorical)

                if len(true_rows) 

    def _partition(rows, feature, candidate, categorical):
        
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


    def _get_candidates(rows, feature):

        if type(rows[0][feature]) == str: # Categorical feature
            return set(row[feature] for row in rows), True

        else: # Continuous feature
            sorted_elements = sorted(set(row[feature] for row in rows))
            candidates = []
            for i in range(len(sorted_elements) - 1):
                midpoint = (sorted_elements[i] + sorted_elements[i+1])/2
                candidates.append(midpoint)

            return candidates, False


    # Method to calculate the Gini Impurity of a given set of rows
    def _gini_impurity(self, rows):

        return 0
    
    # Method to calculate the entropy of a given set of rows
    def _entropy(self, rows):
        return 0

    
'''
-----> Main function call serving as the entry point <-----
'''

def main():



    return

if __name__ == "__main__":
    main()