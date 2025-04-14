import numpy as np
import pandas as pd

'''
-----> TreeNode class <-----
'''
class TreeNode:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.data_points = 0

    def is_leaf(self):
        return self.value is not None
    



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

    # Initialisation method to set hyperparameters
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaef = min_samples_leaf
        self.root = None
        return
    
    # Method to print the contents of the tree including training data-points at each node
    
    # Method to fit the decision tree classifier to the training data
    def build_tree(self, max_depth=0, min_samples_split=2):
        return

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