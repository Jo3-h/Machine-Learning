# src/models/random_forest.py

import pandas as pd
import numpy as np
from decision_tree import DecisionTreeClassifier

"""
-----> Random Forest Tree <------

Inherits from DecisionTreeClassifier but overrides and extends some functionality 
to fit modelling with random forests. 
"""
class RandomForestTree(DecisionTreeClassifier):

    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.root = None
        return

""" 
-----> Random Forest Classifier <-----

Hyperparameters:

"""
class RandomForestClassifier:

    def __init__(self, n_estimators, features=0.5):
        self.features = features
        self.n_estimators = n_estimators
        