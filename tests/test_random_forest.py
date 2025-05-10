# tests/test_random_forest.py

import os
import sys
import unittest
import sklearn
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.random_forest import RandomForestClassifier

class TestRandomForestClassifier(unittest.TestCase):

    # Method to import test / train data and set up the testing environment
    def setUp(self):

        self.criterion = 'gini' # define which criterion the forests will use
        print()
        return
    
    # Method to test accuracy of custom implementation of RandomForestClassifier
    def test_myRandomForestClassifier(self):
        accuracy = 0
        print(f"My RandomForestClassifier accuracy: {accuracy}")
        return
    
    # Method to test accuracy if SciKit-Learn implementation of RandomForestClassifier
    def test_sciKitLearn_RandomForestClassifier(self):
        accuracy = 0
        print(f"Sci-learn RandomForestClassifier accuracy: {accuracy}")
        return
    
if __name__ == "__main__":
    unittest.main()