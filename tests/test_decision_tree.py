# tests/test_decision_tree.py

import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.decision_tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as SkLearnDecisionTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class TestDecisionTreeClassifier(unittest.TestCase):

    # Method to set up the testing environment with the correct training / testing data
    def setUp(self):

        # Load Iris dataset 
        iris_path = '../data/iris.data'
        columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

        df = pd.read_csv(iris_path, header=None, names=columns)

        X = df.drop('class', axis=1).values
        y = df['class'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        print()
        return

    # Method to test the accuracy of the custom implemented DecisionTreeClassifier with the gini criterion
    def test_myDecisionTreeClassifier_gini(self):

        # Combine X and Y values for model
        train_data = np.column_stack((self.X_train, self.y_train))

        DecisionTree = DecisionTreeClassifier(criterion='gini')
        DecisionTree.fit(train_data)
        
        predictions = DecisionTree.predict(self.X_test)
        correct = sum(pred == true for pred, true in zip(predictions, self.y_test))
        accuracy = correct / len(self.y_test)

        self.assertGreaterEqual(accuracy, 0.9)

        # Display the accuracy of the model
        print(f"myDecisionTreeClassifier with gini impurity accuracy: {accuracy}")
        return
    
    # Method to test the accuracy of the custom implemented DecisionTreeClassifier with the entropy criterion
    def test_myDecisionTreeClassifer_entropy(self):

        train_data = np.column_stack((self.X_train, self.y_train))

        DecisionTree = DecisionTreeClassifier(criterion='entropy')
        DecisionTree.fit(train_data)

        predications = DecisionTree.predict(self.X_test)
        correct = sum(pred == true for pred, true in zip(self.y_test, predications))
        accuracy = correct / len(self.y_test)
    
        self.assertGreaterEqual(accuracy, 0.9)

        # display the accuracy of the model
        print(f"myDecisionTreeClassifier with entropy accuracy: {accuracy}")

    # Method to test the accuracy of the SciKit-Learn implementation of DecisionTreeClassifier with gini impurity as criterion
    def test_sciKitLearn_DecisionTreeClassifier_gini(self):

        clf = SkLearnDecisionTree(criterion='gini',random_state=42)
        clf.fit(self.X_train, self.y_train)
        predications = clf.predict(self.X_test)
        accuracy = accuracy_score(predications, self.y_test)
        
        # method call to assert whether model has reached the required accuracy
        self.assertGreaterEqual(accuracy, 0.9)

        print(f"sci-learn DecisionTreeClassifier with gini impurity accuracy: {accuracy}")
        return
    
    # Method to test the accuracy of the SciKit-Learn implementation of DecisionTreeClassifier with entropy as criterion
    def test_sciKitLearn_DecisionTreeClassifier_entropy(self):

        clf = SkLearnDecisionTree(criterion='entropy',random_state=42)
        clf.fit(self.X_train, self.y_train)
        predications = clf.predict(self.X_test)
        accuracy = accuracy_score(predications, self.y_test)
        
        # method call to assert whether model has reached the required accuracy
        self.assertGreaterEqual(accuracy, 0.9)

        print(f"sci-learn DecisionTreeClassifier with entropy accuracy: {accuracy}")
        return

if __name__ == "__main__":
    unittest.main()