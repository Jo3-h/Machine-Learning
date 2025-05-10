** DecisionTreeClassifier Notes **

Implementation of DecisionTreeClassifier was relatively straightforward to construct. Only real issue with it was the handling of data in pandas or numpy. That barrier will continue to get easier as I continue to work with data in this type of environment.

The initial testing of models yielded very high accuracy across both my implementation and the scikit-learn library. I was suspicious of data leakage because of this, however, doing some investigation into the Iris-data file used for testing I found that there is pretty significant distance between features of a given class. As such, both implementations of the classifier is very effectively able to fit unseen data. Even with relatively small training sets.
