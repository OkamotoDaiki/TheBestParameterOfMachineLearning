# TheBestParameterOfMachineLearning
 
This is the tool to run some machinelearning algorithms with pipeline of sckit-learn.
If you overlide the parameter of ClassifierParameter.py, you can change the parameter about machine-learning.
 
# Features
 
What is setting by defalut is, 
* Support Vecotor Machine(C=0.001\~100(logarithm steps), gamma=0.001\~100(logarithm steps))<br>
Preprocessing -> [MinMaxScaler, StanderdScaler, None]
* Random Forest Classifier(max_depth=1\~10, max_features=1\~10, n_estimators=1\~100(logarithm steps)), 
* Gradient Boosting Classifier(max_depth=1\~10, learning_rate=0.01\~10(logarithm stemps)).

If you want to update parameters, to overlide ClassifierParameter.py
 
# Requirement

* Python 3.8.10
* numpy 1.21.4
* sckit-learn 1.0.1
 
# Installation
 
You can import it with the following program.
Refer to the sample.py
 
```python
from ParameterGridSearch.SearchParameter import GridSearch

X_train = []
X_test = []
y_train = []
y_test = []

param_grid = GridSearch(X_train, y_train)
print(param_grid.score(X_test, y_test))
```
 
# Usage
 
Refer to the sample program.

# Author
* Oka.D.
* okamotoschool2018@gmail.com
