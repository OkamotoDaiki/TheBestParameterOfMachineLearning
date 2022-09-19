from ParameterGridSearch.SearchParameter import GridSearch

X_train = []
X_test = []
y_train = []
y_test = []

param_grid = GridSearch(X_train, y_train)
print(param_grid.score(X_test, y_test))