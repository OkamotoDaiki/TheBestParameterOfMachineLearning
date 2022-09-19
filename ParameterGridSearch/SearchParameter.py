from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from ParameterGridSearch.ParameterGridList import ParameterGrid

def GridSearch(X, y, cv=5):
    """
    Search parameter with pipe line.

    Attributes:
        X: training data
        y: label data
        cv: number of times of cross-validation
    
    Returns:
        grid_search: object of GridSearchCV. When you enter the next script, you can see test score.
            "grid_search.score(X_test, y_test)"
    """
    dummy_classifier = SVC()
    pipe = Pipeline([("preprocessing", None), ("classifier", dummy_classifier)])
    pipe.fit(X, y)
    param_grid = ParameterGrid.param_grid
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=cv)
    grid_search.fit(X, y)
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross_validation score: {:.2f}".format(grid_search.best_score_))
    return grid_search