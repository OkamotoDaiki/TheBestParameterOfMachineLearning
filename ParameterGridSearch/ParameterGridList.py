from ParameterGridSearch.ClassifierParameter import ParameterSVC
from ParameterGridSearch.ClassifierParameter import ParameterRandomForestClassifier
from ParameterGridSearch.ClassifierParameter import ParameterGradientBoostingClassifier

class ParameterGrid:
    """
    You can get object to need parameter grid.
    Refer ClassifierParameter.py when you want to rewrite parameter of machine learning.
    """
    param_grid = [ParameterSVC.SVC, 
                    ParameterRandomForestClassifier.rfc, 
                    ParameterGradientBoostingClassifier.gbc]