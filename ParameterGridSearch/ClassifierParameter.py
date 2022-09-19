from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

class ParameterSVC:
    """
    Parameter of SVC.
    """
    preprocessing = [MinMaxScaler(), StandardScaler(), None]
    C_min = 0.001
    C_max = 100
    _C_samples = int(np.log10(C_max/C_min)) + 1
    gamma_min = 0.001
    gamma_max = 100
    _gamma_samples = int(np.log10(gamma_max/gamma_min)) + 1
    _C = np.geomspace(C_min, C_max, _C_samples)
    _gamma = np.geomspace(gamma_min, gamma_max, _gamma_samples)
    SVC = {"classifier": [SVC()], 
            "preprocessing": preprocessing,
            'classifier__C': _C,
            'classifier__gamma': _gamma
            }


class ParameterRandomForestClassifier:
    """
    Parameter of Random Forest Classifier.
    """
    preprocessing = [None]
    max_depth_min = 1
    max_depth_max = 10
    _max_depth = np.arange(max_depth_min, max_depth_max + 1)
    n_estimators_min = 1
    n_estimators_max = 100
    _n_estimators_samples = int(np.log10(n_estimators_max/n_estimators_min)) + 1
    _n_estimators = [int(i) for i in np.geomspace(n_estimators_min, n_estimators_max, _n_estimators_samples)]
    max_features_min = 1
    max_features_max = 10
    _max_features = np.arange(max_features_min, max_features_max + 1)
    
    rfc = {'classifier': [RandomForestClassifier()],
            'preprocessing': preprocessing, 
            'classifier__max_depth': _max_depth, 
            'classifier__n_estimators': _n_estimators, 
            'classifier__max_features': _max_features
            }


class ParameterGradientBoostingClassifier:
    """
    Parameter of Gradient Boosting Classifier.
    """
    preprocessing = [None]
    max_depth_min = 1
    max_depth_max = 10
    _max_depth = np.arange(max_depth_min, max_depth_max)
    learning_rate_min = 0.01
    learning_rate_max = 10
    _learning_rate_samples = int(np.log10(learning_rate_max/learning_rate_min)) + 1
    _learning_rate = np.geomspace(learning_rate_min, learning_rate_max, _learning_rate_samples)
    gbc = {'classifier': [GradientBoostingClassifier()], 
        'preprocessing': preprocessing, 
        'classifier__max_depth': _max_depth, 
        'classifier__learning_rate': _learning_rate
        }