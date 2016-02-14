import xgboost as xgb
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin
import mpld3
from sklearn import neighbors
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

    
import matplotlib.pyplot as plt
def PlotFeatureImportance(clf, features, thresh=2,nfeat=0, combined_features=[]):
    if isinstance(clf, xgb.sklearn.XGBRegressor):
        scoredict = clf.booster().get_fscore()
        feature_importance  = np.array([float(v) for k, v in scoredict.items()])
    else:
        feature_importance = clf.feature_importances_

    features2 = np.array([x.decode('utf-8') for x in features])
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    for combined_feature in combined_features:
        subfeatures = [(index, x) for index,x \
                       in enumerate(features) \
                       if (combined_feature + '_') in x]
        
        indizes = [x[0] for x in subfeatures]
        feature_importance = np.append(feature_importance, feature_importance[indizes].sum())
        features2 = np.append(features2, combined_feature)
        feature_importance = np.delete(feature_importance, indizes)
        features2 = np.delete(features2, indizes)

    sorted_idx = np.argsort(feature_importance)
    feature_importance_sorted = feature_importance[sorted_idx]
    feature_names_sorted = np.array(features2)[sorted_idx]
    feature_names_sorted = feature_names_sorted[feature_importance_sorted > thresh]
    feature_importance_sorted = feature_importance_sorted[feature_importance_sorted > thresh]
    if nfeat > 0:
       feature_names_sorted = feature_names_sorted[-nfeat:]
       feature_importance_sorted = feature_importance_sorted[-nfeat:]
    pos = np.arange(feature_names_sorted.shape[0]) + .5
    plt.barh(pos, feature_importance_sorted, align='center')
    plt.yticks(pos, feature_names_sorted)
    plt.xlabel('normalized importance')

def PlotGBRIterations(clf, X_test, y_test):
    # compute test set deviance
    n_estimators = clf.n_estimators
    test_score = np.zeros((n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)

    plt.figure()
    plt.title('Deviance')
    plt.plot(np.arange(n_estimators) + 1, clf.train_score_, 'b-',label='Training Set Deviance')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')


    
