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


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class BaseClassifier():

    def __init__(self, df, features, impute=True):
        
        self.df = df
        self.features = features
        self.impute = impute

    def Impute(self):
        pass


########################################
# Imputing
########################################

def ImputeValue(df, feature, value):
    df[feature + '_notimputed'] = df[feature]
    df[feature].fillna(value, inplace=True)
    return df
    
def ImputeMedian(df, feature, verbose=0):
    median = np.median(df[df[feature].notnull()][feature])
    df = ImputeValue(df, feature, median)
    if verbose > 0: print "'" + feature + "' : '" + str(median) + ','

    return df

def ImputeCategorical(df, feature, col,  method='mode', verbose=0):
    category_cols = [x for x in df.columns if col in x]
    if method == 'separate_na_category':
        return df
    elif method =='mode':
        for cur_col in category_cols:
            df = ImputeValue(df, cur_col, np.round(df[cur_col].mean()))
            if verbose > 0: print "'" + cur_col +  "' : '" + str(np.round(df[cur_col].mean())) + ','
        return df
    elif method == 'none': # do nothing
        return df

def Impute(df, features, cat_imputation_method='mode', verbose=0):
    imputed = []
    for feature in features:
        col = 'flat_size'
        if feature == col:
            df = ImputeMedian(df, feature, verbose=verbose)
            imputed.append(col)
            
        col = 'mates_minage'
        col2 = 'mates_maxage'
        if (feature == col) | (feature == col2):
            df = ImputeMedian(df, feature, verbose=verbose)
            imputed.append(col)
            
        col = 'minage'
        col2 = 'maxage'
        if (feature == col) | (feature == col2):
            df = ImputeMedian(df, feature, verbose=verbose)
            imputed.append(col)
            
        col = 'duration'
        if feature == col:
            df = ImputeValue(df, feature, 10000) # unbegrenzte mietdauer, 10000 days
            imputed.append(col)
            
        col = 'bathroom'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)

        col = 'heating'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)
            
        col = 'languages'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)

        col = 'floor'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)

        col = 'TV'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)
            
        col = 'other'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)
            
        col = 'parking'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)

        col = 'phone'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)

        col = 'smoking'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)
     
        col = 'wg_type'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)

        col = 'reverse_suburb'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)
            
        col = 'house_info'     
        if (col in feature) and not(col in imputed):
            df = ImputeCategorical(df, feature, col, cat_imputation_method, verbose=verbose)
            imputed.append(col)     
    return df

########################################
# stuff for plotting
########################################

    
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
        subfeature_names = [x[1] for x in subfeatures]
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


    

class ClickInfo(mpld3.plugins.PluginBase):
    """Plugin for getting info on click"""
    
    JAVASCRIPT = """
    mpld3.register_plugin("clickinfo", ClickInfo);
    ClickInfo.prototype = Object.create(mpld3.Plugin.prototype);
    ClickInfo.prototype.constructor = ClickInfo;
    ClickInfo.prototype.requiredProps = ["id"];
    ClickInfo.prototype.requiredProps = ["text_data"];
    function ClickInfo(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };
    
    ClickInfo.prototype.draw = function(){
        var obj = mpld3.get_element(this.props.id);
        var test_text2 = this.props.text_data;
        var objid = this.props.id
        obj.elements().on("mousedown",
                          function(d, i){
    alert(test_text2[i]);
    window.open("file:///home/jefe/Dropbox/work/DataScienceRetreat2015/project/wggesucht/wg_pages/" + test_text2[i],'_blank');
   // console.log(d)
   // console.log(i)
    //console.log(test_text2[i])
    });
    }
    """
    def __init__(self, points, text_data):
        self.dict_ = {"type": "clickinfo",
                      "id": mpld3.utils.get_id(points),
                      "text_data": text_data
                  }


# Define some CSS to control our custom labels
css = """
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #000000;
}
td
{
  background-color: #cccccc;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: right;
}
.mpld3-figure path {
    pointer-events: none;
}
.mpld3-figure path.mpld3-path {
    pointer-events: auto;
}
"""
def ScatterLinksTooltips(df, x, y, tooltip_cols, filename, colors = 'blue',dontsave=False, \
                         dotsize=50, alpha = 0.2, add_line=True,  ax=[], fh=[], **kwargs):
    
    if ax == []:
        ax = plt.gca()
    if fh == []:
        fh = plt.gcf()


    ax.grid(color='white', linestyle='solid')

    if isinstance(x,str):
        xlabel = x
        ylabel = y
        x = df[x]
        y = df[y]
    else:
        xlabel = 'x'
        ylabel = 'y'
        
        
    labels = []
    for i in x.index:
        label = df.loc[[i]][tooltip_cols].T
        label.columns = ['index {0}'.format(i)]
        # .to_html() is unicode; so make leading 'u' go away with str()
        labels.append(str(label.to_html().encode('utf-8')))

    points = ax.scatter(x, y, alpha=alpha,s=dotsize, c=colors,  **kwargs)

    if add_line:
        xx = np.linspace(*ax.get_xlim())
        ax.plot(xx, xx)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    tooltip = mpld3.plugins.PointHTMLTooltip(points, labels,
                                      voffset=10, hoffset=10, css=css)
    try:
        text_data = list(df.loc[x.index].link.apply(lambda x: x.split('wg-zimmer-in-Berlin-')[-1]).values)
    except:
        text_data = [0]
    mpld3.plugins.connect(fh, tooltip)
    mpld3.plugins.connect(fh, ClickInfo(points, text_data))

    if dontsave == False:
        mpld3.save_html(fh, filename)



    

from matplotlib.font_manager import FontProperties

def SavePlot(title='', ax=[], fh=[]):
    if ax == []:
        ax = plt.gca()
    if fh == []:
        fh = plt.gcf()

    if title=='': raise ValueError('you must specify a filename!')
    title = title.replace(' ','_')
    plt.savefig('plots/' +title + '.png', bbox_inches='tight')
    plt.savefig('plots/' +title + '.eps', bbox_inches='tight')

    
def FormatPlot(title='', ax=[], fh=[],colorbar=0):

    if ax == []:
        ax = plt.gca()
    if fh == []:
        fh = plt.gcf()
        
# replace ticks
    def FormatLabels(string):
        string = string.replace('_',' ')
        string = string.replace('map lat','location: latitude')
        string = string.replace('map lon','location: longitude')
        string = string.replace('duration','duration of rental')
        string = string.replace('mates minage','min age flatmates')
        string = string.replace('mates maxage','max age flatmates')
        string = string.replace('minage','min age wanted')
        string = string.replace('maxage','max age wanted')
        string = string.replace('wg type','type: ')
        string = string.replace('other','other: ')
        string = string.replace('parking','parking: ')
        string = string.replace('heating','heating: ')
        string = string.replace('smoking','')
        string = string.replace('bathroom','bathroom: ')
        string = string.replace('floor','floor: ')
        string = string.replace('language','languages: ')
        string = string.replace('Partial','partial')


        return string


    fontP_ticks = FontProperties(size=17)
    fontP_txt = FontProperties(size=19)
    fontP_title = FontProperties(size=21)
        
    plt.setp(fh, facecolor='w')

    if colorbar != 0:
        cbarlabels = [x.get_text() for x in colorbar.ax.yaxis.get_ticklabels()]
        plt.setp( colorbar.ax.yaxis.get_ticklabels(),
                 fontproperties=fontP_ticks)

    if not(isinstance(ax,list)):
        ax = [ax]
        
    for cur_ax in ax:
        xlabels = [x.get_text() for x in cur_ax.get_xticklabels()]
        ylabels = [x.get_text() for x in cur_ax.get_yticklabels()]
        newylabels = [FormatLabels(x) for x in ylabels]
        newxlabels = [FormatLabels(x) for x in xlabels]
        newylabel = FormatLabels(cur_ax.get_ylabel()) 
        newxlabel = FormatLabels(cur_ax.get_xlabel()) 

        cur_ax.set_ylabel(newylabel)
        cur_ax.set_xlabel(newxlabel)

        if ylabels[0] != '':
            cur_ax.set_yticklabels(newylabels)
        if xlabels[0] != '':
            cur_ax.set_xticklabels(newxlabels)


        if title != '':
            cur_ax.set_title(title)
        plt.setp(cur_ax.title, fontproperties=fontP_title)

        plt.setp( cur_ax.get_xticklabels()
                + cur_ax.get_yticklabels(),
                 fontproperties=fontP_ticks)
        try:
            plt.setp( cur_ax.get_legend().get_texts(),
                     fontproperties=fontP_txt)
        except:
            pass
        try:
            plt.setp([cur_ax.xaxis.label, cur_ax.yaxis.label],
                    fontproperties=fontP_txt)
        except:
            pass
       # fh.tight_layout()
    plt.show(block=False)

def EvaluateModel(y_test, y_train, test_predicted, valid_predicted, test, train_predicted, target):

    if target == 'log_price':
        print("MAE train/valid/test: %.1f/%.1f/%.1f" % (mean_absolute_error(np.exp(y_train), np.exp(train_predicted)), \
                                                    mean_absolute_error(np.exp(y_test), np.exp(valid_predicted)), \
                                                    mean_absolute_error(np.exp(test[target]), np.exp(test_predicted))   ))

        print("MAPE train/valid/test: %.1f/%.1f/%.1f" % (mean_absolute_percentage_error(np.exp(y_train), np.exp(train_predicted)), \
                                                    mean_absolute_percentage_error(np.exp(y_test), np.exp(valid_predicted)), \
                                                    mean_absolute_percentage_error(np.exp(test[target]), np.exp(test_predicted))   ))

        return mean_absolute_percentage_error(np.exp(y_test), np.exp(valid_predicted))
    else:
        print("MAE train/valid/test: %.1f/%.1f/%.1f" % (mean_absolute_error(y_train, train_predicted), \
                                                    mean_absolute_error(y_test, valid_predicted), \
                                                    mean_absolute_error(test[target], test_predicted)   ))

        print("MAPE train/valid/test: %.1f/%.1f/%.1f" % (mean_absolute_percentage_error(y_train, train_predicted), \
                                                    mean_absolute_percentage_error(y_test, valid_predicted), \
                                                    mean_absolute_percentage_error(test[target], test_predicted)   ))

        return mean_absolute_percentage_error((y_test), (valid_predicted))
def EvaluateModelCV(train, train_predicted, cv_predicted, test,  test_predicted, target):

       
    if target == 'log_price':
        
        print("MAE train/CV/test: %.1f/%.1f/%.1f" % (mean_absolute_error(np.exp(train[target]), np.exp(train_predicted)), \
                                                     mean_absolute_error(np.exp(train[target]), np.exp(cv_predicted)), \
                                                    mean_absolute_error(np.exp(test[target]), np.exp(test_predicted))   ))

        mape_train = mean_absolute_percentage_error(np.exp(train[target]), np.exp(train_predicted))
        mape_cv = mean_absolute_percentage_error(np.exp(train[target]), np.exp(cv_predicted))
        mape_test =  mean_absolute_percentage_error(np.exp(test[target]), np.exp(test_predicted))
        
        print("MAPE train/CV/test: %.1f/%.1f/%.1f" % (mape_train,mape_cv,mape_test)) 
        return (mape_train,mape_cv,mape_test)

    else:
        print("MAE train/CV/test: %.1f/%.1f/%.1f" % (mean_absolute_error(y_train, train_predicted), \
                                                    mean_absolute_error(y_test, cv_predicted), \
                                                    mean_absolute_error(test[target], test_predicted)   ))

        mape_train = mean_absolute_error(np.exp(train[target]), np.exp(train_predicted))
        mape_cv = mean_absolute_error(np.exp(train[target]), np.exp(cv_predicted))
        mape_test =  mean_absolute_error(np.exp(test[target]), np.exp(test_predicted))
        
        print("MAPE train/CV/test: %.1f/%.1f/%.1f" % (mape_train,mape_cv,mape_test)) 
        return (mape_train,mape_cv,mape_test)
        
class AgeImputer(TransformerMixin, BaseEstimator):

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"mode": self.mode}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
            return self
            
    def __init__(self,mode='knn', doplot=False):
        """Imputing Age
        """
        self.mode=mode
        
    def fit(self, X, y=None):
        notna = X.age.notnull()
        if self.mode == 'knn':
            self.knn = neighbors.KNeighborsRegressor(500, weights='uniform')
            self.knn.fit(X[notna][['mileage']], X[notna].age)
        elif self.mode == 'median':
            self.fillval = X[notna].age.median()
        elif self.mode == 'mean':
            self.fillval = X[notna].age.mean()
                       
        return self

    def transform(self, X, y=None):
        isna = X.age.isnull()
        df = X.copy()
        if self.mode == 'knn':
            predicted  = self.knn.predict(df[['mileage']][isna])
        else:
            predicted = self.fillval
        df.loc[isna, 'age'] = predicted
        return df


class SeparateModels(RegressorMixin, BaseEstimator):
    def __init__(self, estimator):
        from  sklearn.base import clone
        self.model1 = estimator
        self.model2 = clone(estimator)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"estimator": self.model1}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
            return self
            
    def fit(self, X, y):
        
        is_25100 = X.is_25100.astype(bool)
        self.model1 = self.model1.fit(X[is_25100].drop('is_25100',axis=1) ,y[is_25100])
        self.model2 = self.model2.fit(X[~is_25100].drop('is_25100',axis=1) ,y[~is_25100])
        # import ipdb; ipdb.set_trace()   
        return self

    def predict(self, X):
        is_25100 = X.is_25100.astype(bool)
        #import ipdb; ipdb.set_trace()
        y = np.zeros(X.shape[0])
        y[is_25100.values] = self.model1.predict(X[is_25100].drop('is_25100',axis=1))
        y[~is_25100.values] = self.model2.predict(X[~is_25100].drop('is_25100',axis=1))       
        return y
        
class LinRegResidualRegressor(RegressorMixin, BaseEstimator):

    def __init__(self,estimator=GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.1, loss='huber',random_state=42, max_features=0.5, min_samples_split=1, subsample=0.8)):
        self.estimator = estimator
        
    def fit(self, X, y):
         
        is_25100 = X.is_25100.astype(bool)
        self.lr1 = LinearRegression().fit(X[is_25100] ,y[is_25100])
        self.lr2 = LinearRegression().fit(X[~is_25100] ,y[~is_25100])
        ylr = np.zeros(X.shape[0])
        ylr[is_25100.values] = self.lr1.predict(X[is_25100])
        ylr[~is_25100.values] = self.lr2.predict(X[~is_25100])

        residuals = y - ylr
        self.estimator.fit(X, residuals)
        return self

    def predict(self, X):
        is_25100 = X.is_25100.astype(bool)
        #import ipdb; ipdb.set_trace()
        ylr = np.zeros(X.shape[0])
        ylr[is_25100.values] = self.lr1.predict(X[is_25100])
        ylr[~is_25100.values] = self.lr2.predict(X[~is_25100])
        residuals = self.estimator.predict(X)
        y = ylr + residuals                                         
        return y

        
class PandasSelector(TransformerMixin, BaseEstimator):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, x, y = None):
        return self

    def transform(self, x):
        return x[self.cols]
          