import pandas as pd
import sklearn.metrics
import sys

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from sklearn.inspection import permutation_importance
from tpot import TPOTClassifier, TPOTRegressor

_autosklearn_default_kwargs = dict(
    resampling_strategy = 'cv',
    resampling_strategy_arguments = {'folds': 10},
    memory_limit = None,
    time_left_for_this_task = 60
)

_tpot_default_kwargs = dict(
    warm_start = True,
    max_time_mins = 1
)

_permutation_importance_default_kwargs = dict(
    n_repeats=10,
    random_state=0
)

class AutoMLModel:
    
    def __init__(self, model=TPOTClassifier, metric=None, metric_positive=None, *args, **kwargs):
        
        # Convert str to model class
        model = getattr(sys.modules[__name__], model) if isinstance(model, str) else model
        
        # Get class name of model
        name = model.__name__ if hasattr(model, '__name__') else type(model).__name__
        
        # Auto determine model type
        if 'class' in name.lower():
            mtype = 'classifier'
        elif 'regress' in name.lower():
            mtype = 'regressor'
        else:
            mtype = 'unknown'
            
        # Auto determine metric
        if not metric:
            metric = 'f1_score' if mtype == 'classifier' else 'r2_score'
            
        # Auto determine if positive metric (higher values better)
        if metric_positive is None:
            if any([txt in metric.lower() for txt in ['error', 'loss', 'deviance']]):
                metric_positive = False
            else:
                metric_positive = True
            
        # Set default kwargs for model
        if name.lower() in ['autosklearnclassifier', 'autosklearnregressor']:
            kwargs = _autosklearn_default_kwargs | kwargs
            group = 'autosklearn'
        elif name.lower() in ['tpotclassifier', 'tpotregressor']:
            kwargs = _tpot_default_kwargs | kwargs
            group = 'tpot'
        else:
            group = 'unknown'
            
        # Set attrs
        self.model = model(*args, **kwargs)
        self.model_name = name
        self.model_group = group
        self.model_type = mtype
        self.model_metric = metric
        self.model_metric_positive = metric_positive
    
    def fit(self, x, y, *args, **kwargs):
        
        # Get numeric data only
        x = x.select_dtypes('number')
        
        # Get input x and output y data for training
        y = x[y] if isinstance(y, str) else y
        x = x[[c for c in x.columns if c != y.name]]
        
        # Train model
        self.model.fit(x, y, *args, **kwargs)
        
        # Refit if autosklearn cv
        if hasattr(self.model, 'resampling_strategy'):
            if self.model.resampling_strategy.lower() == 'cv':
                self.model.refit(x, y)
        
        # Get model details from fitting
        if self.model_group == 'autosklearn':
            self.model_details = pd.DataFrame(self.model.cv_results_)
        if self.model_group == 'tpot':
            self.model_details = pd.DataFrame.from_dict(self.model.evaluated_individuals_, orient='index').reset_index(drop=True)
        
        # Set last fitted input x and output y
        self.last_x = x
        self.last_y = y
        
    def predict(self, x=None, *args, **kwargs):
        
        # Predict on x input
        x = x if x is not None else self.last_x
        out = self.model.predict(x, *args, **kwargs)
        
        # Set attrs and return prediction
        self.last_predicted = out
        self.last_predicted_args = args
        self.last_predicted_kwargs = kwargs
        return out
    
    def score(self, metric=None, y=None, predicted=None, *args, **kwargs):
        
        # Get metrics from sklearn
        metric = metric if metric else self.model_metric
        metric = getattr(sklearn.metrics, metric) if isinstance(metric, str) else metric
        
        # Calculate score with metric
        y = y if y is not None else self.last_y
        predicted = predicted if predicted is not None else self.last_predicted
        out = metric(y, predicted, *args, **kwargs)
        
        # Set attributes and return
        self.last_score = out
        self.last_metric = metric.__name__
        self.last_score_args = args
        self.last_metric_args = kwargs
        return out
    
    def importance(self, x=None, y=None, *args, **kwargs):
        
        # Set data for calculating variable importance
        x = x if x is not None else self.last_x
        y = y if y is not None else self.last_y
        
        # Calculate variable permutation importance
        kwargs = _permutation_importance_default_kwargs | kwargs
        results = permutation_importance(self.model, x, y, *args, **kwargs)
        
        # Get agg var importance into out df and sort
        out = pd.DataFrame({
            'variable': x.columns,
            'importance_mean': results['importances_mean'],
            'importance_std': results['importances_std']
        }).sort_values(by='importance_mean', ascending=False)
        
        # Set attrs and return importance
        self.last_importance = out
        self.last_importance_args = args
        self.last_importance_kwargs = kwargs
        self.last_importance_details = results['importances']
        return out
