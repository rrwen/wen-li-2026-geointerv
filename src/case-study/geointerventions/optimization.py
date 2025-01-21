import numpy as np
import pandas as pd
import sys

from . import data
from bayes_opt import BayesianOptimization

# Create optimization parameters from dataframe
def _create_params(x, lconstr, gconstr={}):
    
    # Apply global constraints as query
    x = x.query(gconstr['query']) if gconstr else x
    
    # Construct param df structure
    out = {
        'column': [],
        'row': [],
        'param': [],
        'value': [],
        'bounds': []
    }
    
    # Create params from constraints and df
    for c in lconstr:
        
        # Apply local constraints if avail
        constr = lconstr[c]
        values = x.query(constr['query'])[c] if 'query' in constr else x[c]
        
        # Get rows and convert values to list
        rows = values.index.tolist()
        values = values.tolist()
        
        # Get bounds from data or user defined
        bounds = constr['bounds'] if 'bounds' in constr else (x[c].min(), x[c].max())
        
        # Append parameters and their indices for opt
        out['column'] += [c] * len(values)
        out['row'] += rows
        out['param'] += [f'{c}_{i}' for i, v in zip(rows, values)]
        out['value'] += values
        out['bounds'] += [bounds] * len(values)
        
    # Create params df and return
    out = pd.DataFrame(out)
    out.index = out.param
    return out

# Create optimization function
def _create_func(model, x, y, params, yfunc, mult=1):
    def f(**kwargs):
        
        # Use param values for simulated x
        params.loc[kwargs.keys(), 'value'] = list(kwargs.values())
        rows = params['row'].unique()
        xs = x.loc[rows].copy()
        
        # Assign simulated values to real x values
        for c in params.column.unique():
            p = params[params.column == c]
            xs.loc[p.row, c] = p.value.tolist()
        
        # Run prediction on simulated x input
        predicted = model.predict(xs)
        ys = y.copy()
        ys[rows] = predicted
        out = yfunc(ys) * mult
        return out
    
    return f

class Optimizer:
    
    def __init__(
        self,
        model,
        lconstr,
        optimizer=BayesianOptimization,
        gconstr={},
        x=None,
        y=None,
        yfunc='sum',
        ymax=True,
        infer_bounds=True,
        mult=None,
        *args, **kwargs):
        
        # Get class name and optimizer
        optimizer = getattr(sys.modules[__name__], optimizer) if isinstance(optimizer, str) else optimizer
        name = optimizer.__name__ if hasattr(optimizer, '__name__') else type(optimizer).__name__
        
        # Auto determine x and y model input and output
        x = model.last_x if hasattr(model, 'last_x') and not x else x
        y = model.last_y if hasattr(model, 'last_y') and not y else y
        
        # Create opt params
        params = _create_params(x=x, gconstr=gconstr, lconstr=lconstr)
        params['value_orig'] = params['value']
        
        # Calculate opt bounds
        bounds = {p: b for p, b in zip(params.param, params.bounds)}
        
        # Add opt group and call
        group = 'unknown'
        call = 'unknown'
        if name.lower() in 'bayesianoptimization':
            group = 'bayesian'
            call = 'maximize'
            
        # Auto determine opt type and multiplier
        otype = 'unknown'
        if 'max' in call.lower():
            otype = 'maximizer'
            mult = 1 if ymax else -1 if mult is None else mult
        elif 'min' in call.lower():
            otype = 'minimizer'
            mult = -1 if ymax else 1 if mult is None else mult
            
        # Create opt func
        yfunc = getattr(np, yfunc) if isinstance(yfunc, str) else yfunc
        func = _create_func(model=model, x=x, y=y, params=params, yfunc=yfunc, mult=mult)
        
        # Add kwargs based on group
        if name.lower() in 'bayesianoptimization':
            kwargs['f'] = func
            kwargs['pbounds'] = bounds
        
        # Set attrs
        self.optimizer = optimizer(*args, **kwargs)
        self.optimizer_name = name
        self.optimizer_group = group
        self.optimizer_call = call
        self.optimizer_type = otype
        self.optimizer_func = func
        self.optimizer_yfunc = yfunc
        self.optimizer_ymax = ymax
        self.optimizer_mult = mult
        self.optimizer_gconstr = gconstr
        self.optimizer_lconstr = lconstr
        self.optimizer_params = params
        self.optimizer_bounds = bounds
        self.optimizer_x = x
        self.optimizer_y = y
        self.model = model
        
    def optimize(self, *args, **kwargs):
        
        # Run optimization
        getattr(self.optimizer, self.optimizer_call)(*args, **kwargs)
        
        # Get optimal params from runs so far
        if self.optimizer_name.lower() == 'bayesianoptimization':
            self.optimal = self.optimizer.max
            self.optimal_y = self.optimal['target'] * self.optimizer_mult
            self.optimal_params = self.optimal['params']
            self.optimal_details = pd.DataFrame([{'target': r['target']} | r['params'] for r in self.optimizer.res])