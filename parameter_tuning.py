#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.grid_search import ParameterGrid

from joblib import (
    Parallel,
    delayed,
)

from preprocess import preprocess
from metric import rmsle


RF_PARAMTER_GRID = GBR_PARAMTER_GRID = {
    'n_estimators': [400, 800],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 0.8],
    'random_state': [1],
}

FEATURES = [
    'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'windspeed', 'month', 'hour', 'year', 'period', 'week_day'
]
N_CORES = 4


df = pd.read_csv('train.csv', parse_dates=['datetime'])
df = preprocess(df)


def cv_model(df, scaled_target, target, model_cls, config):
    current_results = []
    base_estimator = model_cls(**config)

    for train_idx, test_idx in ShuffleSplit(len(df), n_iter=10, test_size=0.3):
        base_estimator.fit(
            df.iloc[train_idx][FEATURES], df.iloc[train_idx][scaled_target])
        y_pred = np.expm1(base_estimator.predict(df.iloc[test_idx][FEATURES]))
        y_true = df.iloc[test_idx][target]
        current_results.append(rmsle(y_pred, y_true))

    return {
        'config': str(config),
        'model_name': model_cls.__name__,
        'mean_rmsle': np.mean(current_results),
        'std_rmsle': np.std(current_results)
    }


def grid_search_cv(df, scaled_target, target, model_cls, grid):
    return Parallel(n_jobs=N_CORES)(
        delayed(cv_model)(
            df=df,
            scaled_target=scaled_target,
            target=target,
            model_cls=model_cls,
            config=config,
        )
        for config in ParameterGrid(grid)
    )


if __name__ == '__main__':
    gbr_registered_results = grid_search_cv(
        df, 'log_registered', 'registered',
        GradientBoostingRegressor, GBR_PARAMTER_GRID)

    rf_registered_results = grid_search_cv(
        df, 'log_registered', 'registered',
        RandomForestRegressor, RF_PARAMTER_GRID)

    reg_df = pd.DataFrame(gbr_registered_results + rf_registered_results).sort_values(
        ['mean_rmsle', 'std_rmsle'], ascending=[True, True]
    )
    reg_df.to_csv('registered.csv', index=False)

    gbr_casual_results = grid_search_cv(
        df, 'log_casual', 'casual',
        GradientBoostingRegressor, GBR_PARAMTER_GRID)
    rf_casual_results = grid_search_cv(
        df, 'log_casual', 'casual',
        RandomForestRegressor, RF_PARAMTER_GRID)

    cas_df = pd.DataFrame(gbr_casual_results + rf_casual_results).sort_values(
        ['mean_rmsle', 'std_rmsle'], ascending=[True, True]
    )
    cas_df.to_csv('casual.csv', index=False)

    print reg_df.iloc[0]
    print cas_df.iloc[0]
