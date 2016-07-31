#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

from parameter_tuning import FEATURES
from preprocess import preprocess

np.random.seed(1)
df = pd.read_csv('train.csv', parse_dates=['datetime'])
df = preprocess(df)

test_df = pd.read_csv('test.csv', parse_dates=['datetime'])
test_df = preprocess(test_df)

df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

reg_model = GradientBoostingRegressor(
    max_features=0.8,
    min_samples_leaf=2,
    random_state=1,
    n_estimators=800
)
reg_model.fit(df[FEATURES], df['log_registered'])
reg_pred = np.expm1(reg_model.predict(test_df[FEATURES]))

cas_model = GradientBoostingRegressor(
    max_features=0.8,
    min_samples_leaf=1,
    random_state=1,
    n_estimators=800
)
cas_model.fit(df[FEATURES], df['log_casual'])
cas_pred = np.expm1(cas_model.predict(test_df[FEATURES]))

y_pred = reg_pred + cas_pred
y_pred[y_pred < 0] = 0
pd.DataFrame(
    {'datetime': test_df.datetime, 'count': y_pred}
)[['datetime', 'count']].to_csv('my_submission.csv', index=False)
