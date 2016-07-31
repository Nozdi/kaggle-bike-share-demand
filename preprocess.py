from datetime import date

import numpy as np


def calculate_period(timestamp):
    initial_date = date(2011, 1, 1)
    current_date = timestamp.date()
    return (
        (current_date.year - initial_date.year) * 12 +
        (current_date.month - initial_date.month)
    )


def create_datetime_features(df):
    df['month'] = df.datetime.map(lambda ts: ts.date().month)
    df['week_day'] = df.datetime.map(lambda ts: ts.date().isoweekday())
    df['week_number'] = df.datetime.map(lambda ts: ts.date().isocalendar()[1])
    df['hour'] = df.datetime.map(lambda ts: ts.hour)
    df['year'] = df.datetime.map(lambda ts: ts.date().year)
    return df


def preprocess(df):
    df = create_datetime_features(df)
    df['period'] = df.datetime.map(calculate_period)
    if 'count' in df.columns:
        df['log_count'] = np.log1p(df['count'])
        df['log_registered'] = np.log1p(df['registered'])
        df['log_casual'] = np.log1p(df['casual'])
    return df
