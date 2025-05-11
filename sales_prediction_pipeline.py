
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data():
    test = pd.read_csv("test.csv", parse_dates=["date"])
    stores = pd.read_csv("stores.csv")
    oil = pd.read_csv("oil.csv", parse_dates=["date"])
    holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])
    return test, stores, oil, holidays

def preprocess(test, stores, oil, holidays):
    test = test.merge(stores, on='store_nbr', how='left')

    oil['dcoilwtico_interpolated'] = oil['dcoilwtico'].interpolate()
    oil['oil_above_70'] = (oil['dcoilwtico_interpolated'] > 70).astype(int)
    test = test.merge(oil[['date', 'dcoilwtico_interpolated', 'oil_above_70']], on='date', how='left')

    holidays = holidays[(holidays['transferred'] == False)]
    holidays = holidays.groupby(['date', 'locale_name'])['description'].agg(lambda x: ','.join(set(x))).reset_index()
    holidays = pd.pivot_table(holidays, values='description', index='date', columns='locale_name', aggfunc=lambda x: 1, fill_value=0)
    holidays.columns = ['holiday_' + col.lower() + '_binary' for col in holidays.columns]
    holidays.reset_index(inplace=True)
    test = test.merge(holidays, on='date', how='left')
    test.fillna(0, inplace=True)

    test['month'] = test.date.dt.month
    test['day_of_month'] = test.date.dt.day
    test['day_of_year'] = test.date.dt.dayofyear
    test['week_of_month'] = test.date.dt.day // 7 + 1
    test['week_of_year'] = test.date.dt.isocalendar().week.astype(int)
    test['day_of_week'] = test.date.dt.weekday
    test['year'] = test.date.dt.year
    test['is_wknd'] = test.day_of_week > 4
    test['quarter'] = test.date.dt.quarter
    test['is_month_start'] = test.date.dt.is_month_start.astype(int)
    test['is_month_end'] = test.date.dt.is_month_end.astype(int)
    test['is_quarter_start'] = test.date.dt.is_quarter_start.astype(int)
    test['is_quarter_end'] = test.date.dt.is_quarter_end.astype(int)
    test['is_year_start'] = test.date.dt.is_year_start.astype(int)
    test['is_year_end'] = test.date.dt.is_year_end.astype(int)
    test['season'] = test['month'] % 12 // 3 + 1
    test['workday'] = ~test['is_wknd']
    test['wageday'] = test['day_of_week'].isin([4, 5, 6]).astype(int)

    cat_cols = ['family', 'city', 'state', 'type']
    for col in cat_cols:
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col])

    return test

def predict(test_df):
    model = joblib.load("lgb_model.txt")
    feature_cols = [col for col in test_df.columns if col not in ['id', 'date', 'sales']]
    test_df['sales'] = np.expm1(model.predict(test_df[feature_cols]))
    return test_df[['id', 'sales']]

if __name__ == "__main__":
    test, stores, oil, holidays = load_data()
    test_df = preprocess(test, stores, oil, holidays)
    predictions = predict(test_df)
    predictions.to_csv("predictions.csv", index=False)
    print("✅ Predictions saved to predictions.csv")
