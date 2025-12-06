import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    # Очистка от дат в числовых столбцах
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test

def scale_data(train, val, test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return train_scaled, val_scaled, test_scaled, scaler