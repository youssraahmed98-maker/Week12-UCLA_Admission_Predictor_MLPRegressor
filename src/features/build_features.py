import pandas as pd
import pandas as pd


def build_features(df):
    # Store the processed dataset in data/processed
    df.to_csv('data/processed/Processed_Admission_Dataset.csv', index=None)

    # Separate the input features and target variable
    X = df.drop('Admit_Chance', axis=1)
    y = df['Admit_Chance']

    return X, y