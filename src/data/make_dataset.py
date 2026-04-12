import pandas as pd


def load_and_preprocess_data(data_path):
    # Import the data from 'Admission.csv'
    df = pd.read_csv(data_path)

    # Handle missing values (if any exist)
    df['GRE_Score'].fillna(df['GRE_Score'].median(), inplace=True)
    df['TOEFL_Score'].fillna(df['TOEFL_Score'].median(), inplace=True)
    df['University_Rating'].fillna(df['University_Rating'].mode()[0], inplace=True)
    df['SOP'].fillna(df['SOP'].median(), inplace=True)
    df['LOR'].fillna(df['LOR'].median(), inplace=True)
    df['CGPA'].fillna(df['CGPA'].median(), inplace=True)
    df['Research'].fillna(df['Research'].mode()[0], inplace=True)
    df['Admit_Chance'].fillna(df['Admit_Chance'].median(), inplace=True)

    # Drop unnecessary column
    if 'Serial_No' in df.columns:
        df = df.drop('Serial_No', axis=1)

    return df