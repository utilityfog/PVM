import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import os

def standardize_dataframe(df, numerical_columns, exclude_columns):
    columns_to_standardize = [col for col in numerical_columns if col not in exclude_columns]
    scaler = StandardScaler()
    standardized_values = scaler.fit_transform(df[columns_to_standardize])
    standardized_df = pd.DataFrame(standardized_values, index=df.index, columns=columns_to_standardize)
    for col in df.columns:
        if col not in columns_to_standardize:
            standardized_df[col] = df[col]
    return standardized_df

def encode_categorical(df, categorical_vars):
    print(f"encodable categorical vars: {categorical_vars}")
    for col in categorical_vars:
        df[col] = df[col].astype('category')
    categorical_df = df[categorical_vars]
    result = pd.get_dummies(categorical_df, dtype=float, drop_first=True)
    return result

def replace_encoded_categorical(df, encoded_categorical_df, categorical_columns):
    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_categorical_df], axis=1)
    return df

def random_sample_per_group(df, group_col):
    return df.groupby(group_col).apply(lambda x: x.sample(1)).reset_index(drop=True)

def is_categorical(df):
    categorical_vars = [col for col in df.columns if df[col].dtype == 'object']
    return categorical_vars

def check_unique_hhidpn(sampled_df, group_col):
    if sampled_df[group_col].duplicated().any():
        print("Some hhidpn values appear more than once.")
    else:
        print("Each hhidpn value appears only once.")

def main(input_file, output_dir):
    hrs = pd.read_csv(input_file)
    pd.set_option('display.max_columns', None)

    columns_to_drop = ['Unnamed: 0', 'nt', 'n2', 'dage_y', 'rarelig']
    hrs_cleaned = hrs.drop(columns=columns_to_drop, errors='ignore')
    hrs_cleaned = hrs_cleaned.replace('', np.nan)
    hrs_cleaned = hrs_cleaned.dropna()

    categorical_columns = is_categorical(hrs_cleaned)
    categorical_columns = [col for col in categorical_columns if col not in ['raedyrs']]

    encoded_df = encode_categorical(hrs_cleaned.copy(), categorical_columns)
    final_df = replace_encoded_categorical(hrs_cleaned, encoded_df, categorical_columns)

    final_df['raedyrs'] = final_df['raedyrs'].replace('17.17+ yrs', '17').astype(int)
    final_df['raedyrs_17plus'] = (final_df['raedyrs'] >= 17).astype(int)

    final_df = random_sample_per_group(final_df, 'hhidpn')
    check_unique_hhidpn(final_df, 'hhidpn')

    columns_to_drop = ['hhidpn', 'iwbeg', 'id']
    final_df = final_df.drop(columns=columns_to_drop, errors='ignore')

    shuffled_columns = np.random.permutation(final_df.columns)
    midpoint = len(shuffled_columns) // 2
    first_half_columns = shuffled_columns[:midpoint]
    second_half_columns = shuffled_columns[midpoint:]

    df_first_half = final_df[first_half_columns]
    df_second_half = final_df[second_half_columns]

    print("First half columns:", df_first_half.columns)
    print("Second half columns:", df_second_half.columns)

    os.makedirs(output_dir, exist_ok=True)
    df_first_half.to_csv(os.path.join(output_dir, 'first_half.csv'), index=False)
    df_second_half.to_csv(os.path.join(output_dir, 'second_half.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and split data')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')

    args = parser.parse_args()

    main(args.input, args.output)