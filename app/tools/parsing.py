from typing import List
import pandas as pd


def parsing_data(path: str) -> (pd.DataFrame, List):
    # suppose that each set of features will tab-separated
    input_df = pd.read_csv(path, sep='\t')
    feature_sets = []

    for col in input_df.columns[1:]:
        # get feature_set
        tmp_df = input_df[col].str.split(',', 1, expand=True)
        feature_set = 'feature_' + str(tmp_df.iloc[0, 0])
        feature_sets.append(feature_set)

        # create features column with naming
        tmp_features = tmp_df.iloc[:, 1].str.split(',', expand=True).astype('int').add_prefix(feature_set + '_')

        # drop column with raw data
        input_df.drop(columns=[col], inplace=True)

        input_df = pd.concat([input_df, tmp_features], axis=1)

    return input_df, feature_sets
