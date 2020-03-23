from typing import List
import dask.dataframe as dd


def parsing_data(path: str) -> (dd.DataFrame, List):
    # suppose that each set of features will tab-separated
    input_df = dd.read_csv(path, sep='\t')
    feature_sets = []

    for col in input_df.columns[1:]:
        # get feature_set
        feature_set_number = int(input_df[col].str.partition(',')[0].astype(int).mean().compute())
        input_df[col] = input_df[col].str.partition(',')[2]
        feature_set = 'feature_' + str(feature_set_number)
        feature_sets.append(feature_set)

        # create features column with naming
        n_features = len(input_df.loc[0, col].compute()[0].split(','))

        def splitting(ddf: dd.DataFrame, s: dd.Series, n, i=0):
            if i < n:
                ddf[f'{feature_set}_{i}'] = s.str.partition(',')[0].astype(int)
                i += 1
                return splitting(ddf, s.str.partition(',')[2], n, i)
            else:
                return ddf

        splitting(input_df, input_df[col], n_features)
        # drop column with raw data
        input_df = input_df.drop(columns=[col])

    return input_df, feature_sets
