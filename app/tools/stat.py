import pandas as pd


def z_score(df_input: pd.DataFrame, df_output: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    # set features
    features = df_input.columns.str.contains(feature_set)
    tmp = df_input.loc[:, features]
    # calc z_score
    tmp -= tmp.mean(axis=0)
    tmp /= tmp.std(axis=0)
    # rename columns
    new_column_names = {col: feature_set + '_stand_' + col.rsplit('_', 1)[-1] for col in tmp.columns}
    tmp.rename(columns=new_column_names, inplace=True)
    return pd.concat([df_output, tmp], axis=1)


def max_feature_set_index(df_input: pd.DataFrame, df_output: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    # set features
    features = df_input.columns.str.contains(feature_set)
    tmp = df_input.loc[:, features]

    max_feature_name = tmp.idxmax(axis=1)
    max_feature_index = max_feature_name.str.split('_', -1, expand=True).iloc[:, -1].astype(int)
    df_output['max_' + feature_set + '_index'] = max_feature_index
    return df_output


def max_feature_set_abs_mean_diff(df_input: pd.DataFrame, df_output: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    # set features
    features = df_input.columns.str.contains(feature_set)
    tmp = df_input.loc[:, features]
    all_feature_mean = tmp.mean(axis=0).to_dict()
    max_feature_name = tmp.idxmax(axis=1)
    max_feature_value = tmp.max(axis=1)
    max_feature_mean = max_feature_name.apply(lambda x: all_feature_mean[x])
    df_output['max_' + feature_set + '_abs_mean_diff'] = max_feature_value.values - max_feature_mean
    return df_output
