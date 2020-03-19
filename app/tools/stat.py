import pandas as pd


def z_score(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    # set features
    features = df.columns.str.contains(feature_set)
    # calc z_score
    df.loc[:, features] -= df.loc[:, features].mean()
    df.loc[:, features] /= df.loc[:, features].std()
    # rename columns
    new_column_names = {col: feature_set + '_stand_' + col.rsplit('_', 1)[-1] for col in df.loc[:, features].columns}
    df.rename(columns=new_column_names, inplace=True)
    return df
