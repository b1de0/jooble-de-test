from typing import List
import pandas as pd


class BaseTransformer:

    def __init__(self):
        pass

    @staticmethod
    def get_temp_data(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        features = df.columns.str.contains(feature_set)
        tmp = df.loc[:, features]
        return tmp

    def get_column_names(self, df: pd.DataFrame, feature_set: str):
        pass

    def calc_feature_set(self, df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        pass

    def calc(self, df_input: pd.DataFrame, df_output: pd.DataFrame, feature_set_list: List) -> pd.DataFrame:
        for feature_set in feature_set_list:
            df_output = pd.concat([df_output, self.calc_feature_set(df_input, feature_set)], axis=1)
        return df_output


class ZScore(BaseTransformer):

    def get_column_names(self, df: pd.DataFrame, feature_set: str):
        return {col: feature_set + '_stand_' + col.rsplit('_', 1)[-1] for col in df.columns}

    def calc_feature_set(self, df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        tmp = self.get_temp_data(df, feature_set)
        new_column_names = self.get_column_names(df, feature_set)

        mu = tmp.mean(axis=0)
        sigma = tmp.std(axis=0)
        tmp = (tmp - mu) / sigma

        tmp.rename(columns=new_column_names, inplace=True)
        return tmp


class MaxFeatureSetIndex(BaseTransformer):

    def get_column_names(self, df: pd.DataFrame, feature_set: str):
        return 'max_' + feature_set + '_index'

    def calc_feature_set(self, df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        tmp = self.get_temp_data(df, feature_set)
        column_name = self.get_column_names(df, feature_set)

        max_feature_name = tmp.idxmax(axis=1)
        max_feature_index = max_feature_name.str.split('_', -1, expand=True).iloc[:, -1].astype(int)

        tmp[column_name] = max_feature_index
        return tmp[column_name]


class MaxFeatureSetAbsMeanDiff(BaseTransformer):

    def get_column_names(self, df: pd.DataFrame, feature_set: str):
        return 'max_' + feature_set + '_abs_mean_diff'

    def calc_feature_set(self, df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        tmp = self.get_temp_data(df, feature_set)
        column_name = self.get_column_names(df, feature_set)

        all_feature_mean = tmp.mean(axis=0).to_dict()
        max_feature_name = tmp.idxmax(axis=1)
        max_feature_value = tmp.max(axis=1)
        max_feature_mean = max_feature_name.apply(lambda x: all_feature_mean[x])

        tmp[column_name] = max_feature_value.values - max_feature_mean
        return tmp[column_name]
