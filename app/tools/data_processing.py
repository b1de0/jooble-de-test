from typing import List
import pandas as pd


def parsing_data(path: str) -> (pd.DataFrame, List):
    input_df = pd.read_csv(path, sep='\t', skiprows=1, header=None)
    output_df = pd.DataFrame(columns=['id_job', 'feature_set', 'features'])

    # loop through feature sets and split feture_set_id and features
    for column in input_df.columns[1:]:
        temp = pd.concat(
            [
                input_df.iloc[:, 0].to_frame(),
                input_df.iloc[:, column].str.split(',', 1, expand=True)
            ],
            axis=1)
        temp.columns = ['id_job', 'feature_set', 'features']

        output_df = pd.concat([output_df, temp], axis=0)

    feature_set_patterns = []

    # group by feature_set, split features and create new columns with proper name (example: feature_2_0)
    grouped = output_df.groupby('feature_set')
    for set_id, group in grouped:
        feature_set_name = 'feature' + '_' + set_id
        feature_set_patterns.append(feature_set_name)

        output_df = output_df.join(
            group['features'].str.split(',', -1, expand=True).astype('int').add_prefix(feature_set_name + '_')
        )

    output_df.drop(columns=['features', 'feature_set'], inplace=True)
    # remove duplicates (duplicates are appeared when there are more then one feature set)
    output_df = output_df[~output_df.index.duplicated(keep='first')]

    return output_df, feature_set_patterns
