import sys
import pandas as pd
from loguru import logger
from tools import parsing_data, ZScore, MaxFeatureSetIndex, MaxFeatureSetAbsMeanDiff

logfile = "logs/running_process.log"
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
logger.add(logfile)

TRAIN_DATA_DIR = "data/train.tsv"
TEST_DATA_DIR = "data/test.tsv"
OUTPUT_FILE = "output/test_proc.tsv"


def processing():
    logger.info('Reading data.')
    train_data, train_feature_set_list = parsing_data(TRAIN_DATA_DIR)
    test_data, test_feature_set_list = parsing_data(TEST_DATA_DIR)
    logger.info('Done parsing data.')
    test_output = test_data['id_job']

    z_score = ZScore()
    test_output = z_score.calc(df_input=test_data,
                               df_output=test_output,
                               feature_set_list=test_feature_set_list)
    logger.info('z_score - done.')

    max_feature_set_index = MaxFeatureSetIndex()
    test_output = max_feature_set_index.calc(df_input=test_data,
                                             df_output=test_output,
                                             feature_set_list=test_feature_set_list)
    logger.info('max_feature_set_index - done')

    max_feature_set_abs_mean_diff = MaxFeatureSetAbsMeanDiff()
    test_output = max_feature_set_abs_mean_diff.calc(df_input=test_data,
                                                     df_output=test_output,
                                                     feature_set_list=test_feature_set_list)
    logger.info('max_feature_set_abs_mean_diff - done')

    test_output.to_csv(OUTPUT_FILE, index=False, sep='\t')

    return True


if __name__ == '__main__':
    processing()
