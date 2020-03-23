import sys
import multiprocessing as mp

from loguru import logger
from tools import parsing_data, ZScore, MaxFeatureSetIndex, MaxFeatureSetAbsMeanDiff
from dask.distributed import Client
import dask.dataframe as dd


logfile = "logs/running_process.log"
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
logger.add(logfile)

TRAIN_DATA_PATH = "data/train.tsv"
TEST_DATA_PATH = "data/test.tsv"
OUTPUT_PATH = "output/tmp"


def processing():
    logger.info('Reading data.')
    # train_data, train_feature_set_list = parsing_data(TRAIN_DATA_PATH)
    test_data, test_feature_set_list = parsing_data(TEST_DATA_PATH)
    logger.info('Done parsing data.')
    test_output = test_data['id_job'].copy()

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

    dd.to_csv(df=test_output, filename=OUTPUT_PATH, index=False, header=True, float_format='%.6f', sep='\t')
    logger.info('saving output - done')
    return True


if __name__ == '__main__':
    logger.info(f"{'*'*30}")
    client = Client(n_workers=mp.cpu_count(), threads_per_worker=2, memory_limit='500MB')
    logger.info(f"client - {client.ncores()}")
    processing()
    client.shutdown()
