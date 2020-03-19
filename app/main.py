import sys
import pandas as pd
from loguru import logger
from tools import parsing_data, z_score

logfile = "logs/running_process.log"
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
logger.add(logfile)

TRAIN_DATA_DIR = "data/train.tsv"
TEST_DATA_DIR = "data/test.tsv"
OUTPUT_FILE = "output/test_proc.tsv"


def processing():
    logger.info('Reading data.')
    train_data, train_feature_set_patterns = parsing_data(TRAIN_DATA_DIR)
    test_data, test_feature_set_patterns = parsing_data(TEST_DATA_DIR)
    logger.info('Done parsing data.')
    z_score(test_data, feature_set=test_feature_set_patterns[0])

    train_data.to_csv("output/train.csv", index=False)
    test_data.to_csv("output/test.csv", index=False)

    return True


if __name__ == '__main__':
    processing()