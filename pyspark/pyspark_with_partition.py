# This Python file uses the following encoding: utf-8
from __future__ import division

import sys
import argparse
import logging
import pandas as pd
from collections import Counter

from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_TABLE = '1'
INSERT_TABLE = '1'

INPUT_PATH_1 = '1'
OUTPUT_PATH_1 = '1'

WEIGHTS = [1.0] * 10
SEED = 42
NUM_PARTITIONS = 2048
SHUFFLE_PARTITIONS = '2048'
DEFAULT_PARALLELISM = '2048'
BROADCAST_TIMEOUT = '36000'
COALESCE_PARTITIONS = 256

def process(args):
    spark = SparkSession.builder \
        .enableHiveSupport() \
        .config('hive.exec.dynamic.partition', 'true') \
        .config('hive.exec.dynamic.partition.mode', 'nonstrict') \
        .config('spark.io.compression.codec', 'lz4') \
        .config('spark.sql.shuffle.partitions', SHUFFLE_PARTITIONS) \
        .config('spark.default.parallelism', DEFAULT_PARALLELISM) \
        .config("spark.sql.broadcastTimeout", BROADCAST_TIMEOUT) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    logging.info('################# Start preprocessing. #################')

    def process_chunk(df):
        pass

    splits = df.randomSplit(WEIGHTS)
    num = len(WEIGHTS)
    for index, sub in enumerate(splits):
        sub = sub.repartition(NUM_PARTITIONS)
        logging.info('Split: {} of {}'.format(index + 1, num))
        process_chunk(sub)

    logging.info('############ Finishing preprocessing. ############')
    spark.stop()


def main():
    parser = argparse.ArgumentParser(description='.')
    args = parser.parse_args()
    process(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
