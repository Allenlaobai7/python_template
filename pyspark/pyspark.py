# This Python file uses the following encoding: utf-8
from __future__ import division

import sys
import argparse
import logging
import io, os, yaml
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql.window import Window

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_TABLE = '1'
INSERT_TABLE = '1'

INPUT_PATH_1 = '1'
OUTPUT_PATH_1 = '1'


def process(args):
    spark = SparkSession.builder \
        .enableHiveSupport() \
        .config('hive.exec.dynamic.partition', 'true') \
        .config('hive.exec.dynamic.partition.mode', 'nonstrict') \
        .config('spark.io.compression.codec', 'snappy') \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # read from hive
    query = '''
    SELECT * FROM {0}
    WHERE country == 'RU'
    '''.format(DATA_TABLE)
    df = spark.sql(query).dropDuplicates().persist(StorageLevel.MEMORY_AND_DISK)

    # read from uploaded file
    with io.open(args.config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    # read from hdfs
    file = spark.sparkContext.textFile('hdfs path').collect()

    # read csv
    df_2 = spark.read.csv(INPUT_PATH_1).select(F.explode(F.split('_c0', ' ')).alias('id'), F.col('_c1').alias('count'))\
        .orderBy('count', ascending=False).withColumn('new', F.lit(1)).join(df, ['new'], 'leftanti')
    logging.info('number of rows in df_2: {}'.format(df_2.count()))

    # collect one column
    ids = [id for items in df_2.select('id').rdd.flatMap(lambda x: x).collect() for id in items]

    # sample
    window1 = Window.partitionBy(df['col1']).orderBy(F.desc('col2'))
    df_sampled = df.withColumn('row_num', F.row_number().over(window1)).filter(F.col('row_num') <= 5).drop('row_num')

    # groupby collect struct
    df = df.groupBy('id').agg(F.collect_list(F.struct('col1', 'col2')).alias('col3_list'))

    # output
    df.write.mode('overwrite').insertInto(INSERT_TABLE)
    logging.info('Saving {0} rows to {1}'.format(df.count(), INSERT_TABLE))
    df_2.write.mode('append').csv(OUTPUT_PATH_1)
    spark.stop()


def main():
    parser = argparse.ArgumentParser(description='.')
    args = parser.parse_args()
    process(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
