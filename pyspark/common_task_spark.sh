#!/bin/bash

export PYTHONIOENCODING="utf8"
export JAVA_HOME=""
export SPARK_HOME=""
SPARK_SUBMIT="bin/spark-submit"
APP_NAME="${1}"

zip -j dependicies.zip util/*
py_file='./dependicies.zip'
${SPARK_HOME}/${SPARK_SUBMIT} \
  --master yarn \
  --name "${APP_NAME}" \
  --executor-memory 8G \
  --conf "spark.app.name=${APP_NAME}" \
  --py-files $py_file\
	$*
