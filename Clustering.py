import os
import findspark
import pyspark
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

def set_environment_variables():
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop2.7"

def initialize_spark():
    findspark.init()
    findspark.find()
    spark = SparkSession.builder.master('local[*]').appName('flamingo').getOrCreate()
    return spark

def load_and_clean_data(spark, file_path):
    df = spark.read.csv(file_path, sep=',', header=True, inferSchema=True, nullValue='NA')
    cleaned_df = df.na.drop()
    return cleaned_df

def create_dummy_variables(df):
    categories = df.select("platformType").distinct().rdd.flatMap(lambda x: x).collect()
    exprs = [F.when(F.col("platformType") == category, 1).otherwise(0).alias(category) for category in categories]
    dummy_df = df.select("count_buyId", "avg_price", "count_hits", "count_gameclicks", "teamLevel", *exprs)
    return dummy_df

def add_role_column(df):
    df = df.withColumn('Role', when(col('avg_price') > 5, 'Main Role').otherwise('Others'))
    return df

def add_label_column(df):
    df = df.withColumn('label', (df.Role == "Others").cast('integer'))
    return df.drop('avg_price')

def assemble_features(df):
    assembler = VectorAssembler(inputCols=[
        'count_buyId', 'count_hits', 'count_gameclicks', 'teamLevel',
        'iphone', 'android', 'linux', 'mac', 'windows'
    ], outputCol='features')
    assembled_df = assembler.transform(df)
    return assembled_df

def split_data(df, seed=17):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
    return train_df, test_df

def train_decision_tree(train_df):
    tree = DecisionTreeClassifier()
    tree_model = tree.fit(train_df)
    return tree_model

def evaluate_model(model, test_df):
    prediction = model.transform(test_df)
    prediction.select('label', 'prediction', 'probability').show(5, False)
    return prediction

def calculate_metrics(prediction):
    TN = prediction.filter('prediction = 0 AND label = prediction').count()
    TP = prediction.filter('prediction = 1 AND label = prediction').count()
    FN = prediction.filter('prediction = 0 AND label = 1').count()
    FP = prediction.filter('prediction = 1 AND label = 0').count()
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    return accuracy

def main():
    set_environment_variables()
    spark = initialize_spark()
    cleaned_df = load_and_clean_data(spark, './combined-data.csv')
    dummy_df = create_dummy_variables(cleaned_df)
    role_df = add_role_column(dummy_df)
    labeled_df = add_label_column(role_df)
    assembled_df = assemble_features(labeled_df)
    train_df, test_df = split_data(assembled_df)
    tree_model = train_decision_tree(train_df)
    prediction = evaluate_model(tree_model, test_df)
    accuracy = calculate_metrics(prediction)
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    main()
