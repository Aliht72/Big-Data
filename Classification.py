# Import libraries
def import_libraries():
    global os, findspark, pyspark, np, pd, SparkSession, F, when, col, VectorAssembler, DecisionTreeClassifier
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

# Set environment variables
def set_environment_variables():
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop2.7"

# Initialize Spark session
def initialize_spark():
    findspark.init()
    findspark.find()
    spark = SparkSession.builder.master('local[*]').appName('flamingo').getOrCreate()
    return spark

# Load and clean data
def load_and_clean_data(spark, file_path):
    df = spark.read.csv(file_path, sep=',', header=True, inferSchema=True, nullValue='NA')
    return df.na.drop()

# Create dummy variables
def create_dummy_variables(df):
    categories = df.select("platformType").distinct().rdd.flatMap(lambda x: x).collect()
    exprs = [F.when(F.col("platformType") == category, 1).otherwise(0).alias(category) for category in categories]
    return df.select("count_buyId", "avg_price", "count_hits", "count_gameclicks", "teamLevel", *exprs)

# Add role and label columns
def add_role_and_label_columns(df):
    df = df.withColumn('Role', when(col('avg_price') > 5, 'Main Role').otherwise('Others'))
    return df.withColumn('label', (df.Role == "Others").cast('integer')).drop('avg_price')

# Assemble features
def assemble_features(df):
    assembler = VectorAssembler(inputCols=[
        'count_buyId', 'count_hits', 'count_gameclicks', 'teamLevel',
        'iphone', 'android', 'linux', 'mac', 'windows'
    ], outputCol='features')
    return assembler.transform(df)

# Split data into training and test sets
def split_data(df, seed=17):
    return df.randomSplit([0.8, 0.2], seed=seed)

# Train decision tree classifier
def train_decision_tree(train_df):
    tree = DecisionTreeClassifier()
    return tree.fit(train_df)

# Evaluate model
def evaluate_model(model, test_df):
    prediction = model.transform(test_df)
    prediction.select('label', 'prediction', 'probability').show(5, False)
    return prediction

# Calculate accuracy
def calculate_metrics(prediction):
    TN = prediction.filter('prediction = 0 AND label = prediction').count()
    TP = prediction.filter('prediction = 1 AND label = prediction').count()
    FN = prediction.filter('prediction = 0 AND label = 1').count()
    FP = prediction.filter('prediction = 1 AND label = 0').count()
    return (TN + TP) / (TN + TP + FN + FP)

# Main function
def main():
    install_dependencies()
    import_libraries()
    set_environment_variables()
    spark = initialize_spark()
    cleaned_df = load_and_clean_data(spark, './combined-data.csv')
    dummy_df = create_dummy_variables(cleaned_df)
    labeled_df = add_role_and_label_columns(dummy_df)
    assembled_df = assemble_features(labeled_df)
    train_df, test_df = split_data(assembled_df)
    tree_model = train_decision_tree(train_df)
    prediction = evaluate_model(tree_model, test_df)
    accuracy = calculate_metrics(prediction)
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    main()
