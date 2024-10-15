# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

def create_features_table(self):
    from databricks.feature_engineering import FeatureEngineeringClient
    import pandas as pd
    from pyspark.sql.functions import monotonically_increasing_id, col
    
    fe = FeatureEngineeringClient()
    
    # define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")

    # Read the dataset
    dataset_path = f"{DA.paths.datasets}/cdc-diabetes/diabetes_binary_5050split_BRFSS2015.csv"
    df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
    df = df.withColumn("unique_id", monotonically_increasing_id())   # Add unique_id column

    # create the feature table using the PySpark DataFrame
    table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes"
    fe.create_table(
        name=table_name,
        primary_keys=["unique_id"],
        df=df,
        description="Diabetes Feature Table",
        tags={"source": "silver", "format": "delta"}
    )

DBAcademyHelper.monkey_patch(create_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()                         
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student