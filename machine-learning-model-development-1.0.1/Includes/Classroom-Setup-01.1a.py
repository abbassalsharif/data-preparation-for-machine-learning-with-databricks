# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id
    from pyspark.sql.functions import col
    from databricks.feature_engineering import FeatureEngineeringClient


    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    
    dataset_path = f"{DA.paths.datasets}/ca-housing/ca-housing.csv"
    feature_data_pd = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
    
    # Add unique_id column
    feature_data_pd = feature_data_pd.withColumn("unique_id", monotonically_increasing_id())
    
    
    # Create the feature table using the PySpark DataFrame
    fe = FeatureEngineeringClient()
    table_name = f"{DA.catalog_name}.{DA.schema_name}.ca_housing"
    fe.create_table(
        name=table_name,
        primary_keys=["unique_id"],
        df=feature_data_pd,
        description="California Housing Feature Table",
        tags={"source": "bronze", "format": "delta"}
    )

DBAcademyHelper.monkey_patch(create_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()                         
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student