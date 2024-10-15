# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id, col
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline
    
    from databricks.feature_engineering import FeatureEngineeringClient    

    import mlflow

    # disable autologging 
    mlflow.autolog(disable=True)

    # Read the CSV file into a Spark DataFrame
    dataset_path = f"{DA.paths.datasets}/telco/telco-customer-churn.csv"
    feature_data_spark = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # Fill missing values with 0
    feature_data_spark = feature_data_spark.fillna(0)

    # Identify categorical columns
    categorical_cols = [col_name for col_name, data_type in feature_data_spark.dtypes if data_type == 'string']

    # Create a StringIndexer for each categorical column
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") for col in categorical_cols]

    # Create a pipeline to apply all indexers
    pipeline = Pipeline(stages=indexers)

    # Fit the pipeline to the data
    feature_data_spark = pipeline.fit(feature_data_spark).transform(feature_data_spark)

    # Drop the original categorical columns
    feature_data_spark = feature_data_spark.drop(*categorical_cols)

    # Rename the indexed columns to match the original column names
    for col in categorical_cols:
        feature_data_spark = feature_data_spark.withColumnRenamed(f"{col}_index", col)
    
     # Create the feature table using the PySpark DataFrame
    fe = FeatureEngineeringClient()
    table_name = f"{DA.catalog_name}.{DA.schema_name}.telco"
    fe.create_table(
        name=table_name,
        primary_keys=["customerID"],
        df=feature_data_spark,
        description="Telco Dataset",
        tags={"source": "silver", "format": "delta"}
    )

DBAcademyHelper.monkey_patch(create_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()                         
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student