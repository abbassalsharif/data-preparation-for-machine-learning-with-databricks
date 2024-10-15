# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id, col
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline
    
    # define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")

    # Read the dataset
    dataset_path = f"{DA.paths.datasets}/telco/telco-customer-churn.csv"
    telco_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # Select columns of interest
    telco_df = telco_df.select("gender", "SeniorCitizen", "Partner", "tenure", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod", "TotalCharges", "Churn")

    # basic clean-up
    telco_df = telco_df.withColumn("Gender", col("gender"))
    telco_df = telco_df.withColumn("Tenure", col("tenure").cast("double"))
    telco_df = telco_df.withColumn("TotalCharges", col("TotalCharges").cast("double"))

    # Fill missing values with 0
    telco_df = telco_df.fillna(0)

    # Identify categorical columns
    categorical_cols = [col_name for col_name, data_type in telco_df.dtypes if data_type == 'string']

    # Create a StringIndexer for each categorical column
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") for col in categorical_cols]

    # Create a pipeline to apply all indexers
    pipeline = Pipeline(stages=indexers)

    # Fit the pipeline to the data
    telco_df = pipeline.fit(telco_df).transform(telco_df)

    # Drop the original categorical columns
    telco_df = telco_df.drop(*categorical_cols)

    # Rename the indexed columns to match the original column names
    for col in categorical_cols:
        telco_df = telco_df.withColumnRenamed(f"{col}_index", col)


    # save df as delta table
    telco_df.write.format("delta").option("overwriteSchema", 'true').mode("overwrite").saveAsTable("customer_churn")

DBAcademyHelper.monkey_patch(create_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()                         
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student