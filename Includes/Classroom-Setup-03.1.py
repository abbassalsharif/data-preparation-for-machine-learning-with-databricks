# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# Create a delta table for diet features
def create_security_features_table(self):
    
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    
    from pyspark.sql.functions import col

    dataset_path = f"{DA.paths.datasets}/telco/telco-customer-churn.csv"
    df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # drop the taget column
    df = df.select("customerID", "OnlineSecurity", "OnlineBackup", "DeviceProtection")

    table_name = f"{DA.catalog_name}.{DA.schema_name}.security_features"
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)
        
DBAcademyHelper.monkey_patch(create_security_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

DA.create_security_features_table()

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student
