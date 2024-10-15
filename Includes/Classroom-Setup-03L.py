# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# Create a delta table for diet features
def create_diet_features_table(self):
    
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    
    from pyspark.sql.functions import col, monotonically_increasing_id

    dataset_path = f"{DA.paths.datasets}/cdc-diabetes/diabetes_binary_5050split_BRFSS2015.csv"
    df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
    df = df.select("Fruits", "Veggies", "HvyAlcoholConsump", "Smoker")
    df = df.withColumn("UID", monotonically_increasing_id())

    table_name = f"{DA.catalog_name}.{DA.schema_name}.diet_features"
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)
        
DBAcademyHelper.monkey_patch(create_diet_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

DA.create_diet_features_table()

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student
