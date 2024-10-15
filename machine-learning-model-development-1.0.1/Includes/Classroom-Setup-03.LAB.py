# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id, col
    
    # define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")

    # Read the dataset
    dataset_path = f"{DA.paths.datasets}/banking/loan-clean.csv"
    loan_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # Select columns of interest and replace spaces with underscores
    loan_df = loan_df.selectExpr("ID", "Age", "Experience", "Income", "`ZIP Code` as ZIP_Code", "Family", "CCAvg", "Education", "Mortgage", "`Personal Loan` as Personal_Loan", "`Securities Account` as Securities_Account", "`CD Account` as CD_Account", "Online", "CreditCard")

    # Save df as delta table using Delta API
    loan_df.write.format("delta").mode("overwrite").saveAsTable("bank_loan")




DBAcademyHelper.monkey_patch(create_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()                         
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

# COMMAND ----------

