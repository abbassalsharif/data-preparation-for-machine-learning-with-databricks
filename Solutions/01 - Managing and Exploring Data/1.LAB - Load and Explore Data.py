# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB - Load and Explore Data
# MAGIC
# MAGIC
# MAGIC Welcome to the "Load and Explore Data" lab! In this session, you will learn essential skills in data loading and exploration using PySpark in a Databricks environment. Gain hands-on experience reading data from Delta tables, managing data permissions, computing summary statistics, and using data profiling tools to unveil insights in your Telco dataset. Let's dive into the world of data exploration!
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC
# MAGIC In this Lab, you will learn how to:
# MAGIC 1. Read data from delta table
# MAGIC 1. Manage data permissions
# MAGIC 1. Show summary statistics
# MAGIC 1. Use data profiler to explore data frame
# MAGIC     - Check outliers
# MAGIC     - Check data distributions
# MAGIC 1. Read previous versions of the delta table
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Lab Setup
# MAGIC
# MAGIC Before starting the Lab, follow these initial steps:
# MAGIC
# MAGIC 1. Run the provided classroom setup script. This script will establish necessary configuration variables tailored to each user. Execute the following code cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-01

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll make use of the object `DA`, which provides critical variables. Execute the code block below to see various variables that will be used in this notebook:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ##Task 1: Read Data from Delta Table
# MAGIC
# MAGIC
# MAGIC + Use Spark to read data from the Delta table into a DataFrame.
# MAGIC
# MAGIC

# COMMAND ----------

# ANSWER
dataset_path = f"{DA.paths.datasets}/telco/telco-customer-churn-missing.csv"

# Read dataset with spark
telco_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

table_name = "telco_missing"
table_name_bronze = f"{table_name}_bronze"

# Write it as delta table
telco_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name_bronze)
telco_df.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##Task 2: Manage Data Permissions
# MAGIC
# MAGIC Establish controlled access to the Telco Delta table by granting specific permissions for essential actions.
# MAGIC
# MAGIC + Grant permissions for specific actions (e.g., read, write) on the Delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
# MAGIC -- Write query to Grant Permission to all the users to access Delta Table
# MAGIC GRANT SELECT ON TABLE telco_missing_bronze TO `account users`;

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3: Show Summary Statistics
# MAGIC
# MAGIC
# MAGIC Compute and present key statistical metrics to gain a comprehensive understanding of the Telco dataset.
# MAGIC
# MAGIC
# MAGIC + Utilize PySpark to compute and display summary statistics for the Telco dataset.
# MAGIC
# MAGIC + Include key metrics such as mean, standard deviation, min, max, etc.

# COMMAND ----------

# ANSWER
# Show summary of the Data
dbutils.data.summarize(telco_df) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4: Use Data Profiler to Explore DataFrame
# MAGIC Use the Data Profiler and Visualization Editor tools.
# MAGIC
# MAGIC + Use the Data Profiler to explore the structure, data types, and basic statistics of the DataFrame.
# MAGIC     - **Task 4.1.1:** Identify columns with missing values and analyze the percentage of missing data for each column.
# MAGIC     - **Task 4.1.2:** Review the data types of each column to ensure they match expectations. Identify any columns that might need type conversion.
# MAGIC + Use Visualization Editor to Check Outliers and Data Distributions:
# MAGIC     - **Task 4.2.1:** Create a bar chart to visualize the distribution of churned and non-churned customers.
# MAGIC     - **Task 4.2.2:** Generate a pie chart to visualize the distribution of different contract types.
# MAGIC     - **Task 4.2.3:** Create a scatter plot to explore the relationship between monthly charges and total charges.
# MAGIC     - **Task 4.2.4:** Visualize the count of customers for each payment method using a bar chart.
# MAGIC     - **Task 4.2.5:** Compare monthly charges for different contract types using a box plot.
# MAGIC             

# COMMAND ----------

# Display the data and Explore the Data Profiler and Visualization Editor
#<FILL_IN>

# COMMAND ----------

# Display the data and Explore the Data Profiler and Visualization Editor
display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 5: Drop the Column
# MAGIC Remove a specific column, enhancing data cleanliness and focus.
# MAGIC
# MAGIC
# MAGIC + Identify the column that needs to be dropped. For example, let's say we want to drop the 'SeniorCitizen' column.
# MAGIC
# MAGIC
# MAGIC + Use the appropriate command or method to drop the identified column from the Telco dataset.
# MAGIC
# MAGIC
# MAGIC + Verify that the column has been successfully dropped by displaying the updated dataset.

# COMMAND ----------

# ANSWER
# Drop SeniorCitizen Column 
telco_dropped_df = telco_df.drop("SeniorCitizen")

# Overwrite the Delta table
telco_dropped_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 6: Time-Travel to First 
# MAGIC
# MAGIC
# MAGIC Revert the Telco dataset back to its initial state, exploring the characteristics of the first version.
# MAGIC
# MAGIC
# MAGIC + Utilize time-travel capabilities to revert the dataset to its initial version.
# MAGIC
# MAGIC
# MAGIC + Display and analyze the first version of the Telco dataset to understand its original structure and content.
# MAGIC

# COMMAND ----------

# ANSWER
# Extract timestamp of first version (can also be set manually)
timestamp_v0 = spark.sql(f"DESCRIBE HISTORY telco_missing_bronze").orderBy("version").first().timestamp
(spark
        .read
        .option("timestampAsOf", timestamp_v0)
        .table("telco_missing_bronze")
        .printSchema()
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Task 7: Read previous versions of the delta table
# MAGIC Demonstrate the ability to read data from a specific version of the Delta table.
# MAGIC
# MAGIC + Replace the timestamp in the code with the actual version or timestamp of interest.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
# MAGIC -- Show table versions
# MAGIC DESCRIBE HISTORY telco_missing_bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Conclusion
# MAGIC In this lab, you demonstrated how to explore and manipulate the dataset using Databricks, focusing on data exploration, management, and time-travel capabilities. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>