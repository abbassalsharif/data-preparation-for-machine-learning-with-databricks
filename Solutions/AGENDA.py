# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Data Preparation for Machine Learning
# MAGIC
# MAGIC This course focuses on the fundamentals of preparing data for machine learning using Databricks. Participants will learn essential skills for exploring, cleaning, and organizing data tailored for traditional machine learning applications. Key topics include data visualization, feature engineering, and optimal feature storage strategies. Through practical exercises, participants will gain hands-on experience in efficiently preparing data sets for machine learning within the Databricks. This course is designed for associate-level data scientists and machine learning practitioners. and individuals seeking to enhance their proficiency in data preparation, ensuring a solid foundation for successful machine learning model deployment.
# MAGIC
# MAGIC
# MAGIC ## Course agenda
# MAGIC
# MAGIC | Time | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 60m    | **[Managing and Exploring Data]($./01 - Managing and Exploring Data/1.1 - Load and Explore Data)**    | Load and Explore Data<br>[Demo: Load and Explore Data]($./01 - Managing and Exploring Data/1.1 - Load and Explore Data)<br>[Lab: Load and Explore Data]($./01 - Managing and Exploring Data/1.LAB - Load and Explore Data)|
# MAGIC | 110m    | **[Data Preparation and Feature Engineering]($./02 - Data Preparation and Feature Engineering/2.1 - Data Imputation and Transformation Pipeline)** | Data Preparation and Feature Engineering </br> [Demo: Data Imputation and Transformation Pipeline]($./02 - Data Preparation and Feature Engineering/2.1 - Data Imputation and Transformation Pipeline) </br> [Demo: Build a Feature Engineering Pipeline]($./02 - Data Preparation and Feature Engineering/2.2 - Build a Feature Engineering Pipeline) </br> [Lab: Build a  Feature Engineering Pipeline]($./02 - Data Preparation and Feature Engineering/2.LAB - Build a  Feature Engineering Pipeline) | 
# MAGIC | 70m    | **[Feature Store]($./03 - Feature Store/3.1 - Using Feature Store for Feature Engineering)** | Feature Store </br> [Demo: Using Feature Store for Feature Engineering]($./03 - Feature Store/3.1 - Using Feature Store for Feature Engineering) </br> [Lab: Feature Engineering with Feature Store]($./03 - Feature Store/3.LAB - Feature Engineering with Feature Store) |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12, 13.3.x-gpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>