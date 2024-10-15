# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Build a Feature Engineering Pipeline
# MAGIC
# MAGIC In this demo, we will be constructing a feature engineering pipeline to manage data loading, imputation, and transformation. The pipeline will be applied to the training, testing, and validation sets, with the results showcased. The final step involves saving the pipeline to disk for future use, ensuring efficient and consistent data preparation for machine learning tasks.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Create a data preparation and feature engineering pipeline with multiple steps.
# MAGIC * Create a pipeline with tasks for data imputation and transformation.
# MAGIC * Apply a data preparation and pipeline set to a training/modeling set and a holdout set.
# MAGIC * Display the results of the transformation.
# MAGIC * Save a data preparation and feature engineering pipeline for potential future use.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-01

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains **variables such as your username, catalog name, schema name, working directory, and dataset locations**. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data Preparation
# MAGIC
# MAGIC Before building the pipeline, we will ensure consistency in the dataset by converting Integer and Boolean columns to Double data types and addressing missing values in both numeric and string columns within the **`Telco`** dataset. These are the steps we will follow in this section.
# MAGIC
# MAGIC 1. Load dataset
# MAGIC
# MAGIC 1. Split dataset to train and test sets
# MAGIC
# MAGIC 1. Converting Integer and Boolean Columns to Double
# MAGIC
# MAGIC 1. Handling Missing Values
# MAGIC
# MAGIC   * Numeric Columns
# MAGIC
# MAGIC   * String Columns
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Dataset

# COMMAND ----------

from pyspark.sql.functions import when, col

# Load dataset
dataset_path = f"{DA.paths.datasets}/telco/telco-customer-churn-missing.csv"
telco_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

# Select columns of interest
telco_df = telco_df.select("gender", "SeniorCitizen", "Partner", "tenure", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod", "TotalCharges", "Churn")

# COMMAND ----------

# MAGIC %md
# MAGIC Quick pre-processing
# MAGIC * `SeniorCitizen` as `boolean`
# MAGIC * `TotalCharges` as `double`

# COMMAND ----------

# replace "null" values with Null
# for column in telco_df.columns:
#   telco_df = telco_df.withColumn(column, when(col(column) == "null", None).otherwise(col(column)))

# clean-up columns
telco_df = telco_df.withColumn("SeniorCitizen", when(col("SeniorCitizen")==1, True).otherwise(False))
telco_df = telco_df.withColumn("TotalCharges", col("TotalCharges").cast("double"))

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split

# COMMAND ----------

train_df, test_df = telco_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transform Dataset

# COMMAND ----------

from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col, count, when


# Get a list of integer & boolean columns
integer_cols = [column.name for column in train_df.schema.fields if (column.dataType == IntegerType() or column.dataType == BooleanType())]

# Loop through integer columns to cast each one to double
for column in integer_cols:
    train_df = train_df.withColumn(column, col(column).cast("double"))
    test_df = test_df.withColumn(column, col(column).cast("double"))

string_cols = [c.name for c in train_df.schema.fields if c.dataType == StringType()]
num_cols = [c.name for c in train_df.schema.fields if c.dataType == DoubleType()]

# Get a list of columns with missing values
# Numerical
num_missing_values_logic = [count(when(col(column).isNull(),column)).alias(column) for column in num_cols]
row_dict_num = train_df.select(num_missing_values_logic).first().asDict()
num_missing_cols = [column for column in row_dict_num if row_dict_num[column] > 0]

# String
string_missing_values_logic = [count(when(col(column).isNull(),column)).alias(column) for column in string_cols]
row_dict_string = train_df.select(string_missing_values_logic).first().asDict()
string_missing_cols = [column for column in row_dict_string if row_dict_string[column] > 0]

print(f"Numeric columns with missing values: {num_missing_cols}")
print(f"String columns with missing values: {string_missing_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create a Pipeline
# MAGIC
# MAGIC Defines a Spark ML pipeline for preprocessing a dataset, including indexing categorical columns, imputing missing values, scaling numerical features, performing one-hot encoding on categorical features, and assembling the final feature vector for machine learning.
# MAGIC
# MAGIC In this Spark ML pipeline, we preprocess a dataset for predicting customer churn in a telecommunications **`telco`** company. The pipeline includes the following key steps:
# MAGIC
# MAGIC * **Convert Categorical Columns to Numerical Indices:**
# MAGIC This step converts categorical columns to numerical indices, allowing the model to process categorical data.
# MAGIC
# MAGIC * **Impute Missing Values:**
# MAGIC The Imputer is used to fill in missing values in **numerical columns with missing values (e.g. `tenure`, `TotalCharges`) using the `mean` strategy**, ensuring that the dataset is complete and ready for analysis. 
# MAGIC **Missing categorical values will be automatically encoded as a separate category.**
# MAGIC
# MAGIC * **VectorAssembler and RobustScaler:**
# MAGIC These steps combine relevant numerical columns into a feature vector and then scale the features to reduce sensitivity to outliers.
# MAGIC
# MAGIC * **Perform One Hot Encoding on Categorical variable:** 
# MAGIC This step converts the indexed categorical columns into binary sparse vectors, enabling the model to process categorical data effectively.
# MAGIC
# MAGIC * **Pipeline:**
# MAGIC  All these steps are encapsulated in a Pipeline, providing a convenient and reproducible way to preprocess the data for machine learning tasks.
# MAGIC

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, RobustScaler, StringIndexer, OneHotEncoder


# Imputer (mean strategy for all double/numeric)
to_impute = num_missing_cols
imputer = Imputer(inputCols=to_impute, outputCols=to_impute, strategy='mode')

# Scale numerical
numerical_assembler = VectorAssembler(inputCols=num_cols, outputCol="numerical_assembled")
numerical_scaler = RobustScaler(inputCol="numerical_assembled", outputCol="numerical_scaled")

# String/Cat Indexer (will encode missing/null as separate index)
string_cols_indexed = [c + '_index' for c in string_cols]
string_indexer = StringIndexer(inputCols=string_cols, outputCols=string_cols_indexed, handleInvalid="keep")

# OHE categoricals
ohe_cols = [column + '_ohe' for column in string_cols]
one_hot_encoder = OneHotEncoder(inputCols=string_cols_indexed, outputCols=ohe_cols, handleInvalid="keep")

# Assembler (All)
feature_cols = ["numerical_scaled"] + ohe_cols
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Instantiate the pipeline
stages_list = [
    imputer,
    numerical_assembler,
    numerical_scaler,
    string_indexer,
    one_hot_encoder,
    vector_assembler
]

pipeline = Pipeline(stages=stages_list)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Fit the Pipeline
# MAGIC
# MAGIC In the context of machine learning and MLflow, **`fitting`** corresponds to the process of training a machine learning model on a specified dataset. 
# MAGIC
# MAGIC In the previous step we created a pipeline. Now, we will fit a model based on the pipeline. This pipeline will index string columns, impute specified columns, scale numerical columns, one-hot-encode specified columns, and finally create a vector from all input columns.
# MAGIC

# COMMAND ----------

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Next, we can use this model to transform, or apply, to any dataset we want. 

# COMMAND ----------

# Transform both training_df and test_df
train_transformed_df = pipeline_model.transform(train_df)
test_transformed_df = pipeline_model.transform(test_df)

# COMMAND ----------

train_transformed_df.select("features").show(3, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save and Reuse the Pipeline
# MAGIC
# MAGIC Preserving the Telco Customer Churn Prediction pipeline, encompassing the model, parameters, and metadata, is vital for maintaining reproducibility, enabling version control, and facilitating collaboration among team members. This ensures a detailed record of the machine learning workflow. In this section, we will follow these steps;
# MAGIC
# MAGIC 1. **Save the Pipeline:** Save the pipeline model, including all relevant components, to the designated artifact storage. The saved pipeline is organized within the **`spark_pipelines`** folder for clarity.
# MAGIC
# MAGIC 1. **Explore Loaded Pipeline Stages:** Upon loading the pipeline, inspect the stages to reveal key transformations and understand the sequence of operations applied during the pipeline's execution.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Save the Pipeline

# COMMAND ----------

pipeline_model.save(f"{DA.paths.working_dir}/spark_pipelines")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Use Saved Model

# COMMAND ----------

from pyspark.ml import PipelineModel


# Load the pipeline
loaded_pipeline = PipelineModel.load(f"{DA.paths.working_dir}/spark_pipelines")

# Show pipeline stages
loaded_pipeline.stages

# COMMAND ----------

# Let's use loaded pipeline to transform the test dataset
test_transformed_df = loaded_pipeline.transform(test_df)
display(test_transformed_df)

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
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In summary, the featured engineering pipeline showcased in this demo offers a systematic and consistent approach to handle data loading, imputation, and transformation. By demonstrating its application on different sets and emphasizing the importance of data preparation, the pipeline proves to be a valuable tool for efficient and reproducible machine learning tasks. 
# MAGIC
# MAGIC The final step of saving the pipeline to disk ensures future usability, enhancing the overall effectiveness of the data preparation process.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>