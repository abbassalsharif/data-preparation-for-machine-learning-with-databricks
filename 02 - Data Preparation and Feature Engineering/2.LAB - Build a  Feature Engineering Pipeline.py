# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB - Build a  Feature Engineering Pipeline
# MAGIC
# MAGIC Welcome to the "Build a Feature Engineering Pipeline" lab! In this hands-on session, we'll dive into the essential steps of creating a robust feature engineering pipeline. From data loading and preparation to fitting a pipeline and saving it for future use, this lab equips you with fundamental skills in crafting efficient and reproducible machine learning workflows. Let's embark on the journey of transforming raw data into meaningful features for predictive modeling.
# MAGIC
# MAGIC **Lab Outline**
# MAGIC
# MAGIC + **Task 1:** Load Dataset and Data Preparation
# MAGIC   + **1.1.** Load Dataset
# MAGIC   + **1.2.** Data Preparation
# MAGIC + **Task 2:** Split Dataset
# MAGIC + **Task 3:** Create Pipeline for Data Imputation and Transformation
# MAGIC + **Task 4:** Fit the Pipeline
# MAGIC + **Task 5:** Show Transformation Results
# MAGIC + **Task 6:** Save Pipeline
# MAGIC
# MAGIC
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
# MAGIC Before starting the Lab, run the provided classroom setup script. This script will establish necessary configuration variables tailored to each user. Execute the following code cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-01

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Load Dataset and Data Preparation
# MAGIC
# MAGIC
# MAGIC **1.1. Load Dataset:**
# MAGIC + Load a dataset with features that require imputation and transformation
# MAGIC + Display basic information about the dataset (e.g., schema, summary statistics)
# MAGIC
# MAGIC **1.2. Data Preparation:**
# MAGIC
# MAGIC + Examine the dataset.
# MAGIC + Identify and discuss the features that need data preparation.
# MAGIC + Convert data types: Demonstrate converting data types for selected columns (e.g., String to Int, Int to Boolean).
# MAGIC + Remove a column: Discuss and remove a column with too many missing values.
# MAGIC + Remove outliers: Implement a threshold-based approach to remove outlier records for a specific column.
# MAGIC + Save cleaned dataset as "silver table."

# COMMAND ----------

# MAGIC %md
# MAGIC **1.1. Load Dataset:**
# MAGIC
# MAGIC + Load a dataset with features that require imputation and transformation
# MAGIC + Display basic information about the dataset (e.g., schema, summary statistics)
# MAGIC

# COMMAND ----------

# TODO
Set the path of the dataset
dataset_path = f"{DA.paths.datasets}/cdc-diabetes/diabetes_binary_5050_raw.csv"

Read the CSV file using the Spark read.csv function
Set the header parameter to True to indicate that the CSV file has a header
Set the inferSchema option to True for Spark to automatically detect the data types
Set the multiLine option to True to ensure that Spark reads multi-line fields properly
cdc_df = <FILL_IN>

Display the resulting dataframe
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC **1.2. Data Preparation:**
# MAGIC
# MAGIC + Examine the dataset.
# MAGIC + Identify the features that need data preparation.
# MAGIC + Convert data types: Demonstrate converting data types for selected columns (e.g., String to Int, Double to Boolean).

# COMMAND ----------

# TODO
Convert string columns to integer type
from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col

List of string columns to convert: (HighBP, CholCheck, PhysActivity)
string_columns = <FILL_IN>

Iterate over string columns and cast to integer type
for column in string_columns:
   cdc_df = <FILL_IN>

Convert double columns to BooleanType (Diabetes_binary, GenHlth)
double_columns = <FILL_IN>
for column in double_columns:
    cdc_df = <FILL_IN>

Print the schema
cdc_df.<FILL_IN>
Examine the printed schema to verify the changes.

# COMMAND ----------

# MAGIC %md
# MAGIC + **Remove a column with too many missing values.**

# COMMAND ----------

# TODO 
from pyspark.sql.functions import col, when, count, concat_ws, collect_list

First, get the count of missing values per column to create a singleton row DataFrame
missing_cdc_df = cdc_df.select([count(when(col(c).contains('null') | (col(c) == '') | col(c).isNull(), c)).alias(c) for c in cdc_df.columns])

Define a helper function to transpose the DataFrame for better readability
def TransposeDF(df, columns, pivotCol):
    """Helper function to transpose Spark DataFrame"""
    columnsValue = <FILL_IN>
    stackCols = ','.join(x for x in columnsValue)
    df_1 = df.selectExpr(pivotCol, "stack(" + str(len(columns)) + "," + stackCols + ")")\
              .select(pivotCol, "col0", "col1")
    final_df = <FILL_IN>
    return final_df

Transpose the missing_cdc_df for better readability
missing_df_T = TransposeDF(spark.createDataFrame(<FILL_IN>)

Display the count of missing values per column
display(missing_cdc_df)

Set a threshold for missing data to drop columns
per_thresh = 0.6

Calculate the total count of rows in the DataFrame
N = <FILL_IN>

Identify columns with more than the specified percentage of missing data
to_drop_missing = <FILL_IN>

Drop columns with more than 60% missing data
print(f"Dropping columns {to_drop_missing} with more than {per_thresh * 100}% missing data")
cdc_no_missing_df = cdc_df.drop(*to_drop_missing)

Display the DataFrame after dropping columns with excessive missing data
<FILL_IN>


# COMMAND ----------

# MAGIC %md
# MAGIC + **Remove outliers: Implement a threshold-based approach to remove outlier records for a specific column.**

# COMMAND ----------

# TODO
Remove listings with MentHlth > -40
MentHlth_cutoff = -40
cdc_no_outliers_df = <FILL_IN> 

Remove listings with BMI > 110
BMI_cutoff = 110
cdc_no_outliers_df = <FILL_IN>

Display the count before and after removing outliers
print(<FILL_IN>)

Display the DataFrame after removing outliers
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC + **Save the cleaned dataset as the "silver table" for further analysis**

# COMMAND ----------

# TODO
cdc_df_full = "cdc_df_full"

Save as DELTA table (silver)
cdc_df_full_silver = <FILL_IN>
cdc_no_outliers_df.write.mode("overwrite").option("mergeSchema", True).saveAsTable(cdc_df_full_silver)

Print the resulting DataFrame
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Split Dataset
# MAGIC
# MAGIC **2.1. Split Dataset:**
# MAGIC
# MAGIC + Split the cleaned dataset into training and testing sets in 80:20 ratio

# COMMAND ----------

# TODO
Split with 80 percent of the data in train_df and 20 percent of the data in test_df
train_df, test_df = cdc_no_outliers_df.<FILL_IN>

Materialize the split DataFrames as DELTA tables
train_df.write.mode("overwrite").<FILL_IN>
test_df.write.mode("overwrite").<FILL_IN>

# COMMAND ----------

# TODO
from pyspark.ml.feature import StandardScaler, RobustScaler, VectorAssembler

Assuming 'train_data' is your training set DataFrame
feature_columns = ["income"]  # Add your actual feature column names

Assemble features into a vector
assembler = VectorAssembler(<FILL_IN>)
train_assembled_df = assembler.transform(<FILL_IN>)
test_assembled_df = assembler.transform(<FILL_IN>)

Define scaler and fit on the training set
scaler = RobustScaler(inputCol=<FILL_IN>, outputCol=<FILL_IN>)
scaler_fitted = scaler.<FILL_IN>

Apply to both training and test sets
train_scaled_df = scaler_fitted.transform(<FILL_IN>)
test_scaled_df = scaler_fitted.transform(<FILL_IN>)

Show the resulting DataFrames
print("This is the Training set:")
<FILL_IN>
print("This is the Testing set:")
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Create Pipeline using Data Imputation and Transformation
# MAGIC
# MAGIC **3.1. Create Pipeline:**
# MAGIC
# MAGIC + Create a pipeline with the following tasks:
# MAGIC   + StringIndexer
# MAGIC   + Imputer
# MAGIC   + Scaler
# MAGIC   + One-Hot Encoder
# MAGIC

# COMMAND ----------

# TODO
from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col, count, when

Get a list of integer & boolean columns
integer_cols = [column.name for column in train_df.schema.fields if (column.dataType == IntegerType() or column.dataType == BooleanType())]

Loop through integer columns to cast each one to double
Convert integer columns to double
for column in integer_cols:
    train_df = <FILL_IN>
    test_df = <FILL_IN>

Get a list of string, numeric columns
string_cols = <FILL_IN>
num_cols = <FILL_IN>

Get a list of columns with missing values
Numerical
num_missing_values_logic = <FILL_IN>
row_dict_num = train_df.<FILL_IN>
num_missing_cols = <FILL_IN>

String
string_missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in string_cols]
row_dict_string = train_df.<FILL_IN>
string_missing_cols = <FILL_IN>

Print columns with missing values
print(f"Numeric columns with missing values: <FILL_IN>")
print(f"String columns with missing values: <FILL_IN>")

# COMMAND ----------

# TODO
import required libraries
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, RobustScaler, StringIndexer, OneHotEncoder

String/Cat Indexer
create an additional column to index string columns
these columns will retain their original null values via 'handleInvalid="keep"'
string_cols_indexed = [c + '_index' for c in string_cols]
string_indexer = StringIndexer(<FILL_IN>)

Imputer (same strategy for all double/indexes)
create a list of columns containing missing values
utilize the mode strategy to impute all the missing columns
string_missing_cols_indexed = <FILL_IN>
to_impute = <FILL_IN>

imputer = Imputer(inputCols=to_impute, outputCols=to_impute, strategy='mode')

Scale numerical
create a vector of numerical columns as an array in the 'numerical_assembled' column
robustly scale all the numerical_scaled values for this array in the 'numerical_scaled' column
numerical_assembler = VectorAssembler(<FILL_IN>)
numerical_scaler = RobustScaler(<FILL_IN>)

OHE categoricals
create an OHE encoder to turn the indexed string columns into binary vectors
ohe_cols = [column + '_ohe' for column in string_cols]
one_hot_encoder = OneHotEncoder(<FILL_IN>)

Assembler (All)
re-collect all columns and create a 'features' column from them
feature_cols = ["numerical_scaled"] + <FILL_IN>
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

Instantiate the pipeline
instantiate a pipeline with all the above stages
stages_list = [
    string_indexer,
    imputer,
    numerical_assembler,
    numerical_scaler,
    one_hot_encoder,
    vector_assembler
]

pipeline = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4: Fit the Pipeline
# MAGIC **4.1. Fit the Pipeline:**
# MAGIC
# MAGIC + Use the training dataset to fit the created pipeline.

# COMMAND ----------

# TODO
Fit the pipeline using the training dataset
pipeline_model = <FILL_IN>


# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 5: Show Transformation Results
# MAGIC **5.1. Transform Datasets:**
# MAGIC
# MAGIC + Apply the fitted pipeline to transform the training and testing datasets.
# MAGIC + Apply these transformations to different sets (e.g., train, test, validation).

# COMMAND ----------

# TODO
Transform both the training and test datasets using the previously fitted pipeline model
train_transformed_df = pipeline_model.<FILL_IN>
test_transformed_df = <FILL_IN>

Display the transformed features from the training dataset
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 6: Save Pipeline
# MAGIC **6.1. Save Pipeline:**
# MAGIC
# MAGIC + Save the fitted pipeline to the working directory.
# MAGIC + Explore the saved pipeline.

# COMMAND ----------

# TODO 
Save the trained pipeline model to the specified directory in the working directory
pipeline_model.<FILL_IN>

# COMMAND ----------

# TODO 
Load the previously saved pipeline model from the specified directory in the working directory
from pyspark.ml import PipelineModel

Load the pipeline model
loaded_pipeline = <FILL_IN>

Display the stages of the loaded pipeline
<FILL_IN>

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
# MAGIC ## Conclusion
# MAGIC
# MAGIC In conclusion, this lab demonstrated the crucial steps in preparing and transforming a dataset for machine learning. We covered data cleaning, splitting, and created a pipeline for tasks like imputation and scaling. Saving the pipeline ensures reproducibility, and these foundational concepts can be applied in various machine learning workflows.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>