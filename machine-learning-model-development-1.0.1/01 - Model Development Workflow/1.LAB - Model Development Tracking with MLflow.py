# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB - Model Development Tracking with *MLflow* 
# MAGIC
# MAGIC In this lab, you will learn how to leverage MLflow to track and manage the model development process. First, you will load data from a feature table and create train and test splits. Then, you train a classification model and track the training process with MLflow. While MLflow automatically logs all metadata and artifacts with autologging, you will do it manually to learn how to use logging API.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC * **Task 1.** Load dataset from feature store table
# MAGIC * **Task 2.** Define model hyperparameters
# MAGIC * **Task 3.** Track the model with MLflow 
# MAGIC * **Task 4.** Log custom figure
# MAGIC * **Task 5.** Review the model details via the UI in the Experiment runs.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12 13.3.x-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup scripts. 
# MAGIC
# MAGIC **📌 Note:** In this lab you will register MLflow models with Unity Catalog. Therefore, you will need to run the next code block to **set model registery URI to UC**. 
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade 'mlflow-skinny[databricks]'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Next, this script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1.LAB

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
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tasks 1 - Load Dataset from Feature Store Table
# MAGIC
# MAGIC Use the feature store to load a dataset from a specific table.
# MAGIC    - **Load Dataset:** Utilize MLflow's `load_delta` function to seamlessly retrieve and load the dataset from the Feature Store table named **`"telco"`** in the specified catalog and schema `("DA.catalog_name" and "DA.schema_name")`. 
# MAGIC    - Convert dataset to `pandas` dataframe and explore the loaded dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

# Import the necessary library for MLflow
import mlflow

# Load the feature dataset using mlflow.data
feature_dataset = mlflow.<FILL_IN>(
    table_name=<FILL_IN>,
    name=<FILL_IN>
)

# convert the dataset to pandas df and drop the customerID column
feature_data_pd = <FILL_IN>

# Convert all feature_data_pd columns to float
feature_data_pd = feature_data_pd.astype(float)

# inspect final dataset
display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split
# MAGIC
# MAGIC Split the dataset into training and testing sets.

# COMMAND ----------

# Import necessary libraries
import mlflow.sklearn  # For MLflow integration
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets

# Split the dataset into training and testing sets
target_col = <FILL_IN>
X_all = <FILL_IN>
y_all = <FILL_IN>
X_train, X_test, y_train, y_test = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Define Model Hyperparameters
# MAGIC
# MAGIC In this lab, you will train a classification model. In this task define parameters for a Decision Tree Model.

# COMMAND ----------

# Define Decision Tree Classifier parameters
dtc_params = {
  'criterion': <FILL_IN>,
  'max_depth': <FILL_IN>,
  'min_samples_split': <FILL_IN>,
  'min_samples_leaf': <FILL_IN>
}

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3 - Track the Model Development  with MLflow
# MAGIC
# MAGIC Initialize an MLflow run.
# MAGIC    - **Initialize MLflow Run:** Start an MLflow run to track the model development process. This allows for systematic recording of parameters, metrics, and artifacts associated with the model.
# MAGIC
# MAGIC    - **Logging Model Details:** Utilize MLflow tracking to log essential information about the model, including parameters, metrics, and other relevant artifacts.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# TODO  

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from mlflow.models.signature import infer_signature

# set the path for mlflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/LAB-1-Model-Development-Tracking-with-MLflow")

# Turn off autologging as we want to log the model manually
mlflow.autolog(disable=True)

# Start an MLFlow run
with mlflow.start_run(run_name="Model Developing Tracking with MLflow Lab") as run:
   # Log the dataset
   mlflow.log_input(feature_dataset, context="source")
   mlflow.log_input(mlflow.data.from_pandas(<FILL_IN>, source=feature_dataset.source), context="training")
   mlflow.log_input(mlflow.data.from_pandas(<FILL_IN> source=feature_dataset.source), context="test")

   # Log parameters
   mlflow.log_params(<FILL_IN>)

   # Fit the model
   dtc = DecisionTreeClassifier(<FILL_IN>)
   dtc_mdl = dtc.fit(<FILL_IN>)

   # Define model signature
   signature = infer_signature(X_all, y_all)
    
   # Log the model
   # Define the model name based on the feature store catalog and schema
   model_name = f"{DA.catalog_name}.{DA.schema_name}.churnmodel"
   mlflow.sklearn.log_model(
       sk_model=<FILL_IN>,
       artifact_path="model-artifacts",
       signature=<FILL_IN>,
       registered_model_name=<FILL_IN>
   )

   # Evaluate on the training set
   y_pred_train = <FILL_IN>
   mlflow.log_metric("train_accuracy", <FILL_IN>)
   mlflow.log_metric("train_precision", <FILL_IN>)
   mlflow.log_metric("train_recall", <FILL_IN>)
   mlflow.log_metric("train_f1", <FILL_IN>)

   # Evaluate on the test set
   y_pred_test = dtc_mdl.predict(X_test)
   mlflow.log_metric("test_accuracy", <FILL_IN>)
   mlflow.log_metric("test_precision", <FILL_IN>)
   mlflow.log_metric("test_recall", <FILL_IN>)
   mlflow.log_metric("test_f1", <FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Log Custom Figure
# MAGIC
# MAGIC **Log Custom Figure/Visualization:** Include the logging of a custom figure, such as a confusion matrix or any relevant visualization, to further illustrate the model's behavior. This visual representation can be valuable for model evaluation and interpretation.

# COMMAND ----------

# TODO    
# Import necessary libraries for creating and displaying a confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mlflow.client import MlflowClient
client = MlflowClient()

# Compute the confusion matrix
cm = confusion_matrix(<FILL_IN>)

# Create a figure object and axes for the confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))

# Create a ConfusionMatrixDisplay object with the computed confusion matrix
disp = ConfusionMatrixDisplay(<FILL_IN>)

# Plot the confusion matrix using the created axes and specified color map
disp.plot(<FILL_IN>)

# Set the title of the plot
ax.set_title('Confusion Matrix')

# Log the confusion matrix figure to MLflow
client.log_figure(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Review model details via the UI
# MAGIC To review the model details via the MLflow UI in the Experiment runs, follow these steps:
# MAGIC
# MAGIC + Step 1: Go to the "Experiments" Section
# MAGIC
# MAGIC + Step 2: Locate Your Experiment
# MAGIC
# MAGIC + Step 3: Review Run Details
# MAGIC
# MAGIC + Step 4: Reviewing Artifacts and Metrics
# MAGIC
# MAGIC + Step 5: Viewing Confusion Matrix Image
# MAGIC
# MAGIC + Step 6: Retrieve Model Details

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
# MAGIC
# MAGIC In conclusion, this lab showcased the effectiveness of MLflow in seamlessly managing the model development process. Leveraging MLflow's features, such as experiment tracking, custom metric logging, and artifact storage, enhances collaboration and ensures reproducibility. The ability to review model details through the MLflow UI provides valuable insights into model performance and aids in making informed decisions.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>