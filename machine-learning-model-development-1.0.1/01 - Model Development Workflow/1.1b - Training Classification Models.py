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
# MAGIC # Training Regression and Classification Models
# MAGIC
# MAGIC In this demo, we will explore the process of training a classification model using the sklearn API. In addition to fitting the model, we will inspect the model details and show how the decision tree is constructed.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Fit a classification model on modeling data using the sklearn API.
# MAGIC
# MAGIC * Interpret a fit sklearn linear model’s coefficients and intercept.
# MAGIC
# MAGIC * Fit a decision tree model using sklearn API and training data.
# MAGIC
# MAGIC + Visualize a sklearn tree’s split points.
# MAGIC
# MAGIC * Identify which metrics are tracked by MLflow.
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
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-01.1b

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
# MAGIC ## Prepare Dataset
# MAGIC Before training any machine learning models, it's crucial to prepare the dataset. In the previous section, we have covered the steps to load, clean, and preprocess the data, ensuring it's in a suitable format for model training.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Dataset
# MAGIC
# MAGIC In this section, we aim to optimize the process of loading the dataset by leveraging Delta Lake's feature table functionality. Instead of directly reading from the CSV file, **we created a feature table during the setup phase**. A feature table is a structured repository that organizes data for efficient retrieval and analysis. By creating a feature table, we enhance data management and simplify subsequent operations. We can then seamlessly read the data from this feature table during our analysis, promoting a more organized and scalable approach to handling datasets. This setup enhances traceability and facilitates collaboration across different stages of the data science workflow.

# COMMAND ----------

import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient

# Read dataset from the feature store table
fe = FeatureEngineeringClient()
table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes_binary"
feature_data_pd = fe.read_table(name=table_name).toPandas()
feature_data_pd = feature_data_pd.drop(columns=['unique_id'])

# Convert all columns in the DataFrame to the 'double' data type
for column in feature_data_pd.columns:
    feature_data_pd[column] = feature_data_pd[column].astype("double")

# Display the Pandas DataFrame with updated data types
display(feature_data_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split
# MAGIC
# MAGIC The "train_test_split" function from the scikit-learn library is commonly used to split a dataset into training and testing sets. This is a crucial step in machine learning to **evaluate how well a trained model generalizes to new, unseen data**.

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {feature_data_pd.shape[0]} records in our source dataset")

# split target variable into it's own dataset
target_col = "Diabetes_binary" 
X_all = feature_data_pd.drop(labels=target_col, axis=1)
y_all = feature_data_pd[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.95, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit a Classification Model
# MAGIC
# MAGIC Let's go ahead and fit a Decision Tree model!

# COMMAND ----------

from math import sqrt
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# turn on autologging
mlflow.sklearn.autolog(log_input_examples=True)

# fit our model
dtc = DecisionTreeClassifier()
dtc_mdl = dtc.fit(X_train, y_train)

# evaluate the test set
y_predicted = dtc_mdl.predict(X_test)
test_acc = accuracy_score(y_test, y_predicted)
test_prec = precision_score(y_test, y_predicted)
test_rec = recall_score(y_test, y_predicted)
test_f1 = f1_score(y_test, y_predicted)
print("Test evaluation summary:")
print(f"Accuracy: {test_acc}")
print(f"Precision: {test_prec}")
print(f"Recall: {test_rec}")
print(f"F1: {test_f1}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine Model Result
# MAGIC
# MAGIC Examine the **confusion matrix** to visualize the model's classification performance.
# MAGIC
# MAGIC The confusion matrix provides insights into the model's performance, showing how many instances were correctly or incorrectly classified for each class.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Computing the confusion matrix
cm = confusion_matrix(y_test, y_predicted, labels=[1, 0])

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Now we examine the resulting model**
# MAGIC
# MAGIC We can extract and plot the `feature_importances_` inferred from this model to examine which data features are **the most critical for successful prediction**.

# COMMAND ----------

import numpy as np

# Retrieving feature importances
feature_importances = dtc_mdl.feature_importances_
feature_names = X_train.columns.to_list()

# Plotting the feature importances
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(feature_names))
plt.bar(y_pos, feature_importances, align='center', alpha=0.7)
plt.xticks(y_pos, feature_names, rotation=45)
plt.ylabel('Importance')
plt.title('Feature Importances in Decision Tree Classifier')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **We can also examine the resulting tree structure**
# MAGIC
# MAGIC **Decision trees make splitting decisions on different features at different critical values**, so we can visualize the resulting decision logic by plotting that branching tree structure.

# COMMAND ----------

print(f"The fitted DecisionTreeClassifier model has {dtc_mdl.tree_.node_count} nodes and is up to {dtc_mdl.tree_.max_depth} levels deep.")

# COMMAND ----------

# MAGIC %md
# MAGIC This is a very large decision tree, printing out the full tree logic, we can see it is vast and sprawling:

# COMMAND ----------

from sklearn.tree import export_text

text_representation = export_text(dtc_mdl, feature_names=feature_names)
print(text_representation)

# COMMAND ----------

# MAGIC %md
# MAGIC This plot will give you a visual representation of the decision tree structure, helping us to understand how the model makes predictions based on different features and their critical values.

# COMMAND ----------

# MAGIC %md
# MAGIC Since it is so big, we can only reasonably visualize a small portion of the tree at any given time. Here is the root and first 2 levels:

# COMMAND ----------

from sklearn.tree import plot_tree

plt.figure(figsize=(20,20))
plot_tree(dtc_mdl, 
          feature_names=feature_names,
          max_depth=2,
          class_names=['0', '1'], 
          filled=True)
plt.title('Decision Tree Structure')
plt.show()

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
# MAGIC This demonstration guided us through the process of training and evaluating a Decision Tree classification model for predicting diabetes. We started by preparing the dataset, conducting a train/test split, and fitting the model. The examination of the confusion matrix provided insights into the model's classification performance, and the visualization of the decision tree's structure and feature importances offered a deeper understanding of the underlying decision-making process.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>