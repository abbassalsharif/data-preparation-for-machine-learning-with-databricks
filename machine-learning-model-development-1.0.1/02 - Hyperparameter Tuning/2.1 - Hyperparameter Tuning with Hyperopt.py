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
# MAGIC # Hyperparameter Tuning with Hyperopt
# MAGIC
# MAGIC In this hands-on demo, you will learn how to leverage **Hyperopt**, a powerful optimization library, for efficient model tuning. We'll guide you through the process of performing **Bayesian hyperparameter optimization, demonstrating how to define the search space, objective function, and algorithm selection**. Throughout the demo, you will utilize *MLflow* to seamlessly track the model tuning process, capturing essential information such as hyperparameters, metrics, and intermediate results. By the end of the session, you will not only grasp the principles of hyperparameter optimization but also be proficient in finding the best-tuned model using various methods such as the **MLflow API** and **MLflow UI**.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Utilize hyperopt for model tuning.
# MAGIC
# MAGIC * Perform a Bayesian hyperparameter optimization using Hyperopt.
# MAGIC
# MAGIC * Track model tuning process with MLflow.
# MAGIC
# MAGIC * Query previous runs from an experiment using the `MLFlowClient`.
# MAGIC
# MAGIC * Review an MLflow Experiment for the best run.
# MAGIC
# MAGIC * Search and retrieve the best model.  
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

# MAGIC %run ../Includes/Classroom-Setup-02

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
# MAGIC
# MAGIC Before we start fitting a model, we need to prepare dataset. First, we will load dataset, then we will split it to train and test sets.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Dataset
# MAGIC
# MAGIC In this demo we will be using the CDC Diabetes dataset. This dataset has been loaded and loaded to a feature table. We will use this feature table to load data.

# COMMAND ----------

import mlflow.data

# load data from the feature table
table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes"
diabetes_dataset = mlflow.data.load_delta(table_name=table_name)
diabetes_pd =diabetes_dataset.df.drop("unique_id").toPandas()

# review dataset and schema
display(diabetes_pd)
print(diabetes_pd.info())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Train/Test Split
# MAGIC
# MAGIC Next, we will divide the dataset to training and testing sets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {diabetes_pd.shape[0]} records in our source dataset")

# split target variable into it's own dataset
target_col = "Diabetes_binary"
X_all = diabetes_pd.drop(labels=target_col, axis=1)
y_all = diabetes_pd[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.95, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Hyperparameter Search Space
# MAGIC
# MAGIC Hyperopt uses a [Bayesian optimization algorithm](https://hyperopt.github.io/hyperopt/#algorithms) to perform a more intelligent search of the hyperparameter space. Therefore, **the initial space definition is effectively a prior distribution over the hyperparameters**, which will be used as the starting point for the Bayesian optimization process. 
# MAGIC
# MAGIC Instead of defining a range or grid for each hyperparameter, we use [Hyperopt's parameter expressions](https://hyperopt.github.io/hyperopt/getting-started/search_spaces/#parameter-expressions) to define such prior distributions over parameter values.
# MAGIC

# COMMAND ----------

from hyperopt import hp

dtc_param_space = {
  'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
  'max_depth': hp.choice('dtree_max_depth',
                          [None, hp.uniformint('dtree_max_depth_int', 5, 50)]),
  'min_samples_split': hp.uniformint('dtree_min_samples_split', 2, 40),
  'min_samples_leaf': hp.uniformint('dtree_min_samples_leaf', 1, 20)
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Optimization Function
# MAGIC
# MAGIC We wrap our training code up as a function that we pass to hyperopt to optimize. The function takes a set of hyperparameter values as a `dict` and returns the validation loss score.
# MAGIC
# MAGIC **💡 Note:** We are using `f1` score as the cross-validated loss metric. As we goal of optimization function is to minimize the loss, we are returning `-f1`, in other words, **we want to maximize the `f1` score**.

# COMMAND ----------

from math import sqrt

import mlflow
import mlflow.data
import mlflow.sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

from hyperopt import STATUS_OK

def tuning_objective(params):
  # start an MLFlow run
  with mlflow.start_run(nested=True) as mlflow_run:
    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        disable=False,
        log_input_examples=True,
        silent=True,
        exclusive=False)

    # set up our model estimator
    dtc = DecisionTreeClassifier(**params)
    
    # cross-validated on the training set
    validation_scores = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(dtc, 
                                X_train, 
                                y_train, 
                                cv=5,
                                scoring=validation_scores)
    # log the average cross-validated results
    cv_score_results = {}
    for val_score in validation_scores:
      cv_score_results[val_score] = cv_results[f'test_{val_score}'].mean()
      mlflow.log_metric(f"cv_{val_score}", cv_score_results[val_score])

    # fit the model on all training data
    dtc_mdl = dtc.fit(X_train, y_train)

    # evaluate the model on the test set
    y_pred = dtc_mdl.predict(X_test)
    accuracy_score(y_test, y_pred)
    precision_score(y_test, y_pred)
    recall_score(y_test, y_pred)
    f1_score(y_test, y_pred)

    # return the negative of our cross-validated F1 score as the loss
    return {
      "loss": -cv_score_results['f1'],
      "status": STATUS_OK,
      "run": mlflow_run
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run in Hyperopt
# MAGIC
# MAGIC After defining the *objective function*, we are ready to run this function with hyperopt. 
# MAGIC
# MAGIC As you may have noticed, tuning process will need to test many models. We are going to create an instance of **`SparkTrials()` to parallelize hyperparameter tuning trials using Spark**. This is useful for distributing the optimization process across a Spark cluster.
# MAGIC
# MAGIC `SparkTrials` takes a **`parallelism` parameter, which specifies how many trials are run in parallel**. This parameter will depend on the compute resources available for the cluster. You can read more about how to choose the optimal `parallelism` value in this [blog post](https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html). 
# MAGIC
# MAGIC For search algorithm, we will choose the **TPE (Tree-structured Parzen Estimator) algorithm for optimization (`algo=tpe.suggest`)**.

# COMMAND ----------

from hyperopt import SparkTrials, fmin, tpe

# set the path for mlflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/Demo-2.1-Hyperparameter-Tuning-with-Hyperopt")

trials = SparkTrials(parallelism=4)
with mlflow.start_run(run_name="Model Tuning with Hyperopt Demo") as parent_run:
  fmin(tuning_objective,
      space=dtc_param_space,
      algo=tpe.suggest,
      max_evals=5,  # Increase this when widening the hyperparameter search space.
      trials=trials)

best_result = trials.best_trial["result"]
best_run = best_result["run"]

# COMMAND ----------

# MAGIC %md
# MAGIC Note that we used a **nested run** while tracking the tuning process. This means we can access to the *parent_run* and child runs. One of the runs we would definitely be interested in is the *best_run*. Let's check out these runs.

# COMMAND ----------

parent_run.info.run_id

# COMMAND ----------

best_run.info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find the Best Run
# MAGIC
# MAGIC In this section, we will search for registered models. There are couple ways for achieving this. We will show how to search runs using MLflow API, PySpark API and the UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the Best Run - MLFlow API
# MAGIC
# MAGIC Using the MLFlow API, you can search runs in an experiment, which returns results into a Pandas DafaFrame.

# COMMAND ----------

from mlflow.entities import ViewType

# search over all runs
hpo_runs_pd = mlflow.search_runs(
  experiment_ids=[parent_run.info.experiment_id],
  filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' AND attributes.status = 'FINISHED'",
  run_view_type=ViewType.ACTIVE_ONLY,
  order_by=["metrics.cv_f1 DESC"]
)

display(hpo_runs_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the Best Run - PySpark API
# MAGIC
# MAGIC Alternatively, you can read experiment results into a PySpark DataFrame and use standard Spark expressions to search runs in an experiment.

# COMMAND ----------

import pyspark.sql.functions as sfn

all_experiment_runs_df = spark.read.format("mlflow-experiment")\
  .load(parent_run.info.experiment_id)

hpo_runs_df = all_experiment_runs_df.where(f"tags['mlflow.parentRunId'] = '{parent_run.info.run_id}' AND status = 'FINISHED'")\
  .withColumn("cv_f1", sfn.col("metrics").getItem('cv_f1'))\
  .orderBy(sfn.col("cv_f1").desc() )

display(hpo_runs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the Best Run - MLflow UI
# MAGIC
# MAGIC The simplest way of seeing the tuning result is to use MLflow UI. 
# MAGIC
# MAGIC * Click on **Experiments** from left menu.
# MAGIC
# MAGIC * Select experiment which has the same name as this notebook's title (**2.1 - Hyperparameter Tuning with Hyperopt**).
# MAGIC
# MAGIC * View the **parent run** and **nested child runs**. 
# MAGIC
# MAGIC * You can filter and order by metrics and other model metadata.

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
# MAGIC To sum it up, this demo has shown you the process of tuning your models using Hyperopt and MLflow. You've learned a method to fine-tune your model settings through Bayesian optimization and how to keep tabs on the whole tuning journey with MLflow. Moving forward, these tools will be instrumental in improving your model's performance and simplifying the process of fine-tuning machine learning models.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>