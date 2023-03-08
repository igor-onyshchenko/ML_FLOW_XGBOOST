# Databricks notebook source
# MAGIC %pip install xgboost
# MAGIC %pip install mlflow
# MAGIC %pip install markdown
# MAGIC %pip install PyGithub
# MAGIC %pip install gitpython

# COMMAND ----------

### Import libraries
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import *
import pyspark.sql.functions as F
import math
import itertools

# from databricks.feature_store import *

from pyspark.sql.types import *
from pyspark.sql.functions import when, lit, substring, col, coalesce, lead, datediff, expr, posexplode, year, concat, lpad, month, last_day, add_months, trunc, date_add, greatest
from pyspark.sql.window import Window

### Import libraries
import pandas as pd
from datetime import date, datetime
import numpy as np
from random import seed, choice, randrange
from sklearn.metrics import mean_tweedie_deviance, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RandomizedSearchCV, GridSearchCV

from matplotlib import pyplot as plt
import seaborn as sns

import xgboost as xgb
import datetime as dt

import scipy as sc
from scipy.stats import ks_2samp

from statsmodels.stats.proportion import proportion_confint

import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import markdown


# COMMAND ----------

file_name = 'score_with_confidence_2022'

total2 = spark.read.table('es.{}'.format(file_name))

t2 = total2.toPandas()

# # moving history date for composing re-training dataset
history_date = str((dt.datetime.today() - dt.timedelta(18*30)).year) + '-' + str((dt.datetime.today() - dt.timedelta(18*30)).month) + '-01'
if len(history_date) < 10:
  train_test_date = history_date[:5] + '0' + history_date[5:]

# control - training set
train = t2[t2['POLICY_EFF_DT'] < train_test_date]
# test - comparison set
test = t2[t2['POLICY_EFF_DT'] >= train_test_date]

t2.head()

# COMMAND ----------

client = MlflowClient()

train['ks_test_grp'] = pd.qcut(train['pred_new_cab'], 50)
x_tmp = train.groupby('ks_test_grp').agg({'pred_new_cab': np.mean})

test['ks_test_grp'] = pd.qcut(test['pred_new_cab'], 50)
y_tmp = test.groupby('ks_test_grp').agg({'pred_new_cab': np.mean})

ax = plt.figure(figsize = (15,7))
ax = plt.plot(x_tmp['pred_new_cab'].values, label = 'Train')
ax = plt.plot(y_tmp['pred_new_cab'].values, label = 'Test')

ax = plt.plot(np.abs(x_tmp['pred_new_cab'].values-y_tmp['pred_new_cab'].values), label = 'KS values')
ax = plt.axhline(y=np.max(np.abs(x_tmp['pred_new_cab'].values-y_tmp['pred_new_cab'].values)), color = 'red')
ax = plt.ylabel('Predicted Frequency')
ax = plt.xlabel('Decile')
ax = plt.legend()
ax = plt.title('Train vs Test Score Distibution KS-test: '+ str(ks_2samp(train['pred_new_cab'], test['pred_new_cab'])))


# Save plots as .png and log as artifacts:
ax.figure.savefig('KS_SCORE_TEST.png')
mlflow.log_artifact("KS_SCORE_TEST.png")

mlflow.log_param('KS_2sample_test', ks_2samp(train['pred_new_cab'], test['pred_new_cab']))

print('Train vs Test Score Distibution KS-test: ', ks_2samp(train['pred_new_cab'], test['pred_new_cab']))



# COMMAND ----------

# current RUN_ID
curr_run_id = mlflow.active_run().info.run_id
curr_run_id

# COMMAND ----------


predictors = ['REGION','POLICY_SEGMENT', 'OPERATION_EXPOSURE_VEHICLE', 'PKGCOMBO', 'NEW_VENTURE', 'RADIUS_GROUP', 
              # Add Numeric Features
              'VEHICLE_NBR', 'MODEL_AGE', 'VEHICLE_COUNT_INV', 'YRSINBUSCT_INV',
              'YEARS_IN_BUSINESS', 'YRSINBUS_VEHICLE_COUNT',
              # categorical list
              'MODEL_AGE_GRP', 'EQUIPMENT_AGE_RADIUS_GROUP',
              'VEHICLE_TYPE_RADIUS_GROUP',
              'MODEL_AGE_RADIUS_GROUP', 
              'VEHICLE_TYPE_MODEL_AGE', 'REGION_RADIUS_GROUP', 'PKGCOMBO_RADIUS_GROUP', 'OPERATION_EXPOSURE_VEHICLE_RADIUS_GROUP',
              'EQUIPMENT_AGE_PKG_COMBO', 'EQUIPMENT_AGE_REGION', 'EQUIPMENT_AGE_VEHICLE_TYPE', 'EQUIPMENT_AGE_OPERATION_EXPOSURE_VEHICLE',
              #CAB features
              'DRIVER_OUT_OF_SERVICE_RATIO',
              'VEHICLE_OUT_OF_SERVICE_RATIO',
              'DRIVER_OUT_OF_SERVICE',
#               'COUNT_BASIC_ALERTS',
              'DRIVER_POWER_UNIT_RATIO',
              'SPEEDING_VIOLATIONS',
              'TOTAL_CRASHES']

# This function takes a new data set, an initial data set, and a variable name as input to calculate counts, percentages and PSI values for each class of the input variable, and return a dataframe
def psi_table(old, new, varlist):
  df = pd.DataFrame()
  for var in varlist:
    new_total = new.count()
    old_total = old.count()

    new_grouped = new.groupBy(var).count().orderBy(var) \
      .withColumnRenamed('count', 'new_ct') \
      .withColumnRenamed(var, 'class_new') \
      .withColumn('new_perc', col('new_ct') / new_total) \
      .na.fill(value='Blank',subset=["class_new"])
    
    old_grouped = old.groupBy(var).count().orderBy(var) \
      .withColumnRenamed(var, 'class_old') \
      .withColumnRenamed('count', 'old_ct') \
      .withColumn('old_perc', col('old_ct') / old_total) \
      .na.fill(value='Blank',subset=["class_old"])

    final = old_grouped.join(new_grouped,old_grouped['class_old'] ==  new_grouped['class_new'],"outer") \
      .fillna(0, subset=['old_ct', 'new_ct']) \
      .fillna(0.001, subset=['old_perc', 'new_perc']) \
      .withColumn('Feature', lit(str(var))) \
      .withColumn('Class', coalesce(col('class_old'), col('class_new')).cast(StringType())) \
      .select(['Feature', 'Class', "old_ct", "new_ct", "old_perc","new_perc"]) \
      .withColumn('PSI',(col('new_perc') - col('old_perc'))*log10(col('new_perc')/col('old_perc'))) \
      .toPandas().round(8)

    if len(df)==0:
      df = final
    else:
      df = df.append(final,ignore_index=True)
  return(df)

#function to generate plots of the populations for each class of given feature
def plot_shift(psi_full, feature: str, old_label: str, new_label: str):
  plotdf = psi_full[psi_full['Feature']==feature]

  class_labels = plotdf['Class'].tolist()

  old = plotdf['old_perc']
  new = plotdf['new_perc']

  x = np.arange(len(class_labels))  # the label locations
  width = 0.35  # the width of the bars
  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/2, old, width, label=old_label)
  rects2 = ax.bar(x + width/2, new, width, label=new_label)
  fig.set_size_inches(10, 7, forward=True)
  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('% of Population')
  ax.set_title(f"{feature}: {old_label} vs. {new_label}")
  ax.set_xticks(x)
  ax.set_xticklabels(class_labels,rotation = 90)
  ax.legend()
  
  # Save plots as .png and log as artifacts:
  ax.figure.savefig('psi_{}.png'.format(feature))
  mlflow.log_artifact('psi_{}.png'.format(feature))
  
  return(plt)

# This function take a grouped dataframe with calculated PSI for each class/bucket and returns an aggregated PSI for that variable
def psi_sum(psi_df):
  out = psi_df[['Feature', 'PSI']].groupby('Feature',as_index=False).sum().round(6).sort_values(by=['PSI'], ascending=False)
  return(out)

# COMMAND ----------

df_train = total2.filter(col('POLICY_EFF_DT')<train_test_date)
df_test = total2.filter(col('POLICY_EFF_DT') >= train_test_date)

psi_full = psi_table(df_train, df_test, predictors)
display(psi_full)
psi = psi_sum(psi_full)

# Save plots as .png and log as artifacts:
ax = plt.figure(figsize = (10,10))
ax = plt.barh(y = psi.sort_values(by='PSI')['Feature'], width = psi.sort_values(by='PSI')['PSI'])
ax = plt.axvline(x=0.25, color = 'red')
ax = plt.axvline(x=0.15, color = 'yellow')

ax = plt.ylabel('Feature Name')
ax = plt.xlabel('Population Stability Index')
ax = plt.legend()
ax = plt.title('Population Stability Index by Features')

# Save plots as .png and log as artifacts:
ax.figure.savefig('PSI_features.png')
mlflow.log_artifact("PSI_features.png")

# save .csv file with data, prediction and CI as artifact
psi.to_csv('psi_features.csv')
mlflow.log_artifact('psi_features.csv')

display(psi)

# COMMAND ----------


for feature in predictors:
  plot_shift(psi_full, feature,'Before {}'.format(train_test_date),'After {}'.format(train_test_date))
  

# COMMAND ----------

# markdown.markdown(psi.to_markdown(), extensions=['tables'])

psi['img_link'] = '!['+psi['Feature']+'](https://nationwide-ess-prod-virginia.cloud.databricks.com/_mlflow/get-artifact?path=psi_'+psi['Feature']+'.png&run_uuid={})'.format(curr_run_id)

psi
# https://nationwide-ess-prod-virginia.cloud.databricks.com/_mlflow/get-artifact?path=KS_SCORE_TEST.png&run_uuid=4326eec1bbf3469b86b3c154a1d615a6

# ![KS_SCORE_TEST_IMAGE](https://nationwide-ess-prod-virginia.cloud.databricks.com/_mlflow/get-artifact?path=KS_SCORE_TEST.png&run_uuid=4326eec1bbf3469b86b3c154a1d615a6

# COMMAND ----------

import markdown
html = markdown.markdown("""
#[**Rubix 5 Model Monitoring Report**]


## Image
<p>{}</p>
![KS_SCORE_TEST_IMAGE](https://nationwide-ess-prod-virginia.cloud.databricks.com/_mlflow/get-artifact?path=KS_SCORE_TEST.png&run_uuid={})

## PSI Summary Table
![PSI_SUMMARY_TABLE](https://nationwide-ess-prod-virginia.cloud.databricks.com/_mlflow/get-artifact?path=PSI_features.png&run_uuid={})

## PSI by Feature Detailed Table
{}

## HTML
<b>Bold</b> text
""".format( 
  #Section 1. print KS-test for Score DIstribution
  'Train vs Test Score Distibution KS-test: ' + str(np.round(ks_2samp(train['pred_new_cab'], test['pred_new_cab'])[0],4)),
  curr_run_id,
  #Section 2. PSI Summary table
  curr_run_id,
  
  #Section 3. PSI by Feature Detailed Table
  markdown.markdown(psi.to_markdown(index = False), extensions=['tables'])

  )
)

with open('markdown.html', 'w') as f:
    f.write(html)

mlflow.log_artifact('markdown.html')

# COMMAND ----------

1+1

# access Github Token 
# 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

# COMMAND ----------

# import github
from github import InputGitTreeElement

# COMMAND ----------

import base64
from github import Github
from github import InputGitTreeElement

# user = "onyshi1"
# password = "Ihobot44$$44"
# g = Github(user,password)
# repo = g.get_user().get_repo('Nationwide/Rubix5_Monitoring') # repo name

# create the GH client correctly
g = Github(login_or_token='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', base_url='https://github.nwie.net/api/v3')
# create an instance of an AuthenticatedUser, still without actually logging in
user = g.get_user()
print(user) # will print 'AuthenticatedUser(login=None)'
# now, invoke the lazy-loading of the user
login = user.login
print(user) # will print 'AuthenticatedUser(login=<username_of_logged_in_user>)'
print(login) # will print <username_of_logged_in_user>

#available repositories
for repo in user.get_repos():
  print(repo)

repo = user.get_repo("Rubix5_Monitoring")
print('current repo: ', repo)

# COMMAND ----------

master_ref = repo.get_git_ref('.')


# COMMAND ----------

with open('markdown.html') as input_file:
  data = input_file.read()
  
element = InputGitTreeElement('markdown.html', mode="100644", type="blob", data)



# COMMAND ----------

file_list = [
    'markdown.html',
    'PSI_features.png'
]
file_names = [
    'markdown.html',
    'PSI_features.png'
]
commit_message = 'python commit'
master_ref = repo.get_git_ref('heads/master')
master_sha = master_ref.object.sha
base_tree = repo.get_git_tree(master_sha)

element_list = list()
for i, entry in enumerate(file_list):
    with open(entry) as input_file:
        data = input_file.read()
    if entry.endswith('.png'): # images must be encoded
        data = base64.b64encode(data)
    element = InputGitTreeElement(file_names[i], '100644', 'blob', data)
    element_list.append(element)

tree = repo.create_git_tree(element_list, base_tree)
parent = repo.get_git_commit(master_sha)
commit = repo.create_git_commit(commit_message, tree, [parent])
master_ref.edit(commit.sha)

# COMMAND ----------

# Save plots as .png and log as artifacts:
ax.figure.savefig('PSI_features.png')
# mlflow.log_artifact("PSI_features.png")

# COMMAND ----------

import sys
import os

os.path,
sys.path


# COMMAND ----------


ax.figure.savefig('docs/PSI_features.png')


# COMMAND ----------

https://nationwide-ess-prod-virginia.cloud.databricks.com/_mlflow/get-artifact?path=KS_SCORE_TEST.png&run_uuid=355c9e4872d04d9a9f8cab22beac6510

dbfs:/databricks/mlflow-tracking/5682aa3a833a4fc2a4526cf9e6cb354e/355c9e4872d04d9a9f8cab22beac6510/artifacts/KS_SCORE_TEST.png

# COMMAND ----------

with open('markdown.html') as input_file:
  data = input_file.read()
  
element = InputGitTreeElement('markdown.html', '100644', 'blob', data)

# COMMAND ----------

commit_message = 'python commit'
base_tree = repo.get_git_tree()

# COMMAND ----------

commit_message = 'python commit'
master_ref = repo.get_git_ref('')
master_sha = master_ref.object.sha
base_tree = repo.get_git_tree(master_sha)


# COMMAND ----------

import sys
import os
 
# In the command below, replace <username> with your Databricks user name.
sys.path.append(os.path.abspath('/Workspace/Repos/onyshi1@nationwide.com/Rubix5_Monitoring'))


# COMMAND ----------

ax.figure.savefig('/Workspace/Repos/onyshi1@nationwide.com/Rubix5_Monitoring/docs/PSI_features2.png')

# COMMAND ----------



# COMMAND ----------

# Clusteing/segmentation thoughts

1. categorical to numeric
2. numeric - normalizing or not?
3. why the clusters are different
4. technical/metrics distance
5. logical - business distance
6. type of clustering - dbscan or umap...
7. How to select features

