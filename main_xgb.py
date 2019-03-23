import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, recall_score, precision_score, matthews_corrcoef
from sklearn import metrics
from sklearn.metrics.scorer import make_scorer
import xgboost as xgb
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


training_data = pd.read_csv('training_data/financial_data.csv') #9003 samples, 65 columns including CompanyID. Each sample correspond to one company
training_labels = pd.read_csv('training_data/revealed_businesses.csv') #4879 labels, [ID, 0/1]
testing_data = pd.read_csv('testing_data.csv')  #1500 samples, 65 columns including ID

def clean_data(df):
    df = df.replace('?', np.nan) # replace '?' by NaN
    df = df.astype(float) # convert string to float
    df = df.fillna(df.mean()) # fill NaN vlaues with the column mean
    return df

df_train = training_data.merge(training_labels, on='Var1')

df_train = df_train.drop(['Var1'], axis=1)

df_train = df_train.replace('?', -9999)

df_train = df_train.astype(float)

df_test = testing_data.drop(['Var1'], axis=1).replace('?', -9999).astype(float)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=39)

def my_matthews_corrcoef(y_true, y_pred):
  return matthews_corrcoef(y_true, y_pred)

my_MCC_scorer = make_scorer(my_matthews_corrcoef, greater_is_better=True)

xgb_model = xgb.XGBClassifier()


# parameters = {'objective':['binary:logistic','binary:hinge'],
#               'learning_rate': [0.15,0.2,0.25], #so called `eta` value
#               'max_depth': [3,4,5,6],
#               'subsample': [0.85,0.9,0.95],
#               'colsample_bytree': [0.7,0.8,0.9],
#               'gamma': [0,1,5],
#               'n_estimators': [1500, 2000], #number of trees, change it to 1000 for better results
#               'missing':[-9999],
#               'seed': [435]}


# only a few parameters to test if code works
parameters = {'objective':['binary:logistic'],
              'learning_rate': [0.05,0.1], #so called `eta` value
              'max_depth': [6],
              'n_estimators': [10,15], #number of trees, change it to 1000 for better results
              'missing':[-9999],
              'seed': [439]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=-1, cv=5, scoring=my_MCC_scorer, verbose=1, refit=True)

clf.fit(df_train.drop(['Var66'], axis=1), df_train['Var66'])   #, eval_metric=my_MCC_scorer)

print("Best params : ", clf.best_params_)
print("mean_test_score: ", max(clf.cv_results_['mean_test_score']))

y_testing_pred = clf.predict(df_test)

df_submit = pd.DataFrame(columns=['Business_ID','Is_Bankrupted'])
df_submit['Business_ID'] = testing_data['Var1']
df_submit['Is_Bankrupted'] = y_testing_pred
df_submit['Is_Bankrupted'] = df_submit['Is_Bankrupted'].astype(int)

# print(df_submit.head(20))

df_submit.to_csv("submit_8_xgb.csv", sep=',', encoding='utf-8', index=False)










