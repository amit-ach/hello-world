import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

df = pd.read_csv('missing_data.csv')
df1 = df.drop('Unnamed: 0', axis=1)

print("df1 shape: ", df1.shape)

df1['restricted_day_17'] = df1['restricted_day_17'].apply(lambda x: 1 if x >= 5 else 0 )
df1['restricted_day_18'] = df1['restricted_day_18'].apply(lambda x: 1 if x >= 5 else 0 )
df1['restricted_day_19'] = df1['restricted_day_19'].apply(lambda x: 1 if x >= 5 else 0 )

df1['lost_days_17'] = df1['lost_days_17'].apply(lambda x: 1 if x >= 5 else 0 )
df1['lost_days_18'] = df1['lost_days_18'].apply(lambda x: 1 if x >= 5 else 0 )
df1['lost_days_19'] = df1['lost_days_19'].apply(lambda x: 1 if x >= 5 else 0 )

df1["fatality_2017"] = df1['fatality_2017'].apply(lambda x: 1 if x > 0 else 0 )
df1['fatality_2018'] = df1['fatality_2018'].apply(lambda x: 1 if x > 0 else 0 )
df1['fatality_2019'] = df1['fatality_2019'].apply(lambda x: 1 if x > 0 else 0 )

df1["recordable_17"] = df1['recordable_17'].apply(lambda x: 1 if x > 0 else 0 )
df1['recordable_18'] = df1['recordable_18'].apply(lambda x: 1 if x > 0 else 0 )
df1['recordable_19'] = df1['recordable_19'].apply(lambda x: 1 if x > 0 else 0 )

X = df1
y = df1.iloc[:,-10:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

X_train = X_train[(X_train["fatal_19_mis"] == 1) & (X_train["fatal_18_mis"] == 1) & (X_train["fatal_17_mis"] == 1)]
X_test = X_test[(X_test["fatal_19_mis"] == 1) & (X_test["fatal_18_mis"] == 1) & (X_test["fatal_17_mis"] == 1)]
y_train = y_train[(y_train["fatal_19_mis"] == 1) & (y_train["fatal_18_mis"] == 1) & (y_train["fatal_17_mis"] == 1)]
y_test = y_test[(y_test["fatal_19_mis"] == 1) & (y_test["fatal_18_mis"] == 1) & (y_test["fatal_17_mis"] == 1)]

X_train = X_train.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
X_test = X_test.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
y_train = y_train.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)
y_test = y_test.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)

print("X train shape: ", X_train.shape)
print("x test Shape: ", X_test.shape)

X_train = X_train.iloc[:,1:]
output = X_test.iloc[:,:1]
X_test = X_test.iloc[:,1:]

y_train_fatality = y_train.iloc[:,:1]
y_train_restricted = y_train.iloc[:,1:2]
y_train_lost = y_train.iloc[:,2:3]
y_train_recordable = y_train.iloc[:,-1:]

y_test_fatality = y_test.iloc[:,:1]
y_test_restricted = y_test.iloc[:,1:2]
y_test_lost = y_test.iloc[:,2:3]
y_test_recordable = y_test.iloc[:,-1:]


def code(x):
    if x>0.9:
        return "green"
    elif x>0.7 and x<=0.9:
        return "yellow"
    elif x>0.5 and x<=0.7:
        return "orange"
    elif x<=0.5:
        return "red"


print("############### fatality #############")

print(y_train_fatality["fatality_2019"].value_counts())
print(y_test_fatality["fatality_2019"].value_counts())

clf_fatality = LogisticRegression(random_state=0).fit(X_train, y_train_fatality)

print("coef : ", clf_fatality.coef_)
print("intercept : ", clf_fatality.intercept_)

pred_fatality = clf_fatality.predict(X_test)
print(pred_fatality)

#####
count = 0
print(type(pred_fatality))
print(type(y_test_fatality))

fatality = list(y_test_fatality["fatality_2019"])
for a,b in zip(pred_fatality, fatality):
    if a == b:
        count+=1

print("matched values : ", count)
#####

print("For prediction for Fatlity_19 : ")

acc = clf_fatality.score(X_test, y_test_fatality) * 100
print('Accuracy Score : ', acc)

con_mat = confusion_matrix(y_test_fatality, pred_fatality)
print('Confusion Matrix : ', con_mat)

tn, fp, fn, tp = confusion_matrix(y_test_fatality, pred_fatality).ravel()
print("tn : ", tn)
print("fp : ", fp)
print("fn : ", fn)
print("tp : ", tp)

print("Log-loss : ", log_loss(y_test_fatality, pred_fatality))
print(classification_report(y_test_fatality, pred_fatality))

prob = pd.DataFrame()
prob["0"] = clf_fatality.predict_proba(X_train)[:,0]
print(prob["0"].apply(lambda x: code(x)).value_counts())

output["true_fatality"] = y_test_fatality["fatality_2019"]
output["fatality_19_pred"] = pred_fatality


print("########## LOST DAYS #########")

print(y_train_lost["lost_days_19"].value_counts())
print(y_test_lost["lost_days_19"].value_counts())

clf_lost = LogisticRegression(random_state=0).fit(X_train, y_train_lost)
pred_lost = clf_lost.predict(X_test)
print(pred_lost)

print("Coff for given points : ", clf_lost.coef_)
print("Intercept : ", clf_lost.intercept_)

print("For prediction for Lost Days : ")

acc = clf_lost.score(X_test, y_test_lost) * 100
print('Accuracy Score : ', acc)

con_mat = confusion_matrix(y_test_lost, pred_lost)
print('Confusion Matrix : ', con_mat)

tn, fp, fn, tp = confusion_matrix(y_test_lost, pred_lost).ravel()
print("tn : ", tn)
print("fp : ", fp)
print("fn : ", fn)
print("tp : ", tp)

print("Log-loss : ", log_loss(y_test_lost, pred_lost))

output["true_lost_day"] = y_test_lost["lost_days_19"]
output['lost_days_19_pred'] = pred_lost


print("########### Restricted #############")

print(y_train_restricted["restricted_day_19"].value_counts())
print(y_test_restricted["restricted_day_19"].value_counts())

clf_restricted = LogisticRegression(random_state=0).fit(X_train, y_train_restricted)

print("coef : ", clf_restricted.coef_)
print("intercept : ", clf_restricted.intercept_)

pred_restricted = clf_restricted.predict(X_test)
print(pred_restricted)

#####
count = 0
print(type(pred_restricted))
print(type(y_test_restricted))

res = list(y_test_restricted["restricted_day_19"])
for a,b in zip(pred_restricted, res):
    if a == b:
        count+=1

print("matched values : ", count)
#####

print("For prediction for Restricted Days : ")

acc = clf_restricted.score(X_test, y_test_restricted) * 100
print('Accuracy Score : ', acc)

con_mat = confusion_matrix(y_test_restricted, pred_restricted)
print('Confusion Matrix : ', con_mat)

tn, fp, fn, tp = confusion_matrix(y_test_restricted, pred_restricted).ravel()
print("tn : ", tn)
print("fp : ", fp)
print("fn : ", fn)
print("tp : ", tp)

print("Log-loss : ", log_loss(y_test_restricted, pred_restricted))

output["true_restricted_day"] = y_test_restricted["restricted_day_19"]
output['restricted_day_19_pred'] = pred_restricted


print("######### Recordable #########")

print(y_train_recordable["recordable_19"].value_counts())
print(y_test_recordable["recordable_19"].value_counts())

clf_recordable = LogisticRegression(random_state=0).fit(X_train, y_train_recordable)
pred_recordable = clf_recordable.predict(X_test)
print(pred_recordable)

print("Coff for given points : ", clf_recordable.coef_)
print("Intercept : ", clf_recordable.intercept_)

print("For prediction for Recordables : ")

acc = clf_recordable.score(X_test, y_test_recordable) * 100
print('Accuracy Score : ', acc)

con_mat = confusion_matrix(y_test_recordable, pred_recordable)
print('Confusion Matrix : ', con_mat)

tn, fp, fn, tp = confusion_matrix(y_test_recordable, pred_recordable).ravel()
print("tn : ", tn)
print("fp : ", fp)
print("fn : ", fn)
print("tp : ", tp)

print("Log-loss : ", log_loss(y_test_recordable, pred_recordable))

output["true_recordable"] = y_test_recordable["recordable_19"]
output['recordable_19_pred'] = pred_recordable

output = output.reset_index().drop('index', axis=1)
print(output.head(30))

output.to_csv('output_logistic_missing.csv')

print("*****************FATALITIES*********************")
print(classification_report(y_test_fatality, pred_fatality))
print("*****************RESTRICTED DAYS*********************")
print(classification_report(y_test_restricted, pred_restricted))
print("****************LOST DAYS**********************")
print(classification_report(y_test_lost, pred_lost))
print("****************RECORDABLE DAYS**********************")
print(classification_report(y_test_recordable, pred_recordable))
print("**************************************")

