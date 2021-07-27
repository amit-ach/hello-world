import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('missing_data.csv')
df1 = df.drop('Unnamed: 0', axis=1)

print("shape of df1 : ", df1.shape)

df1['restricted_day_17'] = df1['restricted_day_17'].apply(lambda x: 1 if x >= 5 else 0 )
df1['restricted_day_18'] = df1['restricted_day_18'].apply(lambda x: 1 if x >= 5 else 0 )
df1['restricted_day_19'] = df1['restricted_day_19'].apply(lambda x: 1 if x >= 5 else 0 )

df1['lost_days_17'] = df1['lost_days_17'].apply(lambda x: 1 if x >= 5 else 0 )
df1['lost_days_18'] = df1['lost_days_18'].apply(lambda x: 1 if x >= 5 else 0 )
df1['lost_days_19'] = df1['lost_days_19'].apply(lambda x: 1 if x >= 5 else 0 )

X = df1
y = df1.iloc[:,-10:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

X_train_fatal = X_train[(X_train["fatal_19_mis"] == 1) & (X_train["fatal_18_mis"] == 1) & (X_train["fatal_17_mis"] == 1)]
X_test_fatal = X_test[(X_test["fatal_19_mis"] == 1) & (X_test["fatal_18_mis"] == 1) & (X_test["fatal_17_mis"] == 1)]
y_train_fatal = y_train[(y_train["fatal_19_mis"] == 1) & (y_train["fatal_18_mis"] == 1) & (y_train["fatal_17_mis"] == 1)]
y_test_fatal = y_test[(y_test["fatal_19_mis"] == 1) & (y_test["fatal_18_mis"] == 1) & (y_test["fatal_17_mis"] == 1)]

X_train_restricted = X_train[X_train["restricted_mis"] == 1]
X_test_restricted = X_test[X_test["restricted_mis"] == 1]
y_train_restricted = y_train[y_train["restricted_mis"] == 1]
y_test_restricted = y_test[y_test["restricted_mis"] == 1]

X_train_lost = X_train[X_train["lost_days_19_mis"] == 1]
X_test_lost = X_test[X_test["lost_days_19_mis"] == 1]
y_train_lost = y_train[y_train["lost_days_19_mis"] == 1]
y_test_lost = y_test[y_test["lost_days_19_mis"] == 1]

X_train_recordable = X_train[X_train["recordable_mis"] == 1]
X_test_recordable = X_test[X_test["recordable_mis"] == 1]
y_train_recordable = y_train[y_train["recordable_mis"] == 1]
y_test_recordable = y_test[y_test["recordable_mis"] == 1]

X_train_fatal = X_train_fatal.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
X_test_fatal = X_test_fatal.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
y_train_fatal = y_train_fatal.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)
y_test_fatal = y_test_fatal.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)

X_train_restricted = X_train_restricted.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
X_test_restricted = X_test_restricted.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
y_train_restricted = y_train_restricted.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)
y_test_restricted = y_test_restricted.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)

X_train_lost = X_train_lost.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
X_test_lost = X_test_lost.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
y_train_lost = y_train_lost.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)
y_test_lost = y_test_lost.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)

X_train_recordable = X_train_recordable.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
X_test_recordable = X_test_recordable.drop(["fatality_2019", "restricted_day_19", "lost_days_19", "recordable_19", "fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis = 1)
y_train_recordable = y_train_recordable.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)
y_test_recordable = y_test_recordable.drop(["fatal_19_mis", "lost_days_19_mis", "restricted_mis", "recordable_mis", "fatal_18_mis", "fatal_17_mis"], axis=1)

print("X train shape: ", X_train.shape)
print("x test Shape: ", X_test.shape)

print("fatal")
print("X train shape: ", X_train_fatal.shape)
print("x test Shape: ", X_test_fatal.shape)

print("restricted")
print("X train shape: ", X_train_restricted.shape)
print("x test Shape: ", X_test_restricted.shape)

print("lost")
print("X train shape: ", X_train_lost.shape)
print("x test Shape: ", X_test_lost.shape)

print("recordable")
print("X train shape: ", X_train_recordable.shape)
print("x test Shape: ", X_test_recordable.shape)

X_train_fatal = X_train_fatal.iloc[:,1:]
X_train_restricted = X_train_restricted.iloc[:,1:]
X_train_lost = X_train_lost.iloc[:,1:]
X_train_recordable = X_train_recordable.iloc[:,1:]

output_fatal = X_test_fatal.iloc[:,:1]
X_test_fatal = X_test_fatal.iloc[:,1:]

output_restricted = X_test_restricted.iloc[:,:1]
X_test_restricted = X_test_restricted.iloc[:,1:]

output_lost = X_test_lost.iloc[:,:1]
X_test_lost = X_test_lost.iloc[:,1:]

output_recordable = X_test_recordable.iloc[:,:1]
X_test_recordable = X_test_recordable.iloc[:,1:]

y_train_fatality = y_train_fatal.iloc[:,:1]
y_train_restricted = y_train_restricted.iloc[:,1:2]
y_train_lost = y_train_lost.iloc[:,2:3]
y_train_recordable = y_train_recordable.iloc[:,-1:]

y_test_fatality = y_test_fatal.iloc[:,:1]
y_test_restricted = y_test_restricted.iloc[:,1:2]
y_test_lost = y_test_lost.iloc[:,2:3]
y_test_recordable = y_test_recordable.iloc[:,-1:]

print("########## FATALITY #############")

print(y_train_fatality["fatality_2019"].value_counts())
print(y_test_fatality["fatality_2019"].value_counts())

reg = linear_model.LinearRegression()
reg.fit(X_train_fatal, y_train_fatality)

print("Coff for given points : ", reg.coef_)
print("Intercept : ", reg.intercept_)

pred_fatality = reg.predict(X_test_fatal)
pred_fatality = np.around(pred_fatality).astype(int)

print("For prediction for Fataliity : ")
## RMSE
rmse = np.sqrt(np.square(np.subtract(y_test_fatality, pred_fatality)).mean())
print('RMSE :' + str(rmse))

print("Score : ",reg.score(X_train_fatal, y_train_fatality))

acc = accuracy_score(y_test_fatality, pred_fatality)*100
print('Accuracy Score : ', acc)

con_mat = confusion_matrix(y_test_fatality, pred_fatality)
print('Confusion Matrix : ', con_mat)

output_fatal["true_fatality"] = y_test_fatality["fatality_2019"]
output_fatal["fatality_19_pred"] = pred_fatality


#####
count_0 = 0
count_1 = 0
#print(type(pred_fatality))
#print(type(y_test_fatality))

fatality = list(y_test_fatality["fatality_2019"])
for a,b in zip(pred_fatality, fatality):
    if b == 0 and a == b:
        count_0+=1
    if b!=0 and a == b:
        count_1+=1

print("matched values for not-event and event : ", count_0, count_1)
#####

y_test_fatality["fatality_2019"].value_counts()



print("########## LOST DAYS #########")

print(y_train_lost["lost_days_19"].value_counts())
print(y_test_lost["lost_days_19"].value_counts())

clf_lost = LogisticRegression(random_state=0).fit(X_train_lost, y_train_lost)
pred_lost = clf_lost.predict(X_test_lost)
print(pred_lost)

print("Coff for given points : ", clf_lost.coef_)
print("Intercept : ", clf_lost.intercept_)

print("For prediction for Lost Days : ")

acc = clf_lost.score(X_test_lost, y_test_lost) * 100
print('Accuracy Score : ', acc)

con_mat = confusion_matrix(y_test_lost, pred_lost)
print('Confusion Matrix : ', con_mat)

tn, fp, fn, tp = confusion_matrix(y_test_lost, pred_lost).ravel()
print("tn : ", tn)
print("fp : ", fp)
print("fn : ", fn)
print("tp : ", tp)

print("Log-loss : ", log_loss(y_test_lost, pred_lost))

output_lost["true_lost_day"] = y_test_lost["lost_days_19"]
output_lost['lost_days_19_pred'] = pred_lost


print("########### Restricted #############")

print(y_train_restricted["restricted_day_19"].value_counts())
print(y_test_restricted["restricted_day_19"].value_counts())

clf_restricted = LogisticRegression(random_state=0).fit(X_train_restricted, y_train_restricted)

print("coef : ", clf_restricted.coef_)
print("intercept : ", clf_restricted.intercept_)

pred_restricted = clf_restricted.predict(X_test_restricted)
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

acc = clf_restricted.score(X_test_restricted, y_test_restricted) * 100
print('Accuracy Score : ', acc)

con_mat = confusion_matrix(y_test_restricted, pred_restricted)
print('Confusion Matrix : ', con_mat)

tn, fp, fn, tp = confusion_matrix(y_test_restricted, pred_restricted).ravel()
print("tn : ", tn)
print("fp : ", fp)
print("fn : ", fn)
print("tp : ", tp)

print("Log-loss : ", log_loss(y_test_restricted, pred_restricted))

output_restricted["true_restricted_day"] = y_test_restricted["restricted_day_19"]
output_restricted['restricted_day_19_pred'] = pred_restricted



print("############# Recordable ############")
print(y_train_recordable["recordable_19"].value_counts())
print(y_test_recordable["recordable_19"].value_counts())

reg = linear_model.LinearRegression()
reg.fit(X_train_recordable, y_train_recordable)
pred_recordable = reg.predict(X_test_recordable)
pred_recordable = np.around(pred_recordable).astype(int)
pred_recordable[pred_recordable < 0] = 0

print("Coff for given points : ", reg.coef_)
print("Intercept : ", reg.intercept_)

print("For prediction for Recordables : ")
## RMSE
rmse = np.sqrt(np.square(np.subtract(y_test_recordable, pred_recordable)).mean())
print('RMSE :' + str(rmse))

print('Coefficients: \n', reg.coef_)

acc = accuracy_score(y_test_recordable, pred_recordable)*100
print('Accuracy Score : ', acc)

con_mat = confusion_matrix(y_test_recordable, pred_recordable)
print('Confusion Matrix : ', con_mat)

output_recordable['true_recordable'] = y_test_recordable["recordable_19"]
output_recordable['recordable_19_pred'] = pred_recordable

#####
count_0 = 0
count_1 = 0
#print(type(pred_fatality))
#print(type(y_test_fatality))

record = list(y_test_recordable["recordable_19"])
for a,b in zip(pred_recordable, record):
    if b == 0 and a == b:
        count_0+=1
    if b!=0 and a == b:
        count_1+=1

print("matched values for non-events and events : ", count_0, count_1)
#####


output_fatal = output_fatal.reset_index().drop('index', axis=1)
output_restricted = output_restricted.reset_index().drop('index', axis=1)
output_lost = output_lost.reset_index().drop('index', axis=1)
output_recordable = output_recordable.reset_index().drop('index', axis=1)
print(output_fatal.head(30))
print(output_restricted.head(30))
print(output_lost.head(30))
print(output_recordable.head(30))

output_fatal.to_csv('output_fatal.csv')
output_restricted.to_csv('output_restricted.csv')
output_lost.to_csv('output_lost.csv')
output_recordable.to_csv('output_recordable.csv')

print("*****************RESTRICTED DAYS*********************")
print(classification_report(y_test_restricted, pred_restricted))
print("****************LOST DAYS**********************")
print(classification_report(y_test_lost, pred_lost))


