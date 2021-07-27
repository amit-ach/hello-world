import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv', index_col = 0)
# print("===================")
print(df.columns)

def omit(val):
    if val is not '[]':
        return val[3:-4]
    else:
        return None

df['plan_name'] = df['plan_name'].apply(lambda x: omit(x))
df['primary_trade'] = df['primary_trade'].apply(lambda x: omit(x))
df['ladder_use'] = df['ladder_use'].apply(lambda x: omit(x))
df['scaffolds'] = df['scaffolds'].apply(lambda x: omit(x))
df['confined_space'] = df['confined_space'].apply(lambda x: omit(x))
df['aerial_lift'] = df['aerial_lift'].apply(lambda x: omit(x))
df['cpr_training'] = df['cpr_training'].apply(lambda x: omit(x))
df['excavation'] = df['excavation'].apply(lambda x: omit(x))
df['fall_protection'] = df['fall_protection'].apply(lambda x: omit(x))
df['investigation_prgm'] = df['investigation_prgm'].apply(lambda x: omit(x))
df['fatality_2017'] = df['fatality_2017'].apply(lambda x: omit(x))
df['fatality_2018'] = df['fatality_2018'].apply(lambda x: omit(x))
df['fatality_2019'] = df['fatality_2019'].apply(lambda x: omit(x))
df['restricted_day_17'] = df['restricted_day_17'].apply(lambda x: omit(x))
df['restricted_day_18'] = df['restricted_day_18'].apply(lambda x: omit(x))
df['lost_days_17'] = df['lost_days_17'].apply(lambda x: omit(x))
df['lost_days_18'] = df['lost_days_18'].apply(lambda x: omit(x))
df['recordable_17'] = df['recordable_17'].apply(lambda x: omit(x))
df['recordable_18'] = df['recordable_18'].apply(lambda x: omit(x))
df['work_hours_17'] = df['work_hours_17'].apply(lambda x: omit(x))
df['work_hours_18'] = df['work_hours_18'].apply(lambda x: omit(x))
df['emr_17'] = df['emr_17'].apply(lambda x: omit(x))
df['emr_18'] = df['emr_18'].apply(lambda x: omit(x))
df['flame_spark_activity'] = df['flame_spark_activity'].apply(lambda x: omit(x))
df['hoisting_rigging_eq'] = df['hoisting_rigging_eq'].apply(lambda x: omit(x))
df['live_electrical_component'] = df['live_electrical_component'].apply(lambda x: omit(x))
df['electrical_component'] = df['electrical_component'].apply(lambda x: omit(x))
df['safety_health_program'] = df['safety_health_program'].apply(lambda x: omit(x))
df['safety_manager'] = df['safety_manager'].apply(lambda x: omit(x))
df['emp_involvement_plan'] = df['emp_involvement_plan'].apply(lambda x: omit(x))
df['emp_performance_evalution'] = df['emp_performance_evalution'].apply(lambda x: omit(x))
df['osha_citation'] = df['osha_citation'].apply(lambda x: omit(x))
df['lost_days_19'] = df['lost_days_19'].apply(lambda x: omit(x))
df['restricted_day_19'] = df['restricted_day_19'].apply(lambda x: omit(x))
df['recordable_19'] = df['recordable_19'].apply(lambda x: omit(x))
df['dart_17'] = df['dart_17'].apply(lambda x: omit(x))
df['dart_18'] = df['dart_18'].apply(lambda x: omit(x))
df['dart_19'] = df['dart_19'].apply(lambda x: omit(x))
df['days_away_cases_17'] = df['days_away_cases_17'].apply(lambda x: omit(x))
df['days_away_cases_18'] = df['days_away_cases_18'].apply(lambda x: omit(x))
df['days_away_cases_19'] = df['days_away_cases_19'].apply(lambda x: omit(x))


df1 = df.copy()
df1["fatal_19_mis"] = df1["fatality_2019"].apply(lambda x: 0 if x is '' else 1)
df1["lost_days_19_mis"] = df1["lost_days_19"].apply(lambda x: 0 if x is '' else 1)
df1["restricted_mis"] = df1["restricted_day_19"].apply(lambda x: 0 if x is '' else 1)
df1["recordable_mis"] = df1["recordable_19"].apply(lambda x: 0 if x is '' else 1)
df1['fatal_18_mis'] = df1["fatality_2018"].apply(lambda x: 0 if x is '' else 1)
df1['fatal_17_mis'] = df1['fatality_2017'].apply(lambda x: 0 if x is '' else 1)



df["fatal_19_mis"] = df1["fatal_19_mis"]
df["lost_days_19_mis"] = df1["lost_days_19_mis"]
df["restricted_mis"] = df1["restricted_mis"]
df["recordable_mis"] = df1["recordable_mis"]
df["fatal_18_mis"] = df1["fatal_18_mis"]
df["fatal_17_mis"] = df1["fatal_17_mis"]


df.to_csv('new_un_data.csv')


## PRE PROCESSING

labelencoder = LabelEncoder()

df['plan_name'] = df['plan_name'].replace('', 'Missing')
df['plan_name'] = labelencoder.fit_transform(df['plan_name'])

df['primary_trade'] = df['primary_trade'].replace('', 'Misssing')
df['primary_trade'] = labelencoder.fit_transform(df['primary_trade'])

df['fatality_2017'] = pd.to_numeric(df['fatality_2017'], errors='coerce')
df['fatality_2017'] = df['fatality_2017'].fillna(0)
df['fatality_2017'] = df['fatality_2017'].astype(int)

df['fatality_2018'] = pd.to_numeric(df['fatality_2018'], errors='coerce')
df['fatality_2018'] = df['fatality_2018'].fillna(0)
df['fatality_2018'] = df['fatality_2018'].astype(int)

df['fatality_2019'] = pd.to_numeric(df['fatality_2019'], errors='coerce')
df['fatality_2019'] = df['fatality_2019'].fillna(0)
df['fatality_2019'] = df['fatality_2019'].astype(int)

df['restricted_day_17'] = pd.to_numeric(df['restricted_day_17'], errors='coerce')
df['restricted_day_17'] = df['restricted_day_17'].fillna(0)
df['restricted_day_17'] = df['restricted_day_17'].astype(int)

df['restricted_day_18'] = pd.to_numeric(df['restricted_day_18'], errors='coerce')
df['restricted_day_18'] = df['restricted_day_18'].fillna(0)
df['restricted_day_18'] = df['restricted_day_18'].astype(int)

df['restricted_day_19'] = pd.to_numeric(df['restricted_day_19'], errors='coerce')
df['restricted_day_19'] = df['restricted_day_19'].fillna(0)
df['restricted_day_19'] = df['restricted_day_19'].astype(int)

df['lost_days_17'] = pd.to_numeric(df['lost_days_17'], errors='coerce')
df['lost_days_17'] = df['lost_days_17'].fillna(0)
df['lost_days_17'] = df['lost_days_17'].astype(int)

df['lost_days_18'] = pd.to_numeric(df['lost_days_18'], errors='coerce')
df['lost_days_18'] = df['lost_days_18'].fillna(0)
df['lost_days_18'] = df['lost_days_18'].astype(int)

df['lost_days_19'] = pd.to_numeric(df['lost_days_19'], errors='coerce')
df['lost_days_19'] = df['lost_days_19'].fillna(0)
df['lost_days_19'] = df['lost_days_19'].astype(int)

df['recordable_17'] = pd.to_numeric(df['recordable_17'], errors='coerce')
df['recordable_17'] = df['recordable_17'].fillna(0)
df['recordable_17'] = df['recordable_17'].astype(int)

df['recordable_18'] = pd.to_numeric(df['recordable_18'], errors='coerce')
df['recordable_18'] = df['recordable_18'].fillna(0)
df['recordable_18'] = df['recordable_18'].astype(int)

df['recordable_19'] = pd.to_numeric(df['recordable_19'], errors='coerce')
df['recordable_19'] = df['recordable_19'].fillna(0)
df['recordable_19'] = df['recordable_19'].astype(int)

df['work_hours_17'] = pd.to_numeric(df['work_hours_17'], errors='coerce')
df['work_hours_17'] = df['work_hours_17'].fillna(0)
df['work_hours_17'] = df['work_hours_17'].astype(int)

df['work_hours_18'] = pd.to_numeric(df['work_hours_18'], errors='coerce')
df['work_hours_18'] = df['work_hours_18'].fillna(0)
df['work_hours_18'] = df['work_hours_18'].astype(int)

df['emr_17'] = pd.to_numeric(df['emr_17'], errors='coerce')
df['emr_17'] = df['emr_17'].fillna(0)
df['emr_17'] = df['emr_17'].astype(int)

df['emr_18'] = pd.to_numeric(df['emr_18'], errors='coerce')
df['emr_18'] = df['emr_18'].fillna(0)
df['emr_18'] = df['emr_18'].astype(int)

df['dart_17'] = pd.to_numeric(df['dart_17'], errors='coerce')
df['dart_17'] = df['dart_17'].fillna(0)
df['dart_17'] = df['dart_17'].astype(int)

df['dart_18'] = pd.to_numeric(df['dart_18'], errors='coerce')
df['dart_18'] = df['dart_18'].fillna(0)
df['dart_18'] = df['dart_18'].astype(int)

df['dart_19'] = pd.to_numeric(df['dart_19'], errors='coerce')
df['dart_19'] = df['dart_19'].fillna(0)
df['dart_19'] = df['dart_19'].astype(int)

df['days_away_cases_17'] = pd.to_numeric(df['days_away_cases_17'], errors='coerce')
df['days_away_cases_17'] = df['days_away_cases_17'].fillna(0)
df['days_away_cases_17'] = df['days_away_cases_17'].astype(int)

df['days_away_cases_18'] = pd.to_numeric(df['days_away_cases_18'], errors='coerce')
df['days_away_cases_18'] = df['days_away_cases_18'].fillna(0)
df['days_away_cases_18'] = df['days_away_cases_18'].astype(int)

df['days_away_cases_19'] = pd.to_numeric(df['days_away_cases_19'], errors='coerce')
df['days_away_cases_19'] = df['days_away_cases_19'].fillna(0)
df['days_away_cases_19'] = df['days_away_cases_19'].astype(int)



####     categorical data
df['ladder_use'] = df['ladder_use'].replace('', 'Missing')
df['ladder_use'] = labelencoder.fit_transform(df['ladder_use'])

df['scaffolds'] = df['scaffolds'].replace('', 'Missing')
df['scaffolds'] = labelencoder.fit_transform(df['scaffolds'])

df['confined_space'] = df['confined_space'].replace('', 'Missing')
df['confined_space'] = labelencoder.fit_transform(df['confined_space'])

df['aerial_lift'] = df['aerial_lift'].replace('', 'Missing')
df['aerial_lift'] = labelencoder.fit_transform(df['aerial_lift'])

df['cpr_training'] = df['cpr_training'].replace('', 'Missing')
df['cpr_training'] = labelencoder.fit_transform(df['cpr_training'])

df['excavation'] = df['excavation'].replace('', 'Missing')
df['excavation'] = labelencoder.fit_transform(df['excavation'])

df['fall_protection'] = df['fall_protection'].replace('', 'Missing')
df['fall_protection'] = labelencoder.fit_transform(df['fall_protection'])

df['investigation_prgm'] = df['investigation_prgm'].replace('', 'Missing')
df['investigation_prgm'] = labelencoder.fit_transform(df['investigation_prgm'])

df['flame_spark_activity'] = df['flame_spark_activity'].replace('', 'Missing')
df['flame_spark_activity'] = labelencoder.fit_transform(df['flame_spark_activity'])

df['hoisting_rigging_eq'] = df['hoisting_rigging_eq'].replace('', 'Missing')
df['hoisting_rigging_eq'] = labelencoder.fit_transform(df['hoisting_rigging_eq'])

df['live_electrical_component'] = df['live_electrical_component'].replace('', 'Missing')
df['live_electrical_component'] = labelencoder.fit_transform(df['live_electrical_component'])

df['electrical_component'] = df['electrical_component'].replace('', 'Missing')
df['electrical_component'] = labelencoder.fit_transform(df['electrical_component'])

df['safety_health_program'] = df['safety_health_program'].replace('', 'Missing')
df['safety_health_program'] = labelencoder.fit_transform(df['safety_health_program'])

df['safety_manager'] = df['safety_manager'].replace('', 'Missing')
df['safety_manager'] = labelencoder.fit_transform(df['safety_manager'])

df['emp_involvement_plan'] = df['emp_involvement_plan'].replace('', 'Missing')
df['emp_involvement_plan'] = labelencoder.fit_transform(df['emp_involvement_plan'])

df['emp_performance_evalution'] = df['emp_performance_evalution'].replace('', 'Missing')
df['emp_performance_evalution'] = labelencoder.fit_transform(df['emp_performance_evalution'])

df['osha_citation'] = df['osha_citation'].replace('', 'Missing')
df['osha_citation'] = labelencoder.fit_transform(df['osha_citation'])



# print(df[df['primary_trade'] == ''].primary_trade)

df1 = df.iloc[:,:-6]
df1.to_csv('cleaned_data.csv')
df.to_csv('missing_data.csv')

