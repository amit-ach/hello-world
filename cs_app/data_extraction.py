import psycopg2
import pandas as pd
import numpy as np
import string
from extracting_functions import *
from config import config_psql

conn = None
try:
    print('Connecting to the PostgreSQL database...')

    params_ = config_psql()

    conn = psycopg2.connect(**params_)
        
    # create a cursor
    cur = conn.cursor()

    cur.execute("select contractor_id from contractor;")
    con_id = cur.fetchall()

    #print(con_id)
    df = pd.DataFrame(con_id, columns =['contractor_id'])

    df['plan_name'] = df['contractor_id'].apply(lambda x: plan_n(x, cur))
    df['primary_trade'] = df['contractor_id'].apply(lambda x: trade_n(x, cur))
    df['ladder_use'] = df['contractor_id'].apply(lambda x: ladder(x, cur))
    df['scaffolds'] = df['contractor_id'].apply(lambda x: rolling_stage(x, cur))
    df['confined_space'] = df['contractor_id'].apply(lambda x: confined_space(x, cur))
    df['aerial_lift'] = df['contractor_id'].apply(lambda x: aerial_lift(x, cur))
    df['cpr_training'] = df['contractor_id'].apply(lambda x: cpr_training(x, cur))
    df['excavation'] = df['contractor_id'].apply(lambda x: excavation(x, cur))
    df['fall_protection'] = df['contractor_id'].apply(lambda x: fall_protection(x, cur))
    df['investigation_prgm'] = df['contractor_id'].apply(lambda x: investigation_prgm(x, cur))
    df['fatality_2017'] = df['contractor_id'].apply(lambda x: fatality_17(x, cur))
    df['fatality_2018'] = df['contractor_id'].apply(lambda x: fatality_18(x, cur))
    df['restricted_day_17'] = df['contractor_id'].apply(lambda x: restricted_day_17(x, cur))
    df['restricted_day_18'] = df['contractor_id'].apply(lambda x: restricted_day_18(x, cur))
    df['lost_days_17'] = df['contractor_id'].apply(lambda x: lost_days_17(x, cur))
    df['lost_days_18'] = df['contractor_id'].apply(lambda x: lost_days_18(x, cur))
    df['recordable_17'] = df['contractor_id'].apply(lambda x: recordable_17(x, cur))
    df['recordable_18'] = df['contractor_id'].apply(lambda x: recordable_18(x, cur))
    df['work_hours_17'] = df['contractor_id'].apply(lambda x: work_hours_17(x, cur))
    df['work_hours_18'] = df['contractor_id'].apply(lambda x: work_hours_18(x, cur))
    df['emr_17'] = df['contractor_id'].apply(lambda x: emr_17(x, cur))
    df['emr_18'] = df['contractor_id'].apply(lambda x: emr_18(x, cur))
    df['flame_spark_activity'] = df['contractor_id'].apply(lambda x: flame_spark_activity(x, cur))
    df['hoisting_rigging_eq'] = df['contractor_id'].apply(lambda x: hoisting_rigging_eq(x, cur))
    df['live_electrical_component'] = df['contractor_id'].apply(lambda x: live_electrical_comp(x, cur))
    df['electrical_component'] = df['contractor_id'].apply(lambda x: electrical_comp(x, cur))
    df['safety_health_program'] = df['contractor_id'].apply(lambda x: safety_program(x, cur))
    df['safety_manager'] = df['contractor_id'].apply(lambda x: safety_manager(x, cur))
    df['emp_involvement_plan'] = df['contractor_id'].apply(lambda x: emp_involvement(x, cur))
    df['emp_performance_evalution'] = df['contractor_id'].apply(lambda x: emp_performance(x, cur))
    df['osha_citation'] = df['contractor_id'].apply(lambda x: osha_citation(x, cur))
    df['fatality_2019'] = df['contractor_id'].apply(lambda x: fatality_2019(x, cur))
    df['restricted_day_19'] = df['contractor_id'].apply(lambda x: restricted_2019(x, cur))
    df['lost_days_19'] = df['contractor_id'].apply(lambda x: lost_days_2019(x, cur))
    df['recordable_19'] = df['contractor_id'].apply(lambda x: recordable_2019(x, cur))
    df['dart_17'] = df['contractor_id'].apply(lambda x: dart_17(x, cur))
    df['dart_18'] = df['contractor_id'].apply(lambda x: dart_18(x, cur))
    df['dart_19'] = df['contractor_id'].apply(lambda x: dart_19(x, cur))
    df['days_away_cases_17'] = df['contractor_id'].apply(lambda x: da_cases_17(x, cur))
    df['days_away_cases_18'] = df['contractor_id'].apply(lambda x: da_cases_18(x, cur))
    df['days_away_cases_19'] = df['contractor_id'].apply(lambda x: da_cases_19(x, cur))

    print(df.head(20))
    
    df.to_csv('data.csv')

    cur.close()
except(Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print("Database connection closed.")


