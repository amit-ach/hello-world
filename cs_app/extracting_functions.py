
def plan_n(id_, cur):
    pg_query = "select plan_name from plan where plan_id = (select plan_id from contractor where contractor_id = "+ str(id_) +");"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def trade_n(id_, cur):
    pg_query = "select trade_name from trade where trade_id = (select trade_id from contractor_trade where is_primary = true and contractor_id = "+ str(id_) +")"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result
    
def ladder(id_, cur):
    pg_query = "select answer from question_response where question_id = 2082 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def rolling_stage(id_, cur):
    pg_query = "select answer from question_response where question_id = 2084 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def confined_space(id_, cur):
    pg_query = "select answer from question_response where question_id = 2104 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def aerial_lift(id_, cur):
    pg_query = "select answer from question_response where question_id = 2108 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def cpr_training(id_, cur):
    pg_query = "select answer from question_response where question_id = 2116 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def excavation(id_, cur):
    pg_query = "select answer from question_response where question_id = 2068 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def fall_protection(id_, cur):
    pg_query = "select answer from question_response where question_id = 2057 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def investigation_prgm(id_, cur):
    pg_query = "select answer from question_response where question_id = 2015 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def fatality_17(id_, cur):
    pg_query = "select answer from question_response where question_id = 1835 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def fatality_18(id_, cur):
    pg_query = "select answer from question_response where question_id = 1826 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def restricted_day_17(id_, cur):
    pg_query = "select answer from question_response where question_id = 1834 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def restricted_day_18(id_, cur):
    pg_query = "select answer from question_response where question_id = 1834 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def lost_days_17(id_, cur):
    pg_query = "select answer from question_response where question_id = 1833 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def lost_days_18(id_, cur):
    pg_query = "select answer from question_response where question_id = 1824 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def recordable_17(id_, cur):
    pg_query = "select answer from question_response where question_id = 1831 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def recordable_18(id_, cur):
    pg_query = "select answer from question_response where question_id = 1822 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def work_hours_17(id_, cur):
    pg_query = "select answer from question_response where question_id = 1836 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def work_hours_18(id_, cur):
    pg_query = "select answer from question_response where question_id = 1827 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def emr_17(id_, cur):
    pg_query = "select answer from question_response where question_id = 1914 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def emr_18(id_, cur):
    pg_query = "select answer from question_response where question_id = 1911 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def flame_spark_activity(id_, cur):
    pg_query = "select answer from question_response where question_id = 2086 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def hoisting_rigging_eq(id_, cur):
    pg_query = "select answer from question_response where question_id = 2076 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def live_electrical_comp(id_, cur):
    pg_query = "select answer from question_response where question_id = 2074 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def electrical_comp(id_, cur):
    pg_query = "select answer from question_response where question_id = 2072 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def safety_program(id_, cur):
    pg_query = "select answer from question_response where question_id = 2011 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def safety_manager(id_, cur):
    pg_query = "select answer from question_response where question_id = 2031 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def emp_involvement(id_, cur):
    pg_query = "select answer from question_response where question_id = 2021 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def emp_performance(id_, cur):
    pg_query = "select answer from question_response where question_id = 2020 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def osha_citation(id_, cur):
    pg_query = "select answer from question_response where question_id = 1962 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result


def fatality_2019(id_, cur):
    pg_query = "select answer from question_response where question_id = 1888 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def restricted_2019(id_, cur):
    pg_query = "select answer from question_response where question_id = 1887 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def lost_days_2019(id_, cur):
    pg_query = "select answer from question_response where question_id = 1885 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def recordable_2019(id_, cur):
    pg_query = "select answer from question_response where question_id = 1880 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def dart_17(id_, cur):
    pg_query = "select answer from question_response where question_id = 1832 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def dart_18(id_, cur):
    pg_query = "select answer from question_response where question_id = 1823 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def dart_19(id_, cur):
    pg_query = "select answer from question_response where question_id = 1881 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def da_cases_17(id_, cur):
    pg_query = "select answer from question_response where question_id = 1838 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def da_cases_18(id_, cur):
    pg_query = "select answer from question_response where question_id = 1829 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result

def da_cases_19(id_, cur):
    pg_query = "select answer from question_response where question_id = 1884 and contractor_id = "+ str(id_) +";"
    cur.execute(pg_query)
    result = cur.fetchall()
    return result
