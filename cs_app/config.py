#!/usr/bin/python

from configparser import ConfigParser

def config_psql(filename='psql_db.ini', section='postgresql'):
    #create a parser
    parser= ConfigParser()
    #readfile
    parser.read(filename)

    #get section, default(psql)
    db={}
    # check to see if section(psql) parser exists
    if parser.has_section(section):
        params=parser.items(section)
        for param in params:
            db[param[0]]=param[1]

    # return an error if param is called that is NOT listed in (init) file
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db

