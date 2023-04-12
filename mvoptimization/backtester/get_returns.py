import pandas as pd
import numpy as np
import pyathena
import boto3
import time
import re
import datetime as dt

s3 = boto3.resource('s3')
bucket = s3.Bucket('optimizer-api-stg')
now = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

def athena_to_s3(session1, params):
    client = session1.client('athena', region_name=params["region"])
    execution = athena_query(client, params)
    execution_id = execution['QueryExecutionId']
    state = 'RUNNING'

    while (state in ['RUNNING', 'QUEUED']):
        response = client.get_query_execution(QueryExecutionId = execution_id)

        if 'QueryExecution' in response and \
                'Status' in response['QueryExecution'] and \
                'State' in response['QueryExecution']['Status']:
            state = response['QueryExecution']['Status']['State']
            
            if state == 'FAILED':
                print('Task Failed')
                return False
            elif state == 'SUCCEEDED':
                print('SUCCESS')
                s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                filename = re.findall('.*\/(.*)', s3_path)[0]
                return filename
        time.sleep(1)
    
    return False
    
def athena_query(client, params):

    response = client.start_query_execution(
        QueryString=params["query"],
        QueryExecutionContext={
            'Database': params['database']
        },
        ResultConfiguration={
            'OutputLocation': 's3://' + params['bucket'] + '/' + params['path']
        }
    )
    return response
    
def fetch_athena_ret(param, ids, rm_date):
    

    params1 = {
        'region': 'us-east-1',
        'database': 'rm_{}_prod_glb_eq_usd'.format(rm_date),
        'bucket': 'quant-manan',
        'path': 'athena/output',
        'query':  'SELECT * FROM "rm_{}_prod_glb_eq_usd"."riskmodel__returns" where r_securityid in {}'.format(rm_date, ids)
    }

    params2 = {
        'region': 'us-east-1',
        'database': 'rm_{}_prod_glb_eq_usd'.format(rm_date),
        'bucket': 'quant-manan',
        'path': 'athena/output',
        'query':  'SELECT * FROM "rm_{}_prod_glb_eq_usd"."riskmodel_vw_companyshareclass" where shareclassid in {}'.format(rm_date, ids)
    }
    
    params3 = {
        'region': 'us-east-1',
        'database': 'rm_{}_prod_glb_eq_usd'.format(rm_date),
        'bucket': 'quant-manan',
        'path': 'athena/output',
        'query':  'SELECT * FROM "rm_{}_prod_glb_eq_usd"."riskmodel_vw_companyshareclass" where companyid in {}'.format(rm_date, ids)
    }
    
    session1 = boto3.Session()
    
    maps = {
        "shareclassid_fetch" : params3,
        "cid_fetch" : params2,
        "returns_fetch" : params1}
    
    s3_filename = athena_to_s3(session1, maps[param])
    path = 's3://quant-manan/athena/output/' + s3_filename
    data = pd.read_csv(path)

    return data
    
def fetch_returns(id_list, rm_date, rm_date2):
    
    ids = '(' + (", ".join( repr(e) for e in id_list )) + ')'

    
    if id_list[0][:2] == "0C":
        mapping = fetch_athena_ret("shareclassid_fetch", ids, rm_date)[["shareclassid","companyid"]]
    elif id_list[0][:2] == "0P":
        mapping = fetch_athena_ret("cid_fetch", ids, rm_date)[["shareclassid","companyid"]]
        id_list = mapping["companyid"].unique().tolist()
        ids = '(' + (", ".join( repr(e) for e in id_list )) + ')'
    else:
        print("Wrong ID Type")

    data = fetch_athena_ret("returns_fetch", ids, rm_date)

    timeindex = pd.read_parquet(f's3://quant-prod-riskmodel-data-monthly/{rm_date2}_equity_global/output/morn-123456-GlobalRiskModel_timeindex/_timeindex/')
    data = data.merge(timeindex[["MODELDATE", "CANONICALDATE", "TIMEINDEX"]], left_on = "timeindex", right_on = "TIMEINDEX")
    data.sort_values(by = "TIMEINDEX", inplace = True)
    data.rename(columns = {'MODELDATE' : 'MODELDATE_og', 'CANONICALDATE' : 'MODELDATE'}, inplace = True)
    data = data.merge(mapping, left_on = "r_securityid", right_on = "companyid")[["MODELDATE", 'MODELDATE_og', "TIMEINDEX", "r_securityid","shareclassid", "dailyreturn_converted"]]
    
    return data