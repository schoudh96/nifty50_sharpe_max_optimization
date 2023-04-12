import boto3
import json
import pyarrow.feather as feather
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
import pickle
import s3fs
import stringcase
import mstarlogging as logging
import os
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
import pdb

logger = logging.get_logger(__name__)

HIVE_TO_PYARROW_MAP = {
    "boolean": pa.bool_(),
    "date": pa.date64(),
    "double": pa.float64(),
    "int": pa.int64(),
    "string": pa.string(),
    "timestamp": pa.timestamp('s')
}


def split_s3_bucket_key(s3_url):
    split_url = s3_url.split('/')
    bucket = split_url[2]
    key = '/'.join(split_url[3:])
    return bucket, key


def get_s3_data(s3_url):
    bucket, key = split_s3_bucket_key(s3_url)
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, key)
    body = obj.get()['Body'].read()
    return json.loads(body)


# def get_aws_client(service_name, resource_params={}, session_params=None):
#     if not session_params:
#         session_params = {'profile_name': 'default'}
#     session = boto3.session.Session(**session_params)
#     return session.client(service_name, **resource_params)
def get_aws_client(service_name, resource_params={}, session_params=None):
    """
    :param service_name: name of service for e.g. s3 or ec2
    :param resource_params: additional parameters used to create service for e.g. region_name
    :param session_params: parameters to pass into the boto3 session function, e.g. to specify a profile_name or an access key
        If specified, session_params will be used regardless of whether the environment is in AWS or not.
    :return:
    """
    env = os.environ.get('ENV', None)
    if (env == 'AWS') and (not session_params):
        return boto3.client(service_name, **resource_params)
    else:
        if not session_params:
            session_params = {'profile_name': 'saml'}
        session = boto3.session.Session(**session_params)
        return session.client(service_name, **resource_params)

def get_s3_file_system(session_params=None, **kwargs):
    if not session_params:
        file_system = s3fs.S3FileSystem(**kwargs)
    else:
        session = boto3.session.Session(**session_params)
        credentials = session.get_credentials()
        file_system = s3fs.S3FileSystem(key=credentials.access_key,
                                        secret=credentials.secret_key,
                                        token=credentials.token,
                                        client_kwargs={'region_name': session.region_name},
                                        **kwargs)
    file_system.read_timeout = 3600
    file_system.retries = 2
    file_system.connect_timeout = 10
    file_system.default_block_size = 1024 ** 3
    return file_system


def get_s3_file(s3_filename, file_type = 'pickle'):
    
    filetype2func = {
        'parquet' : pd.read_parquet,
        'csv' : pd.read_csv, 
        'pickle' : pickle.load
    }

    try:
        logger.info(f"Loading {s3_filename} from s3")
        s3_file_system = get_s3_file_system()
        with s3_file_system.open(s3_filename, 'rb') as file:
            f = filetype2func[file_type](file)
        # f = s3_file_system.open(s3_filename, 'rb').read()
        return f
    except:
        logger.info(f"Failed to load {s3_filename} from s3")
        raise


def query_athena(query, as_date=None):
    """ Executes the athena query and returns the results in a pandas DataFrame.
    :param query: SQL query to run on Athena
    :param as_date: list of column names to be treated as date columns (default = None)
    :returns: Pandas DataFrame containing the results of the athena query

    """
    logger.info(f"Executing query: {query}")
#     print("Printing query")
#     print(query)
#     print("\n")
    cursor = connect(s3_staging_dir=config.aws['s3']['s3_staging_dir'],
                     region_name=config.aws['region_name'], cursor_class=PandasCursor).cursor()
    df = cursor.execute(query).as_pandas()
    return df

def query_athena2(query, s3_staging_dir, region_name):
    """ Executes the athena query and returns the results in a pandas DataFrame.
    :param query: SQL query to run on Athena
    :param as_date: list of column names to be treated as date columns (default = None)
    :returns: Pandas DataFrame containing the results of the athena query

    """
    logger.info(f"Executing query: {query}")

    cursor = connect(s3_staging_dir=s3_staging_dir,
                     region_name=region_name, cursor_class=PandasCursor).cursor()
    df = cursor.execute(query).as_pandas()
    return df

def pandas_to_s3_feather(df, df_name):
    base_s3_path = f"{config.aws['s3']['output_bucket']}/{df_name}"
    s3_file_system = get_s3_file_system()
    with s3_file_system.open(base_s3_path, 'wb') as f:
        feather.write_feather(df, f)


def s3_feather_to_pandas(table_name):
    base_s3_path = f"{config.aws['s3']['output_bucket']}/{table_name}"
    s3_file_system = get_s3_file_system()
    with s3_file_system.open(base_s3_path, 'wb') as f:
        read_df = feather.read_feather(f)
    return read_df

def df_to_s3(obj, path_prefix, object_name, file_type):
    """
    Save object to s3 path as pickle, csv or parquet.
    :param obj: DataFrame to store in s3
    :param path_prefix: dir in s3 file path to store object in 
    :param object_name: file_name of the saved file in s3
    :param file_type: ['parquet'/'csv'] type file to save as
    :return: None, prints s3 object path.
    """
    file_type_map = {
        'parquet' : '.parquet',
        'csv' : '.csv'
    }

    base_s3_path = f"{config.aws['s3']['output_bucket']}/{path_prefix}/{object_name}" + file_type_map[file_type]
    # s3_file_system = get_s3_file_system()
    print(base_s3_path)
    if file_type == 'parquet':
        obj.to_parquet(base_s3_path, index = False)
    elif file_type == 'csv':
        obj.to_csv(base_s3_path, index = False)
    else:
        raise Exception("file_type parameter can only have values:'parquet' or 'csv'.")

def object_to_s3(obj, path_prefix, object_name):
    """
    Save object to s3 path as pickle, csv or parquet.
    :param obj: obj to store in s3
    :param path_prefix: dir in s3 file path to store object in 
    :param object_name: file_name of the saved file in s3
    :return: None, prints s3 object path.
    """

    base_s3_path = f"{config.aws['s3']['output_bucket']}/{path_prefix}/{object_name}.p"
    s3_file_system = get_s3_file_system()
    print(base_s3_path)
    with s3_file_system.open(base_s3_path, 'wb') as f:
        pickle.dump(obj, f)
    
def build_s3_paths(path, partition_values, depth=0):
    """
    Recursively build s3 paths using partitions
    for e.g. ['mstar-quant-dev-fund-rating-us-east-1/mqr_fund_id_level/end_date=2019-09-30/fund_id=FSUSA08RYK']
    :param path: Base s3 path
    :param partition_values: list of named pd.Series objects containing unique values to create partitions.
        for e.g.    [0  2019-09-30  1   2019-10-30  Name: end_date, dtype: object,
                    0   FSUSA08RYK  1   FSUSA07I42  2   FSUSA07I3R  Name: fund_id, dtype: object]
    :param depth: used during recursion to track traversal of partition_values
    :return: list of s3_paths. If partition_values = [], returns [path]
    """
    output = []

    if len(partition_values) == depth:
        output.append(path)
        return output

    partition_column_name = partition_values[depth].name
    for partition_value in partition_values[depth]:
        output += build_s3_paths(f"{path}/{partition_column_name}={partition_value}", partition_values, depth + 1)

    return output


def hive_type_to_pyarrow_type(data_type):
    """
    Convert a hive data type to an equivalent pyarrow data type. For primitive data types, it simply looks up the data_type in
    HIVE_TO_PYARROW_MAP.
    :param data_type: a data type in string format.
    :return: a pyarrow data type (not a string, but the actual DataType). Raises exception for a data type that can not be converted.
    """
    try:
        return HIVE_TO_PYARROW_MAP[data_type]
    except KeyError:
        logger.info('Unimplemented non-structural hive data type to pyarrow type conversion requested.')
        raise


def pandas_to_pyarrow(df, table_definition, name_attribute='name'):
    """
    Converts pandas df to pyarrow table.
    :param df: Pandas DataFrame
    :param table_definition: dict containing columns and partition keys (if any) along with its respective hive data types
    :param name_attribute: a string that denotes what attribute in the table_definition should be used as the name of the column.
        Options: ['name', 'data_lake_name']
    :return: pyarrow table. Raises exceptions if can't convert df to pyarrow table
    """

    try:
        # Define pyarrow schema
        fields = []
        for column_dict in table_definition['columns']:
            if column_dict.get(name_attribute):
                fields.append(pa.field(column_dict.get(name_attribute), hive_type_to_pyarrow_type(column_dict['type'])))

        for partition_col_dict in table_definition['partition_keys']:
            if partition_col_dict.get(name_attribute):
                fields.append(pa.field(partition_col_dict.get(name_attribute),
                                       hive_type_to_pyarrow_type(partition_col_dict['type'])))

        schema = pa.schema(fields)
        return pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    except TypeError as e:
        logger.info(e)
        raise

    except pa.lib.ArrowInvalid as e:
        logger.info(e)
        raise


def pandas_to_s3(athena_client, df, table_name, overwrite=True):
    """
    Write Pandas DataFrame to s3 location in parquet format. The function also performs data validation using the Great Expectations
    library.

    :param athena_client: AWS client object representing athena
    :param df: Pandas DataFrame
    :param table_name: name of table
    :param expectation_suite_base_path: a string path that represents the path to the directory in the repository
        where all "Great Expectations" expectation suites are stored.
    :param overwrite: boolean determining whether to delete any existing files in S3 based on partition key defined in the glue table
        and the unique partition values in the df. If the table is not partitioned then the entire table is overwritten.
    :return: None. Raises Exception if table is not defined in glue_table_definition.json or a critical data quality expectation fails.
    """
    df = df.copy()
    # convert column names to snake case to align with Athena
    df.columns = [stringcase.snakecase(col) for col in df.columns]
    try:
        table_definition = next(item for item in config.glue_table_definitions if item["name"] == table_name)
    except StopIteration:
        raise Exception(
            f'The table definition for {table_name} was not found in data/glue_table_definitions.json file.')

    # TODO: Add great expectation validation here

    partition_columns = [item['name'] for item in table_definition['partition_keys']]
    base_s3_path = f"{config.aws['s3']['output_bucket']}/{table_name}"
    table = pandas_to_pyarrow(df, table_definition)
    file_system = get_s3_file_system()
    if overwrite:
        partition_values = [df[col].drop_duplicates().astype('str') for col in partition_columns]
        all_s3_paths = build_s3_paths(base_s3_path, partition_values)
        # delete s3 paths that exist
        existing_s3_paths = sum([file_system.ls(path) for path in all_s3_paths if file_system.exists(path)], [])
        if existing_s3_paths:
            logger.info(f'Deleting the following s3 paths: {existing_s3_paths}')
            file_system.rm(existing_s3_paths)

    pq.write_to_dataset(table=table, root_path=base_s3_path, partition_cols=partition_columns, filesystem=file_system,
                        compression='snappy')
    # register new partitions in data catalogue
    athena_client.start_query_execution(
        QueryString=f"MSCK REPAIR TABLE `{config.aws['athena']['databases']['local']}.{table_name}`",
        ResultConfiguration={'OutputLocation': config.aws['s3']['s3_staging_dir']},
        WorkGroup=config.aws['athena']['work_group'])
