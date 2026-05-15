import psycopg2
import yaml


def read_yaml_file(yaml_file: str):
    """ load yaml cofigurations """
    config = None
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
    except:
        raise FileNotFoundError('Couldnt load the file')
    return config


def get_db_conn(
        creds_file: str = None
):
    """ Get an authenticated psycopg db connection, given a credentials file"""
    if not creds_file:
        creds_file = ".config/credentials.yaml"

    creds = read_yaml_file(creds_file)
    connection = psycopg2.connect(
        user=creds["user"],
        password=creds["pass"],
        host=creds["host"],
        port=creds["port"],
        dbname=creds["db"]
    )
    return connection
