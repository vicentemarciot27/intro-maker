import os
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from sqlalchemy import create_engine
import boto3
from sqlalchemy import inspect
import streamlit as st

os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["aws"]["access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws"]["secret_access_key"]
os.environ["AWS_REGION"] = st.secrets["aws"]["region"]

# Load environment variables from .env
load_dotenv()

AWS_REGION = "us-east-1"
S3_STAGING_DIR = "s3://walter-agent/operational/agent_query_results/"

CONNECTION_STRING = (
    f"awsathena+rest://@athena.{AWS_REGION}.amazonaws.com:443/"
    f"nekt_trusted"
    f"?s3_staging_dir={S3_STAGING_DIR}"
    f"&work_group=primary"
)

_db_instance = None
_engine_instance = None

def create_db():
    """
    Cria e retorna um banco de dados conectado ao Athena, 
    reutilizando conexões existentes
    
    Returns:
        SQLDatabase: Banco de dados configurado e engine para o Langchain
    """
    global _db_instance, _engine_instance
    
    if _db_instance is None or _engine_instance is None:
        # Criar engine do SQLAlchemy apenas se não existir
        _engine_instance = create_engine(
            CONNECTION_STRING,
            echo=True,
            connect_args={
                'catalog': 'AwsDataCatalog'
            }
        )
        
        # Criar conexão do SQLDatabase apenas se não existir
        _db_instance = SQLDatabase(_engine_instance)

    return _db_instance, _engine_instance

def get_tables_schema_glue(engine, glue_db):
    """
    Retorna a descrição e as colunas, garantindo que arrays e JSON sejam corretamente identificados.
    """
    glue_client = boto3.client('glue', region_name=AWS_REGION)
    inspector = inspect(engine)

    schema_list = []

    for table in inspector.get_table_names(schema=glue_db):
        # Busca info de colunas via SQLAlchemy
        columns = inspector.get_columns(table, schema=glue_db)

        # Busca metadata do Glue para identificar tipos corretos
        response = glue_client.get_table(DatabaseName=glue_db, Name=table)
        glue_table = response["Table"]
        table_description = glue_table.get("Description", "")

        # Mapeia tipos do Glue
        glue_columns = {
            col["Name"]: col["Type"]
            for col in glue_table["StorageDescriptor"]["Columns"]
        }

        # Monta string com os tipos corretos
        header = f"{table}, Desc: [{table_description}]"
        col_lines = []
        for col in columns:
            col_name = col.get("name")
            col_comment = col.get("comment") or ""

            # Obtém o tipo correto do Glue
            col_type = glue_columns.get(col_name, "UNKNOWN")

            # Mapeia para os tipos do Trino
            if col_type.startswith("array<"):
                col_type = f"ARRAY<{col_type[6:-1]}>"
            elif col_type.startswith("map<"):
                col_type = f"MAP<{col_type[4:-1]}>"
            elif col_type.startswith("struct<"):
                col_type = f"ROW({col_type[7:-1]})"

            col_lines.append(f"{col_name}, {col_type}, {col_comment}")

        table_schema = "\n".join([header] + col_lines)
        schema_list.append(table_schema)

    return schema_list

def create_db_sqlalchemy():
    """
    Cria e retorna um banco de dados conectado ao SQLAlchemy, 
    reutilizando conexões existentes
    """
    return create_engine(CONNECTION_STRING)
    

if __name__ == "__main__":
    # Cria o banco de dados e o engine (conforme sua função create_db)
    db, engine = create_db()
    
    # Obtém a lista de strings com as descrições das tabelas
    schema_info = get_tables_schema_glue(engine, "nekt_trusted")
    
    # Exibe o resultado
    for table_desc in schema_info:
        print("====================================")
        print(table_desc)

    # Algumas queries
    query = "SELECT * FROM nekt_trusted.latest_norte_events_snapshot LIMIT 1"
    result = db.run(query)
    print(result)
