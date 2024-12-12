import sqlite3
import streamlit as st
import pandas as pd
from pandasql import sqldf
from vanna.remote import VannaDefault
import logging
import os

data_path = os.path.join('example_data', 'similarity_table.parquet')

similarity_table = pd.read_parquet(data_path)

def log_similarity_table():
    try:
        conn = sqlite3.connect("similarity.db")
        df = pd.read_sql_query("SELECT * FROM similarity_table LIMIT 10", conn)
        logging.info(f"Sample data from similarity_table:\n{df}")
        conn.close()
    except Exception as e:
        logging.error(f"Error reading similarity_table: {e}")

logging.basicConfig(level=logging.DEBUG)  

def setup_vanna():
    try:
        # Create a temporary SQLite database
        empty_df = pd.DataFrame(similarity_table.columns)
        conn = sqlite3.connect("similarity.db")
        
        # Store the DataFrame into the SQLite database
        empty_df.to_sql("similarity_table", conn, if_exists="replace", index=False)
        logging.info("SQLite database setup completed.")

        vn = VannaDefault(api_key=st.secrets["general"]["VANNA_API_KEY"], model='chinook')  
        vn.connect_to_sqlite("similarity.db")
        logging.info("VannaDefault object successfully initialized.")
        
        return vn
    except Exception as e:
        logging.error(f"Error in setup_vanna: {e}")
        return None


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    try:
        vn = setup_vanna() 
        
        if vn is None:
            logging.error("Vanna object is None, unable to generate SQL.")
            return None
        logging.info(f"Generating SQL query for the question: {question}")
        generated_query, _, _ = vn.ask(question=question,allow_llm_to_see_data=False, print_results=False, visualize=False)
        
        logging.info(f"Generated SQL query: {generated_query}")
        return generated_query
    
    except Exception as e:
        logging.error(f"Error generating SQL: {e}")
        return None



@st.cache_data(show_spinner="Checking SQL validity ...")
def validate_sql_cached(sql: str):
    try:
        vn = setup_vanna()
        is_valid = vn.is_sql_valid(sql=sql)
        logging.info(f"SQL Validity for '{sql}': {is_valid}")
        return is_valid
    except Exception as e:
        logging.error(f"Error validating SQL '{sql}': {e}")
        return False

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    result_df = sqldf(sql, globals())
    
    return result_df

