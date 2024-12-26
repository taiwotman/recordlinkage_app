import os
import sqlite3
import streamlit as st
import pandas as pd
import logging
import re
from pandasql import sqldf
from vanna.remote import VannaDefault
from constants import SIM_TABLE_PATH, UPLOAD_DIR
from dotenv import load_dotenv

load_dotenv()


def log_similarity_table():
    try:
        conn = sqlite3.connect(UPLOAD_DIR + "similarity.db")
        df = pd.read_sql_query("SELECT * FROM similarity_table LIMIT 10", conn)
        logging.info(f"Sample data from similarity_table:\n{df}")
        conn.close()
    except Exception as e:
        logging.error(f"Error reading similarity_table: {e}")


logging.basicConfig(level=logging.DEBUG)


def setup_vanna():
    try:
        # Create a temporary SQLite database
        similarity_table = pd.read_parquet(SIM_TABLE_PATH)
        # empty_df = pd.DataFrame(similarity_table.columns)
        conn = sqlite3.connect(UPLOAD_DIR + "\similarity.db")

        # Store the DataFrame into the SQLite database
        similarity_table.to_sql(
            "similarity_table", conn, if_exists="replace", index=False
        )
        logging.info("SQLite database setup completed.")

        vn = VannaDefault(
            api_key=st.secrets["general"]["VANNA_API_KEY"], model="chinook"
        )
        vn.connect_to_sqlite(UPLOAD_DIR + "\similarity.db")
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
        generated_query, _, _ = vn.ask(
            question=question,
            allow_llm_to_see_data=True,
            print_results=False,
            visualize=False,
        )

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
    similarity_table = pd.read_parquet(SIM_TABLE_PATH)
    globals()["similarity_table"] = similarity_table
    result_df = sqldf(sql, globals())

    return result_df


def is_read_only_sql(sql: str) -> bool:
    normalized_sql = sql.strip().upper()
    prohibited_keywords = ["DELETE", "INSERT", "UPDATE", "DROP", "ALTER", "TRUNCATE"]

    if not normalized_sql.startswith("SELECT"):
        return False
    for keyword in prohibited_keywords:
        if re.search(rf"\b{keyword}\b", normalized_sql):
            return False

    return True
