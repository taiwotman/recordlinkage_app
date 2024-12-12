import streamlit as st
import logging
from vanna_calls import generate_sql_cached, run_sql_cached, validate_sql_cached
import pandas as pd

st.set_page_config(layout="wide")
st.sidebar.title("Output Settings")
show_sql = st.sidebar.checkbox("Show SQL", value=True)  
show_table = st.sidebar.checkbox("Show Table", value=True)  

st.title("Record Linkage ChatBot")
if "query_history" not in st.session_state:
    st.session_state["query_history"] = []

question = st.text_input("Ask questions about the similarity table:")
st.sidebar.title("Query History")
if st.session_state["query_history"]:
    selected_query = st.sidebar.selectbox(
        "Select a previous query to rerun:", st.session_state["query_history"]
    )
    if st.sidebar.button("Rerun Selected Query"):
        question = selected_query

if question:
    
    if question not in st.session_state["query_history"]:
        st.session_state["query_history"].append(question)

    st.write("### Current Query")
    st.write(question)

    sql = generate_sql_cached(question)
    if sql:
        logging.info(f"SQL Query for question '{question}': {sql}")
        if show_sql:
            st.code(sql, language="sql")

        if validate_sql_cached(sql):
            try:
                df = run_sql_cached(sql)
                if df is not None:
                    if df.empty:
                        st.warning("No results found for your query.")
                        st.write("### Query Results (Empty)")
                        st.dataframe(df)
                        st.write("### AI Explainability")
                        st.info(
                            "No records exist in the similarity table that satisfy the conditions. "
                            "Try refining your query by using different names, columns, or less restrictive conditions."
                        )
                    else:
                        if show_table:
                            st.write("### Query Results")

                            sort_column = st.selectbox("Sort by Column:", df.columns, index=0)
                            sort_order = st.radio("Order:", ["Ascending", "Descending"])
                            df_sorted = df.sort_values(
                                by=sort_column, ascending=(sort_order == "Ascending")
                            )

                            st.dataframe(df_sorted)
                            st.write("### Download Options")
                            csv = df_sorted.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv",
                            )

                            # AI Explainability
                            st.write("### AI Explainability")
                            st.info(
                                f"The table results were sorted by '{sort_column}' in "
                                f"{'ascending' if sort_order == 'Ascending' else 'descending'} order."
                            )
                else:
                    logging.error(f"Failed to fetch results for SQL: {sql}")
                    st.error("An unexpected error occurred while fetching results.")
            except Exception as e:
                logging.error(f"Error executing SQL '{sql}': {e}")
                st.error("Failed to execute the SQL query.")
        else:
            logging.error(f"Invalid SQL: {sql}")
            st.error("The generated SQL query is not valid.")
    else:
        st.error("Unable to generate SQL for the question.")

