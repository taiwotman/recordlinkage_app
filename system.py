import streamlit as st
# from email_service import send_msg
import re

import pandas as pd
import logging
import sys
import os
import ray

import time

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fsevents').setLevel(logging.WARNING)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training_script.embedding_creation import (
    SentenceBertModel,
    preprocessing,
    get_similar_records,
    save_similarity_table,
)
from vanna_calls import (
    generate_sql_cached,
    run_sql_cached,
    validate_sql_cached,
    is_read_only_sql,
)
from constants import SIM_TABLE_PATH

'''
Utility Functions
'''

# Initialize session state if it doesn't exist
if 'users_db' not in st.session_state:
    st.session_state.users_db = pd.DataFrame(columns=["email", "first_name", "last_name", "verificationcode"])

if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if 'user_email' not in st.session_state: st.session_state.user_email = ""

verification_code_sent = "12345"  # For test purpose onlye. Later use send_msg(email=email, msg=verification_code, subject="Recordlinkage Verify Code for Login")


# Function to get the DataFrame from session state
def get_users_db():
    return st.session_state.users_db

# Function to save the DataFrame to session state
def save_users_db(df):
    st.session_state.users_db = df


def is_valid_email(email):
    # Regex pattern for validating an email address
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(pattern, email):
        return True
    else:
        return False



'''
Protected landing home page
'''
def home():
    st.write(f"Welcome, {st.session_state.user_email}!")
    

    # Initialize session state for progress_stage
    if "progress_stage" not in st.session_state:
        st.session_state["progress_stage"] = 1

    def increment_stage():
        st.session_state["progress_stage"] += 1


    # Title
    # st.title("Record Linkage ChatBot")

    # Centered progress slider (bound to session_state)
    st.markdown(
        "<style>.block-container {text-align: center;}</style>", unsafe_allow_html=True
    )
    st.session_state["progress_stage"] = st.slider(
        "Progress through stages:",
        1,
        3,
        st.session_state["progress_stage"],
        format="%d: Stage",
        label_visibility="visible",
    )

    # Dynamic side menu
    st.sidebar.title("Infometric")
    if st.session_state["progress_stage"] == 1:
        st.sidebar.success("Stage 1: File Upload")
        st.sidebar.warning("Stage 2: Record Linkage (Pending)")
        st.sidebar.warning("Stage 3: Search Entity (Pending)")
    elif st.session_state["progress_stage"] == 2:
        st.sidebar.success("Stage 1: File Upload (Completed)")
        st.sidebar.success("Stage 2: Record Linkage")
        st.sidebar.warning("Stage 3: Search Entity (Pending)")
    elif st.session_state["progress_stage"] == 3:
        st.sidebar.success("Stage 1: File Upload (Completed)")
        st.sidebar.success("Stage 2: Record Linkage (Completed)")
        st.sidebar.success("Stage 3: Search Entity")

    # Stage logic
    if st.session_state["progress_stage"] == 1:
        st.header("Stage 1: Upload File")
        uploaded_file = st.file_uploader("Upload a CSV file for processing", type=["csv"])
        if uploaded_file is not None:
            st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data")
            st.dataframe(st.session_state["uploaded_df"])
            st.success("File uploaded successfully.")
            if st.button("Next: Create Embeddings", on_click=increment_stage):
                pass
            

    elif st.session_state["progress_stage"] == 2:
        st.header("Stage 2: Record Linkage")
        if "uploaded_df" not in st.session_state:
            st.warning("Please upload a file in Stage 1 first.")
        else:
            with st.spinner("Processing data and creating embeddings..."):
                cols = st.session_state["uploaded_df"].columns.tolist()
                preprocessed_df = preprocessing(st.session_state["uploaded_df"], cols)
                st.session_state["preprocessed_df"] = preprocessed_df
                records_list = preprocessed_df["record_text"].tolist()

                sbert_embedding = SentenceBertModel(records_list)
                embeddings = sbert_embedding.vectorize()

                threshold = 0.8
                # ray.init()
                similar_records_list = ray.get(
                    get_similar_records.remote(embeddings, records_list, threshold)
                )

                save_similarity_table(similar_records_list, preprocessed_df, SIM_TABLE_PATH)

                st.success(
                    "Record Linkage completed! You can now proceed to querying your similarity table."
                )
                if st.button("Next: Search Entity", on_click = increment_stage):
                    pass

    elif st.session_state["progress_stage"] == 3:
        st.header("Stage 3: Search Entity")
        if "preprocessed_df" not in st.session_state:
            st.warning("Please complete Record Linkage in Stage 2 first.")
        else:
            if "query_history" not in st.session_state:
                st.session_state["query_history"] = []
            st.info("Advance Search")
            with st.expander("Text to SQL"):
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
                    # col1 = st.columns(1)
                    # with col1:

                    sql = generate_sql_cached(question)
                    if sql:
                        logging.info(f"SQL Query for question '{question}': {sql}")
                        st.code(sql, language="sql")

                        if not is_read_only_sql(sql):
                            logging.error(f"Unathorized query detected: {sql}")
                            st.error(
                                "The generated SQL is not allowed.The expected query should be READ ONLY"
                            )

                        elif validate_sql_cached(sql):
                                        try:
                                            df = run_sql_cached(sql)
                                            if df is not None:
                                                if df.empty:
                                                    st.write("### Query Results (Empty)")
                                                    st.info(
                                                        "No records exist in the similarity table that satisfy the conditions. "
                                                        "Try refining your query by using different names, columns, or less restrictive conditions."
                                                    )
                                                else:
                                                    st.write("### Query Results")

                                                    sort_column = st.selectbox(
                                                        "Sort by Column:", df.columns, index=0
                                                    )
                                                    sort_order = st.radio(
                                                        "Order:", ["Ascending", "Descending"]
                                                    )
                                                    df_sorted = df.sort_values(
                                                        by=sort_column,
                                                        ascending=(sort_order == "Ascending"),
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
                                                st.error(
                                                    "An unexpected error occurred while fetching results."
                                                )
                                        except Exception as e:
                                            logging.error(f"Error executing SQL '{sql}': {e}")
                                            st.error("Failed to execute the SQL query.")
                        else:
                            logging.error(f"Invalid SQL: {sql}")
                            st.error("The generated SQL query is not valid.")
                    else:
                        st.error("Unable to generate SQL for the question.")

    
    if st.button("Logout"): 
        logout_user()



'''
Registration for login
'''
def register_user():
    # Apply custom CSS to set the width of the text input box
    st.markdown(
        """
        <style>
        .stTextInput {
            width: 500px;  /* Adjust the width as needed */
        }

        .stAlert {
            width: 500px;  /* Adjust the width as needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display an image
    st.image('./images/register.png', use_container_width="auto")
    email = st.text_input("Email")
    verification_code = None

    if email:
        if is_valid_email(email):
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")

                if last_name:
                    st.info(f"Verification code sent to {email}. Enter Code to complete registration")
                    #send_msg(email=email, msg=verification_code, subject="Recordlinkage Verify Code for Registration")
                    verification_code = st.text_input("Verification Code", type='password')
                    register_button = st.button("Register")

                    if register_button:
                            if verification_code ==str(verification_code_sent):
                                st.success(f"Verified and registered successfully. Proceed to login")
                                users_db = get_users_db() 
                                users_db.loc[len(users_db)] = [email, first_name, last_name, verification_code_sent] 
                                save_users_db(users_db)
                            else:
                                st.error("Verification code is incorrect")
        else:
                st.error("Invalid Email")
            
'''
User login page
'''
def login_user():
    # Apply custom CSS to set the width of the text input box
    st.markdown(
        """
         <style>
        .stTextInput {
            width: 500px;  /* Adjust the width as needed */
        }

        .stAlert{
            width: 500px;  /* Adjust the width as needed */
        }
       
        """,
        unsafe_allow_html=True,
    )

    # Display an image
    st.image('./images/login.png', use_container_width="auto")

    if not st.session_state.logged_in:
        email = st.text_input("Email")
        verification_code = None
        verify = None

        # Example usage
        if email:
            if is_valid_email(email):
                st.write("Verification code sent to email. Enter verification code.")
                verification_code = st.text_input("Verification Code", type='password')
                verify = st.button("Verify")
            else:
                st.error("Invalid email")

            
        if verify:
            if verification_code == str(verification_code_sent):
                # Simplified: Just check if the email exists in the database
                users_db = get_users_db() 
                if email in users_db["email"].values: 
                    st.success("Login successful.") 
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.query_params["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Hmmm...email not found. Please register.") 
            else: 
                    st.error("Invalid verification code") 
                    return False
    else: 
        st.success("You are already logged in.")
        home()


def logout_user():
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    # st.success("Logged out successfully.")
    st.query_params["logged_in"] = False
    # time.sleep(1.0)
    st.rerun()






def main():
    # Streamlit page configuration
    st.set_page_config(layout="wide")
    st.title("RecordLinkage SmartApp")

    try:

        if st.query_params.get("logged_in", ["False"])[0] == "True": 
            st.session_state.logged_in = True 
        if st.session_state.logged_in: 
                home() 
        else:
            menu = ["Login", "Register"] 
            choice = st.sidebar.selectbox("Menu", menu) 
            if choice == "Login": login_user() 
            elif choice == "Register": register_user()
    except AttributeError as AE:
        logout_user()






