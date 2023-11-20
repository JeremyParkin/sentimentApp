import streamlit as st
import pandas as pd
import mig_functions as mig


# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment App",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

# Sidebar configuration
mig.standard_sidebar()

st.title("Getting Started")

st.info("NOTE: It is strongly recommended to clear out junky mentions from your CSV before uploading.")

# Initialize Session State Variables
string_vars = {'page': '1: Getting Started', 'sentiment_type': '3-way', 'client_name': '', 'focus': '',
               'model_choice': 'GPT-3.5', 'similarity_threshold': 0.95, 'counter': 0, 'analysis_note': '', 'group_ids':'',
               'sample_size': 0, 'highlight_keyword':''}
for key, value in string_vars.items():
    if key not in st.session_state:
        st.session_state[key] = value

df_vars = ['df_traditional', 'unique_stories', 'full_dataset']
for _ in df_vars:
    if _ not in st.session_state:
        st.session_state[_] = pd.DataFrame()

bool_vars = ['upload_step', 'config_step', 'sentiment_opinion', 'random_sample']
for _ in bool_vars:
    if _ not in st.session_state:
        st.session_state[_] = False



if st.session_state.upload_step:
    st.success('File uploaded.')
    st.dataframe(st.session_state.df_traditional)

    if st.button('Start Over?'):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()


if not st.session_state.upload_step:

    with st.form('User Inputs'):
        client = st.text_input('Client organization name*', placeholder='eg. Air Canada', key='client',
                               help='Required to build export file name.')
        focus = st.text_input('Reporting period or focus*', placeholder='eg. March 2022', key='period',
                               help='Required to build export file name.')
        uploaded_file = st.file_uploader(label='Upload your CSV*', type='csv',
                                         accept_multiple_files=False,
                                         help='Only use CSV files exported from the Agility Platform.')

        submitted = st.form_submit_button("Submit", type="primary")


    if submitted and (client == "" or focus == "" or uploaded_file is None):
        st.error('Missing required form inputs above.')

    elif submitted:
        with st.spinner("Converting file format."):
            st.session_state.df_traditional = pd.read_csv(uploaded_file)
            st.session_state.df_traditional = st.session_state.df_traditional.dropna(thresh=3)
            # st.session_state.df_traditional["Mentions"] = 1
            if 'Assigned Sentiment' not in st.session_state.df_traditional.columns:
                st.session_state.df_traditional['Assigned Sentiment'] = pd.NA  # Initialize with NA values

            st.session_state.client_name = client
            st.session_state.focus = focus

            st.dataframe(st.session_state.df_traditional)

            st.session_state.df_traditional.rename(columns={"Coverage Snippet": "Snippet"}, inplace=True)
            st.session_state.full_dataset = st.session_state.df_traditional.copy()

            st.session_state.upload_step = True
            st.rerun()