import mig_functions as mig
import pickle
import dill
import base64
import streamlit as st
import pandas as pd
import io
from datetime import datetime



# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment App",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

# Sidebar configuration
mig.standard_sidebar()

st.session_state.current_page = 'Save and Load'

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M")



def save_session_state():
    # Copy all session state except DataFrames for manual handling
    session_data = {key: value for key, value in st.session_state.items() if key not in ['df_traditional', 'unique_stories', 'full_dataset']}

    # Handle specific DataFrames
    for df_name in ['df_traditional', 'unique_stories', 'full_dataset']:
        if df_name in st.session_state:
            df = st.session_state[df_name]
            if not df.empty:
                session_data[df_name] = df  # Save DataFrame directly

    # Serialize the session state using dill
    serialized_data = dill.dumps(session_data)
    # st.write("Serialized data size:", len(serialized_data))  # Debugging output

    file_name = f"{st.session_state.get('client_name', 'Session')} - {dt_string}.pkl"
    st.download_button(label="Download Session File",
                       data=serialized_data,
                       file_name=file_name,
                       mime="application/octet-stream")


def load_session_state(uploaded_file):
    if uploaded_file is not None:
        # Deserialize session state using dill
        session_data = dill.loads(uploaded_file.read())
        # st.write("Loaded session data keys:", session_data.keys())  # Debugging output

        # Restore DataFrames manually
        for df_name in ['df_traditional', 'unique_stories', 'full_dataset']:
            if df_name in session_data:
                data = session_data[df_name]
                # Check for legacy CSV format
                if isinstance(data, str) and "\n" in data:
                    buffer = io.StringIO(data)
                    df = pd.read_csv(buffer)

                    # Convert 'Date' column to datetime if present
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

                    # Save restored DataFrame to session state
                    st.session_state[df_name] = df
                else:
                    # Directly restore DataFrame
                    st.session_state[df_name] = data

        # Restore other session state variables
        for key, value in session_data.items():
            if key not in ['df_traditional', 'unique_stories', 'full_dataset']:
                st.session_state[key] = value

        st.session_state.pickle_load = True
        st.success("Session state loaded successfully!")



# def save_session_state():
#     # Manually copy necessary items from session state
#     session_data = {key: value for key, value in st.session_state.items()}
#
#     # Convert DataFrames to CSV strings
#     for df_name in ['df_traditional', 'unique_stories', 'full_dataset']:
#         if df_name in session_data:
#             buffer = io.StringIO()
#             session_data[df_name].to_csv(buffer, index=False)
#             session_data[df_name] = buffer.getvalue()
#
#     # Serialize the session state
#     serialized_data = pickle.dumps(session_data)
#
#     # Encode the serialized data for downloading
#     b64 = base64.b64encode(serialized_data).decode()
#
#     # Generate a download link
#     href = f'<a href="data:file/pkl;base64,{b64}" download="{st.session_state.client_name} - {dt_string}.pkl">Download Session State</a>'
#     return href
#
#
#
# def load_session_state(uploaded_file):
#     if uploaded_file is not None:
#         # Read the uploaded file
#         session_data = uploaded_file.getvalue()
#
#         # Deserialize the session state
#         deserialized_data = pickle.loads(session_data)
#
#         # Convert CSV strings back to DataFrames
#         for df_name in ['df_traditional', 'unique_stories', 'full_dataset']:
#             if df_name in deserialized_data:
#                 buffer = io.StringIO(deserialized_data[df_name])
#                 deserialized_data[df_name] = pd.read_csv(buffer)
#
#         # Update the session state
#         st.session_state.update(deserialized_data)
#
#         st.success("Session state loaded successfully!")


st.title("Save & Load")

if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before SAVING.')

elif not st.session_state.config_step:
    st.error('Please run the configuration step before SAVING.')

# else:
#     st.divider()
#
#     st.header("Save")
#     st.info("**SAVE** your current data-processing session to a downloadable .pkl file")
#
#     # When this button is clicked, the save_session_state function will be executed
#     if st.button("Generate Session File to Download"):
#         # Generate the download link (or any other way you handle the saving)
#         href = save_session_state()
#
#         st.markdown(href, unsafe_allow_html=True)
#
#     st.divider()
#
# st.header("Load")
# st.info("**LOAD** a previously saved data-processing session from a downloaded .pkl file")
# uploaded_file = st.file_uploader("Upload Session State", type="pkl")
# if uploaded_file is not None:
#     load_session_state(uploaded_file)

else:
    st.info("**SAVE** your current data-processing session to a downloadable .pkl file")



    # When this button is clicked, the save_session_state function will be executed
    if st.button("Generate Session File to Download"):
        placeholder = st.empty()
        placeholder.info("Processing... please wait.")
        # Generate the download link
        href = save_session_state()

        # Show the download link
        if href:  # Only display if href is not None
            st.markdown(href, unsafe_allow_html=True)
        placeholder.empty()



    st.divider()

st.header("LOAD")
st.info("**LOAD** a previously saved data-processing session from a downloaded .pkl file")

uploaded_file = st.file_uploader("Restore a Previous Session", type="pkl", label_visibility="hidden")
if uploaded_file is not None:
    load_session_state(uploaded_file)