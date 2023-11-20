import mig_functions as mig
import pickle
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

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M")



def save_session_state():
    # Manually copy necessary items from session state
    session_data = {key: value for key, value in st.session_state.items()}

    # Convert DataFrames to CSV strings
    for df_name in ['df_traditional', 'unique_stories', 'full_dataset']:
        if df_name in session_data:
            buffer = io.StringIO()
            session_data[df_name].to_csv(buffer, index=False)
            session_data[df_name] = buffer.getvalue()

    # Serialize the session state
    serialized_data = pickle.dumps(session_data)

    # Encode the serialized data for downloading
    b64 = base64.b64encode(serialized_data).decode()

    # Generate a download link
    href = f'<a href="data:file/pkl;base64,{b64}" download="{st.session_state.client_name} - {dt_string}.pkl">Download Session State</a>'
    return href



def load_session_state(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded file
        session_data = uploaded_file.getvalue()

        # Deserialize the session state
        deserialized_data = pickle.loads(session_data)

        # Convert CSV strings back to DataFrames
        for df_name in ['df_traditional', 'unique_stories', 'full_dataset']:
            if df_name in deserialized_data:
                buffer = io.StringIO(deserialized_data[df_name])
                deserialized_data[df_name] = pd.read_csv(buffer)

        # Update the session state
        st.session_state.update(deserialized_data)

        st.success("Session state loaded successfully!")


st.title("Save & Load")

# st.title("Save & Load")
# #
# #
st.header("Save")
st.markdown(save_session_state(), unsafe_allow_html=True)
# data = save_session_state()
# st.download_button("Save & Download Your Session", data, type="primary")




# st.write("")
# st.write("")
# st.write("")
st.header("Load")
uploaded_file = st.file_uploader("Upload Session State", type="pkl")
if uploaded_file is not None:
    load_session_state(uploaded_file)

