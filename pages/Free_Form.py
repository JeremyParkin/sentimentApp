import streamlit as st
import pandas as pd
import json
import replicate
import io
import time

# Set Streamlit configuration
st.set_page_config(page_title="MIG Free Form Analysis Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

# action_rows = 5
total_time = 0.0  # Initialize the total time taken for all API calls
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
cost_per_sec = 0.001400

st.title("Free Form Analysis")
st.subheader("Experimental")



analysis_placeholder = """You are a news-tagging AI.  Select the best fit tag from the following list to categorize the news story below:
ACADEMIC: Story mentions scholarly happenings or achievements at DePaul University.
COMMUNITY: Story mentions community events or contributions by DePaul University.
SME: Story presents or quotes a DePaul professor as a subject matter expert.
MISSION: Story touches on DePaul's mission or values as a catholic university.
SPORTS: The story focuses on DePaul athletics.
OTHER: If none of the other tags apply.
"""

analysis_note = st.text_area("Write your analysis prompt here:", max_chars=1500, key="analysis_notes",
                             help="This will be added as a note in the prompt for each story. Use it as is or feel free to, edit, delete, or replace it as needed.",
                             placeholder=analysis_placeholder, height=200)

c2, c3 = st.columns(2)

with c2:
    random_sample = st.radio('Randomize Sample?', ['No', 'Yes'])

with c3:
    action_rows = st.number_input('Limit rows for testing (max 100)', min_value=1, value=5, max_value=600)


model_call = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781"

estimated_cost = action_rows * cost_per_sec * 3
st.write(f"Estimated cost based on max rows: ${estimated_cost:.2f}")


def prompt_preview():
    prompt_text = f"""
                \n{analysis_note}                
                \nHere is the news story...
            """
    return prompt_text


with st.sidebar:
    st.header("Prompt Preview:")
    st.divider()
    st.write(prompt_preview())
    st.divider()

with st.form('User Inputs'):
    csv_file = st.file_uploader("Upload a CSV or cleaned XLSX file:", type=["csv", "xlsx"])


    if csv_file:
        try:
            if csv_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(csv_file, nrows=action_rows)
            else:
                df = pd.read_csv(csv_file, nrows=action_rows)

            if random_sample == 'Yes':
                df = df.sample(frac=1).reset_index(drop=True)

            df = df.rename(columns={'Published Date': 'Date'}, errors='ignore')
            df = df.rename(columns={'Coverage Snippet': 'Snippet'}, errors='ignore')


        except Exception as e:
            st.error("Error reading the file: {}".format(str(e)))
            st.stop()

    submitted = st.form_submit_button("Submit")

if submitted and (csv_file is None):
    st.error('Missing required form inputs above.')


elif submitted:
    replicate_api = st.secrets['REPLICATE_API_TOKEN']
    df['Freeform Analysis'] = ""

    # Progress bar initialization
    progress = st.progress(0)
    number_of_rows = len(df)
    to_be_done = min(action_rows, number_of_rows)


    def generate_prompt(row):
        prompt_text = f"""
            \n{analysis_note} 

            Generate a JSON formatted response with the following fields:
                {{
                'label': ' '
                }}

            NOTES:
            -Quote marks in JSON must be escaped with a backslash
            -DO NOT include any text beyond the JSON formatted sentiment indication.

            This is the news story:

            {row['Headline']}. {row['Snippet']}
            """
        return prompt_text


    for i, row in df.iterrows():
        if len(row['Snippet']) < 350:
            st.warning(f"Snippet is too short for story {i + 1}")
            progress.progress((i + 1) / to_be_done)
            continue
        if len(row['Snippet']) > 9250:
            st.warning(f"Snippet is too long for story {i + 1}")
            progress.progress((i + 1) / to_be_done)
            continue

        story_prompt = generate_prompt(row)

        try:
            start_time = time.time()  # Record the start time

            retries = 0
            while retries < MAX_RETRIES:
                output = replicate.run(
                    model_call,
                    input={
                        "prompt": story_prompt,
                        "system_prompt": "You produce only JSON-formatted responses, NOTHING MORE"
                    }
                )

                # Concatenate the pieces of the streamed response
                full_response = ''.join(output)

                # Extract the JSON portion of the full response
                start_index = full_response.find('{')
                end_index = full_response.rfind('}') + 1
                json_response = full_response[start_index:end_index]

                # st.write(json_response)
                response_json = json.loads(json_response)
                label = response_json['label']
                df.at[i, 'Freeform Analysis'] = label

                break  # If successful, break out of the retry loop


        except (TimeoutError, Exception) as e:
            retries += 1
            if retries == MAX_RETRIES:
                st.error(f"Error processing message {i + 1} after {MAX_RETRIES} retries: {str(e)}")

            else:
                time.sleep(RETRY_DELAY)  # Wait before retrying

        # Update progress bar
        progress.progress((i + 1) / to_be_done)

    required_cols = ['Headline', 'Snippet', 'Freeform Analysis']
    df_display = df.filter(required_cols)
    st.dataframe(df_display, hide_index=True)

    # Create a download link for the DataFrame as an Excel file
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()  # Use writer.close() instead of writer.save()
    output.seek(0)
    st.download_button(
        label="Download analysis as Excel file",
        data=output,
        file_name="freeform_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )