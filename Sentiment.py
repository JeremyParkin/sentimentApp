import streamlit as st
import pandas as pd
import json
import replicate
import io
import time
import openpyxl

# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

# action_rows = 5
total_time = 0.0  # Initialize the total time taken for all API calls
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
cost_per_sec = 0.001400

st.title("MIG Sentiment Tool")
st.subheader("Experimental")

named_entity = st.text_input("What brand, org, or person should the sentiment be based on?", max_chars=100,
                             key="named_entity")

if len(named_entity) < 1:
    named_entity = "the organization"

analysis_placeholder = "Eg. As an academic institution, news stories covering academic presentations are typically positive or neutral, even if the subject of the presentation is not."
analysis_placeholder2 = f"As a guideline, positive stories might focus on {named_entity}'s good qualities, values, and successes as demonstrated by it and its representatives.  Negative might focus on {named_entity}'s legal or financial difficulties, lawsuits or accusations against it or its representatives, safety issues, racism, abuse, harassment, fraud, or other types of misconduct."

analysis_note = st.text_area("Any special note on sentiment approach?", max_chars=600, key="analysis_notes",
                             help="This will be added as a note in the prompt for each story. Use it as is or feel free to, edit, delete, or replace it as needed.",
                             placeholder=analysis_placeholder, height=80)

c1, c2, c3 = st.columns(3)
with c1:
    sentiment_switch = st.radio("Sentiment Type", ['3-way', '5-way'], )
with c2:
    random_sample = st.radio('Randomize Sample?', ['No', 'Yes'])

with c3:
    action_rows = st.number_input('Limit rows for testing (max 100)', min_value=1, value=5, max_value=600)


if sentiment_switch == '5-way':
    sentiment_type = "VERY POSITIVE / MODERATELY POSITIVE / NEUTRAL / MODERATELY NEGATIVE / VERY NEGATIVE / UNAVAILABLE"
    sentiment_rubric = f"""
            \nVERY POSITIVE: A positive portrayal of {named_entity}, focusing on its merits, successes, or positive contributions. A positive headline or first sentence is a good clue of a VERY POSITIVE story.
            \nSOMEWHAT POSITIVE: A net positive view of {named_entity}, but with some minor reservations or criticisms, maintaining a supportive stance overall.
            \nNEUTRAL: A passing mention or objective perspective of {named_entity}, balancing both praise and critique without favoring either.
            \nSOMEWHAT NEGATIVE: A mildly negative depiction of {named_entity}, highlighting its shortcomings, but may acknowledge some positive elements.
            \nVERY NEGATIVE: A strongly critical portrayal of {named_entity}, emphasizing significant concerns or failings.
            \nUNAVAILABLE: {named_entity} is not mentioned, or there is not adequate information to choose one of the other options.
            """

else:
    sentiment_type = "POSITIVE / NEUTRAL / NEGATIVE / UNAVAILABLE"
    sentiment_rubric = f"""
            \nPOSITIVE: A net positive portrayal of {named_entity}, focusing on its merits and successes.
            \nNEUTRAL: A passing mention or objective perspective of {named_entity}, balancing both praise and critique without favoring either.
            \nNEGATIVE: A net critical portrayal of {named_entity}, emphasizing significant concerns or failings.
            \nUNAVAILABLE: {named_entity} is not mentioned, or there is not adequate information to choose one of the other options.
            """

model_call = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781"

estimated_cost = action_rows * cost_per_sec * 3
st.write(f"Estimated cost based on max rows: ${estimated_cost:.2f}")


def prompt_preview():
    prompt_text = f"""
                You are acting as an entity sentiment AI, indicating how a news story portrays {named_entity} based on these options: 
                \n{sentiment_rubric}
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

if submitted and (csv_file is None or named_entity == "the organization"):
    st.error('Missing required form inputs above.')


elif submitted:
    replicate_api = st.secrets['REPLICATE_API_TOKEN']
    df['Entity Sentiment'] = ""

    # Progress bar initialization
    progress = st.progress(0)
    number_of_rows = len(df)
    to_be_done = min(action_rows, number_of_rows)


    def generate_prompt(row, named_entity):
        prompt_text = f"""
            You are acting as an entity sentiment AI, indicating how a news story portrays {named_entity} based on these options: 
            \n{sentiment_rubric}
            \n{analysis_note} 

            Generate a JSON formatted response with the following fields:
                {{
                'sentiment': ' '
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

        story_prompt = generate_prompt(row, named_entity)

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
                sentiment = response_json['sentiment']
                df.at[i, 'Entity Sentiment'] = sentiment

                break  # If successful, break out of the retry loop


        except (TimeoutError, Exception) as e:
            retries += 1
            if retries == MAX_RETRIES:
                st.error(f"Error processing message {i + 1} after {MAX_RETRIES} retries: {str(e)}")

            else:
                time.sleep(RETRY_DELAY)  # Wait before retrying

        # Update progress bar
        progress.progress((i + 1) / to_be_done)

    required_cols = ['Headline', 'Snippet', 'Entity Sentiment']
    df_display = df.filter(required_cols)
    st.dataframe(df_display, hide_index=True)

    # Create a download link for the DataFrame as an Excel file
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()  # Use writer.close() instead of writer.save()
    output.seek(0)
    st.download_button(
        label="Download entity analysis as Excel file",
        data=output,
        file_name="entity_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )