import streamlit as st
import pandas as pd
import mig_functions as mig
import re
import json
import openai
from openai import OpenAI
client = OpenAI(api_key=st.secrets["key"])

# Sidebar configuration
mig.standard_sidebar()

original_list = st.session_state.highlight_keyword

# Initialize session state variables if not already set
if 'skipped_articles' not in st.session_state:
    st.session_state.skipped_articles = []


def escape_markdown(text):
    markdown_special_chars = r"\`*_{}[]()#+-.!$"

    # Regular expression pattern for URLs
    url_pattern = r'https?:\/\/[^\s]+'

    # Function to escape markdown characters in a text part
    def escape_chars(part):
        return re.sub(r"([{}])".format(re.escape(markdown_special_chars)), r"\\\1", part)

    # Function to process each part of the text
    def process_part(part):
        # If part is a URL, return it as is
        if re.match(url_pattern, part):
            return part
        # Otherwise, escape markdown characters
        return escape_chars(part)

    # Split text into parts (URLs and non-URLs) and process each part
    parts = re.split(r'(' + url_pattern + r')', text)
    escaped_text = ''.join(process_part(part) for part in parts)
    return escaped_text


def highlight_keywords(text, keywords, background_color="goldenrod", text_color="black"):
    # Create a regular expression pattern to match all keywords (case-insensitive)
    pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'

    # Function to replace each match with highlighted version
    def replace(match):
        return f"<span style='background-color: {background_color}; color: {text_color};'>{match.group(0)}</span>"

    # Use re.sub() to replace all occurrences in the text
    highlighted_text = re.sub(pattern, replace, text, flags=re.IGNORECASE)
    return highlighted_text


if not st.session_state.upload_step:
    st.error('Please upload a CSV before trying this step.')
elif not st.session_state.config_step:
    st.error('Please run the configuration step before trying this step.')
else:
    counter = st.session_state.counter
    unique_stories = st.session_state.unique_stories


    if counter < len(unique_stories):
        # Display the story
        URL = f"{unique_stories.iloc[counter]['URL']}"
        head = escape_markdown(f"{unique_stories.iloc[counter]['Headline']}")
        body = escape_markdown(f"{unique_stories.iloc[counter]['Snippet']}")
        count = unique_stories.iloc[counter]['Group Count']


        # Define your keywords
        keywords = original_list  # Add your keywords here

        # Use the function to highlight keywords in the headline and body
        highlighted_head = highlight_keywords(head, keywords)
        highlighted_body = highlight_keywords(body, keywords)


        col1, col2 = st.columns([3, 1], gap='large')
        with col1:
            st.markdown(URL)

            st.subheader(f"**{head}**")
            st.markdown(f"{highlighted_body}", unsafe_allow_html=True)


        # Handle sentiment selection and navigation
        with col2:
            # Check if sentiment has already been assigned
            current_group_id = unique_stories.iloc[counter]['Group ID']
            assigned_sentiment = st.session_state.df_traditional.loc[
                st.session_state.df_traditional['Group ID'] == current_group_id, 'Assigned Sentiment'].iloc[0]


            if pd.notna(assigned_sentiment):
                st.info(f"Assigned Sentiment: {assigned_sentiment}")

            with st.form("Sentiment Selector"):
                # Sentiment selection radio buttons
                if st.session_state.sentiment_type == '3-way':
                    sentiment_choice = st.radio("Sentiment Choice", ['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'NOT RELEVANT'])
                else:
                    sentiment_choice = st.radio("Sentiment Choice", ['VERY POSITIVE', 'SOMEWHAT POSITIVE', 'NEUTRAL', 'SOMEWHAT NEGATIVE', 'VERY NEGATIVE', 'NOT RELEVANT'])

                submitted = st.form_submit_button("Confirm Sentiment", type='primary')

            # Display group count
            st.info(f"Impacted Stories: {count}")

            if submitted:
                # Update sentiment in DataFrame
                current_group_id = unique_stories.iloc[counter]['Group ID']
                st.session_state.df_traditional.loc[st.session_state.df_traditional[
                                                        'Group ID'] == current_group_id, 'Assigned Sentiment'] = sentiment_choice

                # Find next unassigned article
                next_unassigned_idx = None
                for i in range(counter + 1, len(unique_stories)):
                    group_id = unique_stories.iloc[i]['Group ID']
                    if pd.isna(st.session_state.df_traditional.loc[
                                   st.session_state.df_traditional['Group ID'] == group_id, 'Assigned Sentiment'].iloc[
                                   0]):
                        next_unassigned_idx = i
                        break

                if next_unassigned_idx is not None:
                    st.session_state.counter = next_unassigned_idx
                else:
                    st.success("All articles have been assigned a sentiment.")

                st.rerun()


            # Navigation buttons
            prev_button, next_button = st.columns(2)
            with prev_button:
                if st.button('Back'):
                    st.session_state.counter = max(0, st.session_state.counter - 1)
                    st.rerun()


            with next_button:
                if st.button('Next'):
                    st.session_state.counter = min(len(unique_stories) - 1, st.session_state.counter + 1)
                    st.rerun()

            numbers, progress = st.columns(2)
            with progress:
                assigned_articles_count = st.session_state.df_traditional['Assigned Sentiment'].notna().sum()
                percent_done = assigned_articles_count / len(st.session_state.df_traditional)
                st.metric("Percent done", "{:.1%}".format(percent_done), "")

            with numbers:
                st.write("Story")
                st.write(f"{counter}/{len(unique_stories)}")


        with col2:
            # Optional API call
            if st.session_state.sentiment_opinion:
                current_group_id = unique_stories.iloc[counter]['Group ID']

                # Check if the response for this story is already stored or exists in the dataframe
                if pd.notna(unique_stories.iloc[counter]['Sentiment Opinion']):
                    sentiment = unique_stories.iloc[counter]['Sentiment Opinion']
                else:
                    try:
                        story_prompt = f"\n{st.session_state.sentiment_instruction}\nThis is the news story:\n{head}\n{body}"
                        response = client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=[
                            {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                            {"role": "user", "content": story_prompt}])
                        sentiment = response.choices[0].message.content.strip()

                        # Update the sentiment opinion in both dataframes
                        unique_stories.at[counter, 'Sentiment Opinion'] = sentiment
                        st.session_state.df_traditional.loc[
                            st.session_state.df_traditional[
                                'Group ID'] == current_group_id, 'Sentiment Opinion'] = sentiment
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")


                st.write(sentiment)