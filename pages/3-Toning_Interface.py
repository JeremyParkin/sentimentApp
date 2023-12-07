import streamlit as st
import pandas as pd
import mig_functions as mig
import re
import json
import openai
from openai import OpenAI
client = OpenAI(api_key=st.secrets["key"])
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

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



# def split_text(text, limit=700):
#     """Split text into chunks, each with a maximum length of 'limit'."""
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     chunks = []
#     current_chunk = sentences[0]
#
#     for sentence in sentences[1:]:
#         if len(current_chunk) + len(sentence) <= limit:
#             current_chunk += " " + sentence
#         else:
#             chunks.append(current_chunk)
#             current_chunk = sentence
#     chunks.append(current_chunk)
#
#     return chunks


def split_text(text, limit=700, sentence_limit=350):
    """Split text into chunks, each with a maximum length of 'limit', further splitting long sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        # Split long sentences further
        while len(sentence) > sentence_limit:
            part, sentence = sentence[:sentence_limit], sentence[sentence_limit:]
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk = part

        if len(current_chunk) + len(sentence) <= limit:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks





# def translate_concurrently(chunks):
#     """Translate a list of text chunks concurrently."""
#     with ThreadPoolExecutor(max_workers=30) as executor:
#         # Submit translation tasks
#         futures = [executor.submit(GoogleTranslator(source='auto', target='en').translate, chunk) for chunk in chunks]
#
#         # Collect results as they complete
#         results = []
#         for future in as_completed(futures):
#             results.append(future.result())
#
#     return results



def translate_concurrently(chunks):
    """Translate a list of text chunks concurrently, preserving the order."""
    with ThreadPoolExecutor(max_workers=30) as executor:
        # Submit translation tasks with indices
        future_to_index = {executor.submit(GoogleTranslator(source='auto', target='en').translate, chunk): i for i, chunk in enumerate(chunks)}

        # Collect results in order of submission
        results = [None] * len(chunks)
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                # Handle exceptions, e.g., log them or use a placeholder
                results[index] = f"Error: {e}"

    return results




def translate(text):
    """Translate text to English in chunks if it's longer than 5000 characters."""
    chunks = split_text(text)
    translated_chunks = translate_concurrently(chunks)
    return " ".join(translated_chunks)




if not st.session_state.upload_step:
    st.error('Please upload a CSV before trying this step.')
elif not st.session_state.config_step:
    st.error('Please run the configuration step before trying this step.')
else:
    counter = st.session_state.counter
    unique_stories = st.session_state.unique_stories

    col1, col2 = st.columns([3, 1], gap='large')

    if counter < len(unique_stories):
        # Display the story
        URL = f"{unique_stories.iloc[counter]['URL']}"
        head = escape_markdown(f"{unique_stories.iloc[counter]['Headline']}")
        body = escape_markdown(f"{unique_stories.iloc[counter]['Snippet']}")
        count = unique_stories.iloc[counter]['Group Count']

        with col2:
            # Add a button for translation
            if st.button('Translate to English'):
                # Translate the headline and body
                translated_head = translate(head)
                translated_body = translate(body)

                # Update the dataframe with translated text
                unique_stories.at[counter, 'Translated Headline'] = translated_head
                unique_stories.at[counter, 'Translated Body'] = translated_body

                # Update the display with translated text
                head, body = translated_head, translated_body

        # Define your keywords
        keywords = original_list  # Add your keywords here

        # Use the function to highlight keywords in the headline and body
        highlighted_head = highlight_keywords(head, keywords)
        highlighted_body = highlight_keywords(body, keywords)


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
            st.info(f"Grouped Stories: {count}")

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

                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo-1106",
                            messages=[
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