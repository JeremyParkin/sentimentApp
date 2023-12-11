import streamlit as st
import pandas as pd
import mig_functions as mig
import re
# import openai
from openai import OpenAI
client = OpenAI(api_key=st.secrets["key"])
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

# Initialize session state variables if not already set
if 'view_flagged_only' not in st.session_state:
    st.session_state.view_flagged_only = False

if 'filtered_counter' not in st.session_state:
    st.session_state.filtered_counter = 0



# Filter the DataFrame based on the review toggle state
if st.session_state.view_flagged_only:
    # Check if there are any flagged stories
    if (st.session_state.unique_stories['Flagged for Review'] == True).any():
        st.session_state.filtered_stories = st.session_state.unique_stories[
            st.session_state.unique_stories['Flagged for Review'] == True].copy()
    else:
        st.info('No stories flagged for review yet.  Please disable the "Review Flagged" toggle to see all stories.')
        if st.button('Back to full list'):
            st.session_state.view_flagged_only = False
            st.rerun()
        st.stop()

else:
    st.session_state.filtered_stories = st.session_state.unique_stories.copy()



# Sidebar configuration
mig.standard_sidebar()

st.session_state.current_page = 'Toning Interface'

original_list = st.session_state.highlight_keyword




# Function to escape markdown characters in a text part
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
    st.title("Toning Interface")
    st.error('Please upload a CSV before trying this step.')
elif not st.session_state.config_step:
    st.title("Toning Interface")
    st.error('Please run the configuration step before trying this step.')
else:
    # Add a toggle to the sidebar
    st.session_state.view_flagged_only = st.sidebar.toggle('Review Flagged')

    # Use the appropriate counter depending on the view
    if st.session_state.view_flagged_only:
        counter = st.session_state.filtered_counter
    else:
        counter = st.session_state.counter


    col1, col2 = st.columns([3, 1], gap='large')

    if counter < len(st.session_state.filtered_stories):
        # Display the story
        URL = f"{st.session_state.filtered_stories.iloc[counter]['URL']}"
        head = escape_markdown(f"{st.session_state.filtered_stories.iloc[counter]['Headline']}")
        body = escape_markdown(f"{st.session_state.filtered_stories.iloc[counter]['Snippet']}")
        count = st.session_state.filtered_stories.iloc[counter]['Group Count']

        with col2:
            button1, button2 = st.columns(2)
            with button1:
                # Add a button for translation
                if st.button('Translate'):
                    # Translate the headline and body
                    translated_head = translate(head)
                    translated_body = translate(body)

                    # Update the dataframe with translated text
                    st.session_state.filtered_stories.at[counter, 'Translated Headline'] = translated_head
                    st.session_state.filtered_stories.at[counter, 'Translated Body'] = translated_body

                    # Update the display with translated text
                    head, body = translated_head, translated_body

            with button2:
                current_group_id = st.session_state.filtered_stories.iloc[counter]['Group ID']

                # Add buttons to flag / unflag story for review
                flagged_status = st.session_state.df_traditional.loc[
                    st.session_state.df_traditional['Group ID'] == current_group_id, 'Flagged for Review'].iloc[0]

                if flagged_status == False:
                    if st.button('Flag'):
                        st.session_state.df_traditional.loc[st.session_state.df_traditional[
                                                                'Group ID'] == current_group_id, 'Flagged for Review'] = True
                        st.session_state.unique_stories.loc[st.session_state.unique_stories[
                                                                'Group ID'] == current_group_id, 'Flagged for Review'] = True
                        st.session_state.filtered_stories.loc[st.session_state.filtered_stories[
                                                                  'Group ID'] == current_group_id, 'Flagged for Review'] = True

                        st.rerun()

                if flagged_status == True:
                    if st.button('Unflag'):
                        st.session_state.df_traditional.loc[st.session_state.df_traditional[
                                                                'Group ID'] == current_group_id, 'Flagged for Review'] = False
                        st.session_state.unique_stories.loc[st.session_state.unique_stories[
                                                                'Group ID'] == current_group_id, 'Flagged for Review'] = False
                        st.session_state.filtered_stories.loc[st.session_state.filtered_stories[
                                                                  'Group ID'] == current_group_id, 'Flagged for Review'] = False

                        st.rerun()

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
            assigned_sentiment = st.session_state.df_traditional.loc[
                st.session_state.df_traditional['Group ID'] == current_group_id, 'Assigned Sentiment'].iloc[0]

            if pd.notna(assigned_sentiment):
                st.info(f"Assigned Sentiment: {assigned_sentiment}")

            with st.form("Sentiment Selector"):
                # Sentiment selection radio buttons
                if st.session_state.sentiment_type == '3-way':
                    sentiment_choice = st.radio("Sentiment Choice", ['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'NOT RELEVANT'], index=1,)
                else:
                    sentiment_choice = st.radio("Sentiment Choice", ['VERY POSITIVE', 'SOMEWHAT POSITIVE', 'NEUTRAL', 'SOMEWHAT NEGATIVE', 'VERY NEGATIVE', 'NOT RELEVANT'], index=2, )


                submitted = st.form_submit_button("Confirm Sentiment", type='primary', )



            # Display group count
            st.info(f"Grouped Stories: {count}")

            if submitted:

                # Update sentiment in DataFrame
                current_group_id = st.session_state.filtered_stories.iloc[counter]['Group ID']

                st.session_state.df_traditional.loc[st.session_state.df_traditional[
                                                        'Group ID'] == current_group_id, 'Assigned Sentiment'] = sentiment_choice


                # Go to the next article
                if st.session_state.view_flagged_only:
                    st.session_state.filtered_counter = min(len(st.session_state.filtered_stories) - 1,
                                                            st.session_state.filtered_counter + 1)
                else:
                    st.session_state.counter = min(len(st.session_state.filtered_stories) - 1,
                                                   st.session_state.counter + 1)

                st.rerun()



            # Navigation buttons
            prev_button, next_button = st.columns(2)

            # Update the appropriate counter when a button is clicked
            if st.session_state.view_flagged_only:
                with prev_button:
                    if st.button('Back', disabled=(st.session_state.filtered_counter == 0)):
                        st.session_state.filtered_counter = max(0, st.session_state.filtered_counter - 1)
                        st.rerun()

                with next_button:
                    if st.button('Next', disabled=(st.session_state.filtered_counter == len(st.session_state.filtered_stories) - 1)):
                        st.session_state.filtered_counter = min(len(st.session_state.filtered_stories) - 1,
                                                                st.session_state.filtered_counter + 1)
                        st.rerun()
            else:
                with prev_button:
                    if st.button('Back', disabled=(st.session_state.counter == 0)):
                        st.session_state.counter = max(0, st.session_state.counter - 1)
                        st.rerun()

                with next_button:
                    if st.button('Next', disabled=(st.session_state.counter == len(st.session_state.filtered_stories) - 1)):
                        st.session_state.counter = min(len(st.session_state.filtered_stories) - 1,
                                                       st.session_state.counter + 1)
                        st.rerun()


            numbers, progress = st.columns(2)
            with progress:
                assigned_articles_count = st.session_state.df_traditional['Assigned Sentiment'].notna().sum()
                percent_done = assigned_articles_count / len(st.session_state.df_traditional)
                st.metric("Total done", "{:.1%}".format(percent_done), "")

            with numbers:

                if st.session_state.view_flagged_only:
                    counter = st.session_state.filtered_counter
                    total_stories = (st.session_state.unique_stories['Flagged for Review'] == True).sum()
                else:
                    counter = st.session_state.counter
                    total_stories = len(st.session_state.unique_stories)
                st.metric("Unique story", f"{counter + 1}/{total_stories}", "")



        with col2:
            # Optional API call
            if st.session_state.sentiment_opinion:
                current_group_id = st.session_state.filtered_stories.iloc[counter]['Group ID']

                # Create a placeholder for the sentiment opinion
                sentiment_placeholder = st.empty()


                # Check if the response for this story is already stored or exists in the dataframe
                if pd.notna(st.session_state.filtered_stories.iloc[counter]['Sentiment Opinion']):
                    sentiment = st.session_state.filtered_stories.iloc[counter]['Sentiment Opinion']
                    sentiment_placeholder.write(sentiment)

                else:
                    try:
                        # Display a loading message
                        sentiment_placeholder.info('Generating sentiment opinion...')

                        story_prompt = f"\n{st.session_state.sentiment_instruction}\nThis is the news story:\n{head}\n{body}"

                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo-1106",
                            messages=[
                                {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                                {"role": "user", "content": story_prompt}])

                        sentiment = response.choices[0].message.content.strip()

                        # Update the sentiment opinion in both dataframes
                        st.session_state.filtered_stories.at[counter, 'Sentiment Opinion'] = sentiment
                        st.session_state.unique_stories.loc[
                            st.session_state.unique_stories[
                                'Group ID'] == current_group_id, 'Sentiment Opinion'] = sentiment
                        st.session_state.df_traditional.loc[
                            st.session_state.df_traditional[
                                'Group ID'] == current_group_id, 'Sentiment Opinion'] = sentiment

                        # Update the placeholder with the sentiment opinion
                        sentiment_placeholder.write(sentiment)

                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")


            if counter + 1 == total_stories:

                if st.button('Back to the first story'):
                    st.session_state.counter = 0
                    st.session_state.filtered_counter = 0
                    st.rerun()

    else:
        st.info("You have reached the end of the stories.")
        # st.rerun()

        if counter + 1 == total_stories:

            if st.button('Back to the first story'):
                st.session_state.counter = 0
                st.session_state.filtered_counter = 0
                st.rerun()


