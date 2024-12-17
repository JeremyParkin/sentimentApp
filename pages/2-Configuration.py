import streamlit as st
import pandas as pd
import mig_functions as mig
import math
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from streamlit_tags import st_tags
import time


# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")

st.session_state.current_page = 'Configuration'

# Sidebar configuration
mig.standard_sidebar()


# Initialize st.session_state.elapsed_time if it does not exist
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0


def normalize_text(text):
    """Convert to lowercase, remove extra spaces, remove punctuation, etc."""
    # Convert to string in case the input is not a string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces from the beginning and end
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation (optional)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Further normalization steps like stemming could be added here
    return text


def remove_extra_spaces(text):
    """Remove extra spaces from the beginning and end of a string and replace multiple spaces with a single space."""
    # Convert to string in case the input is not a string
    text = str(text)
    # Remove extra spaces from the beginning and end
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text



def preprocess_online_news(df):
    """Pre-process ONLINE/ONLINE_NEWS articles by grouping by Date and Headline."""
    # Handle column name variations
    date_column = 'Date' if 'Date' in df.columns else 'Published Date'
    type_column = 'Media Type' if 'Media Type' in df.columns else 'Type'

    if date_column not in df.columns or 'Headline' not in df.columns:
        st.warning("Required columns for preprocessing (Date, Headline) are missing!")
        return df

    # Filter only ONLINE and ONLINE_NEWS articles
    online_df = df[df[type_column].isin(['ONLINE', 'ONLINE_NEWS'])].copy()

    # Ensure Date is in datetime format and extract year/month/day
    online_df[date_column] = pd.to_datetime(online_df[date_column], errors='coerce')
    online_df['Published Date'] = online_df[date_column].dt.strftime('%Y-%m-%d')

    # Group by Published Date and Headline
    grouped = online_df.groupby(['Published Date', 'Headline']).first().reset_index()

    # Merge grouped data back with non-online rows
    non_online_df = df[~df[type_column].isin(['ONLINE', 'ONLINE_NEWS'])]
    preprocessed_df = pd.concat([grouped, non_online_df], ignore_index=True)

    return preprocessed_df



# def cluster_similar_stories(df, similarity_threshold=0.85):
#     """Cluster similar stories using agglomerative clustering."""
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(df['Normalized Headline'] + " " + df['Normalized Snippet']).toarray()
#
#     # Compute cosine distances
#     cosine_distance_matrix = cosine_distances(tfidf_matrix)
#
#     # Use Agglomerative Clustering with a distance threshold
#     clustering = AgglomerativeClustering(
#         n_clusters=None,  # Let the algorithm decide the number of clusters
#         metric="precomputed",  # Use precomputed cosine distances
#         linkage="average",  # Average linkage for cosine distances
#         distance_threshold=1 - similarity_threshold  # Convert similarity to distance
#     )
#     cluster_labels = clustering.fit_predict(cosine_distance_matrix)
#
#     # Add cluster labels as 'Group ID'
#     df['Group ID'] = cluster_labels
#     return df

def cluster_similar_stories(df, similarity_threshold=0.92):
    """Cluster similar stories using agglomerative clustering."""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Normalized Headline'] + " " + df['Normalized Snippet']).toarray()

    # Ensure there are at least 2 samples for clustering
    if tfidf_matrix.shape[0] < 2:
        df['Group ID'] = 0  # Assign default group ID
        return df

    # Compute cosine distances
    cosine_distance_matrix = cosine_distances(tfidf_matrix)

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1 - similarity_threshold
    )
    cluster_labels = clustering.fit_predict(cosine_distance_matrix)

    # Add cluster labels as 'Group ID'
    df['Group ID'] = cluster_labels
    return df




def cluster_by_media_type(df, similarity_threshold=0.92):
    """Cluster stories by media type and ensure unique Group IDs across media types."""
    type_column = 'Media Type' if 'Media Type' in df.columns else 'Type'

    # Identify unique media types
    unique_media_types = df[type_column].unique()

    clustered_frames = []
    group_id_offset = 0  # Offset to ensure unique Group IDs across media types

    for media_type in unique_media_types:
        st.write(f"Processing media type: {media_type}")

        # Filter data for the current media type
        media_df = df[df[type_column] == media_type].copy()

        if not media_df.empty:
            # Fill missing Headline/Snippet with empty strings
            media_df['Headline'] = media_df['Headline'].fillna("")
            media_df['Snippet'] = media_df['Snippet'].fillna("")

            # Skip processing if all headlines and snippets are empty
            if media_df[['Headline', 'Snippet']].apply(lambda x: x.str.strip()).eq("").all(axis=None):
                st.warning(f"Skipping media type {media_type} due to missing headlines and snippets.")
                continue

            # Normalize and clean text
            media_df['Normalized Headline'] = media_df['Headline'].apply(normalize_text)
            media_df['Normalized Snippet'] = media_df['Snippet'].apply(normalize_text)

            # Cluster stories for this media type
            media_df = cluster_similar_stories(media_df, similarity_threshold=similarity_threshold)

            # Offset Group IDs to make them unique
            media_df['Group ID'] += group_id_offset
            group_id_offset += media_df['Group ID'].max() + 1

            # Drop normalized columns
            normalized_columns = [col for col in ['Normalized Headline', 'Normalized Snippet'] if
                                  col in media_df.columns]
            media_df = media_df.drop(columns=normalized_columns, errors='ignore')

            clustered_frames.append(media_df)

    # Combine all clustered frames
    return pd.concat(clustered_frames, ignore_index=True) if clustered_frames else df





def identify_duplicates(cluster_labels):  ## refactored for agg clustering
    """Group articles based on cluster labels."""
    from collections import defaultdict

    duplicates = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        duplicates[label].append(idx)

    return duplicates



def clean_snippet(snippet):
    """Remove the '>>>' or '>>' from braodcast snippets."""
    if snippet.startswith(">>>"):
        return snippet.replace(">>>", "", 1)
    if snippet.startswith(">>"):
        return snippet.replace(">>", "", 1)
    else:
        return snippet


st.title("Configuration")
if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before trying this step.')


else:
    if not st.session_state.config_step:
        named_entity = st.session_state.client_name

        # with st.form('User Inputs'):

        highlight_keyword = st_tags(
            label='Keywords to highlight in text (not case sensitive):',
            text='Press enter to add more',
            value=[st.session_state.client_name],
            maxtags=10,
            # key='1'
        )

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            # Sampling options
            sampling_option = st.radio(
                'Sampling options:',
                ['Take a statistically significant sample', 'Set my own sample size', 'Use full data'],
                help="Choose how to sample your uploaded data set."
            )

            if sampling_option == 'Take a statistically significant sample':
                def calculate_sample_size(N, confidence_level=0.95, margin_of_error=0.05, p=0.5):
                    # Z-score for 95% confidence level
                    Z = 1.96  # 95% confidence

                    numerator = N * (Z ** 2) * p * (1 - p)
                    denominator = (margin_of_error ** 2) * (N - 1) + (Z ** 2) * p * (1 - p)

                    return math.ceil(numerator / denominator)


                population_size = len(st.session_state.full_dataset)
                st.session_state.sample_size = calculate_sample_size(population_size)
                st.write(f"Calculated sample size: {st.session_state.sample_size}")

            elif sampling_option == 'Set my own sample size':
                # If the data set is smaller than 400, set the sample size to the data set size
                max_sample_size = min(400, len(st.session_state.full_dataset))
                custom_sample_size = st.number_input(
                    "Enter your desired sample size:",
                    min_value=1, max_value=max_sample_size, step=1, value=max_sample_size
                )
                st.session_state.sample_size = int(custom_sample_size)

            # elif sampling_option == 'Set my own sample size':
            #     custom_sample_size = st.number_input(
            #         "Enter your desired sample size:",
            #         min_value=1, max_value=len(st.session_state.full_dataset), step=1, value=400
            #     )
            #     st.session_state.sample_size = int(custom_sample_size)

            else:
                st.session_state.sample_size = len(st.session_state.full_dataset)
                st.write(f"Data set size: {st.session_state.sample_size}")

            similarity_threshold = 0.93
            st.session_state.similarity_threshold = similarity_threshold


            # random_sample = st.radio('Toning sample?', ['Yes, take a sample', 'No, use full data', ],
            #                          help="Get a statistically significant random sample based on your uploaded data set.")

        with c2:
            sentiment_opinion_selector = st.radio(
                "A.I. sentiment opinions?", ('Yes please', 'No thanks',),
                key='opinion_choice', help='Get GPT sentiment suggestions based on the article and client brand.')

        with c3:
            sentiment_type = st.radio("Sentiment Type", ['3-way', '5-way'],
                                      help='3-way is the standard approach.  5-way insteade uses *very positive*, *somewhat positive*, *neutral*, etc.')



        # submitted = submit_button("Save Configuration", type="primary")

        if st.button("Save Configuration", type="primary"):
            start_time = time.time()
            st.session_state.config_step = True

            sample_size = st.session_state.sample_size

            # Apply sampling
            if sample_size < len(st.session_state.full_dataset):
                df = st.session_state.df_traditional.sample(n=sample_size, random_state=1).reset_index(drop=True)
            else:
                df = st.session_state.df_traditional.copy()

            st.write(f"Full data size: {len(st.session_state.df_traditional)}")
            st.write(f"Sample size used: {len(df)}")  # Confirm the sample size

            # Check if 'Coverage Snippet' column exists and rename it to 'Snippet'
            if 'Coverage Snippet' in df.columns:
                df.rename(columns={'Coverage Snippet': 'Snippet'}, inplace=True)

            # Normalize and preprocess
            df['Headline'] = df['Headline'].apply(remove_extra_spaces)
            df['Snippet'] = df['Snippet'].apply(remove_extra_spaces)
            df['Snippet'] = df['Snippet'].apply(clean_snippet)
            df['Normalized Headline'] = df['Headline'].apply(normalize_text)
            df['Normalized Snippet'] = df['Snippet'].apply(normalize_text)

            # Cluster similar stories
            df = cluster_by_media_type(df, similarity_threshold=similarity_threshold)

            # Assign group IDs directly to the DataFrame
            df['Group ID'] = df['Group ID']

            # Drop the normalized columns
            df = df.drop(columns=['Normalized Headline', 'Normalized Snippet'], errors='ignore')

            # Update session state
            st.session_state.df_traditional = df.copy()

            # Calculate group counts
            group_counts = df.groupby('Group ID').size().reset_index(name='Group Count')

            # Group by 'Group ID' and calculate unique stories
            unique_stories = df.groupby('Group ID').agg(lambda x: x.iloc[0]).reset_index()
            unique_stories_with_counts = unique_stories.merge(group_counts, on='Group ID')

            # Debugging outputs
            st.write(f"Number of unique stories: {len(unique_stories_with_counts)}")  # Debugging output
            st.write(unique_stories_with_counts.head())  # Show the first few rows for verification

            # Sort unique stories by group count
            unique_stories_sorted = unique_stories_with_counts.sort_values(by='Group Count',
                                                                           ascending=False).reset_index(drop=True)

            # Update session state
            st.session_state.unique_stories = unique_stories_sorted

            # Time tracking
            end_time = time.time()
            st.session_state.elapsed_time = end_time - start_time

            if sentiment_opinion_selector == 'Yes please':
                st.session_state.sentiment_opinion = True
            else:
                st.session_state.sentiment_opinion = False

            if st.session_state.sentiment_opinion == True:
                # Ensure the "Sentiment Opinion" column exists in both dataframes
                if 'Sentiment Opinion' not in st.session_state.unique_stories.columns:
                    st.session_state.unique_stories['Sentiment Opinion'] = None

                if 'Sentiment Opinion' not in st.session_state.df_traditional.columns:
                    st.session_state.df_traditional['Sentiment Opinion'] = None

            # st.session_state.random_sample = random_sample
            # st.session_state.similarity_threshold = similarity_threshold
            st.session_state.highlight_keyword = highlight_keyword

            if sentiment_type == '3-way':
                st.session_state.sentiment_type = '3-way'

                sentiment_instruction = f"""
                    Instructions: Analyze the sentiment of the following news story specifically toward the named entity, {named_entity}. Focus on how the entity is portrayed using the following criteria to guide your analysis:
                    Positive Sentiment: Praises or highlights the entity's achievements, contributions, or strengths. Uses favorable or supportive language toward the entity. Attributes beneficial or advantageous outcomes to the entity's actions.
                    Neutral Sentiment: Provides balanced or factual coverage regarding the entity without clear positive or negative framing. Avoids strong language or bias in favor of or against the entity. Mentions the entity in a way that is neither supportive nor critical.
                    Negative Sentiment: Criticizes, highlights failures, or associates the entity with challenges or issues. Uses unfavorable, disparaging, or hostile language toward the entity. Attributes negative outcomes or controversies to the entity's actions or decisions.
                    IMPORTANT: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. Provide the uppercase sentiment classification (POSITIVE, NEUTRAL, or NEGATIVE) followed by a colon, then a one to two sentence explanation supporting your assessment. 
                    If {named_entity} is not mentioned in the story, please reply with the phrase "NOT RELEVANT: {named_entity} not mentioned in the story. Here is the story:
                    """

            else:
                st.session_state.sentiment_type = '5-way'
                sentiment_instruction = f"""
                    Instructions: Analyze the sentiment of the following news story specifically toward the named entity, {named_entity}. Focus on how the entity is portrayed using the following criteria to guide your analysis:
                    Very Positive: Strongly praises the entity's achievements, contributions, or strengths. Uses highly favorable, supportive, or celebratory language toward the entity. Attributes significant beneficial outcomes or positive impacts to the entity's actions.
                    Somewhat Positive: Highlights positive aspects of the entity or acknowledges strengths without strong praise. Uses moderately favorable language or frames the entity in a generally positive light. Attributes minor or moderate beneficial outcomes to the entity's actions.
                    Neutral: Provides balanced or factual coverage regarding the entity without clear positive or negative framing. Avoids strong language or bias in favor of or against the entity. Mentions the entity in a way that is neither supportive nor critical.
                    Somewhat Negative: Points out minor flaws, challenges, or criticisms related to the entity. Uses mildly unfavorable language or frames the entity in a somewhat negative light. Attributes minor or moderate negative outcomes to the entity's actions.
                    Very Negative: Strongly criticizes the entity's actions, failures, or associations with significant issues. Uses highly unfavorable, disparaging, or hostile language toward the entity. Attributes substantial negative outcomes or impacts to the entity's actions.
                    IMPORTANT: Focus your analysis strictly on the sentiment toward {named_entity} rather than the broader topic or context of the story. Provide the uppercase sentiment classification (VERY POSITIVE, SOMEWHAT POSITIVE, NEUTRAL, SOMEWHAT NEGATIVE, or VERY NEGATIVE) followed by a colon, then a one to two sentence explanation supporting your assessment.  
                    If {named_entity} is not mentioned in the story, please reply with the phrase "NOT RELEVANT: {named_entity} not mentioned in the story. Here is the story:
                    """



            st.session_state.sentiment_instruction = sentiment_instruction

            if 'Assigned Sentiment' not in st.session_state.df_traditional.columns:
                st.session_state.df_traditional['Assigned Sentiment'] = pd.NA

            if 'Flagged for Review' not in st.session_state.df_traditional.columns:
                st.session_state.df_traditional['Flagged for Review'] = False

            # st.session_state.df_traditional = st.session_state.df_traditional.dropna(thresh=3)

            # if random_sample == 'Yes, take a sample':
            #     def calculate_sample_size(N, confidence_level=0.95, margin_of_error=0.05, p=0.5):
            #         # Z-score for 95% confidence level
            #         Z = 1.96  # 95% confidence
            #
            #         numerator = N * (Z ** 2) * p * (1 - p)
            #         denominator = (margin_of_error ** 2) * (N - 1) + (Z ** 2) * p * (1 - p)
            #
            #         return math.ceil(numerator / denominator)
            #
            #
            #     # Example usage
            #     population_size = len(st.session_state.full_dataset)
            #     sample_size = calculate_sample_size(population_size)
            #
            #     st.session_state.sample_size = sample_size
            #
            #     # Take a random sample of the DataFrame
            #     if sample_size < population_size:
            #         df = st.session_state.df_traditional.sample(n=sample_size, random_state=1).reset_index(
            #             drop=True)  # n=sample_size, random_state=1
            #     else:
            #         # If the sample size is greater than or equal to the population, use the entire DataFrame
            #         df = st.session_state.df_traditional.copy()
            #
            #     st.write(f"Full data size: {len(st.session_state.df_traditional)}")
            #     st.write(f"Calculated sample size: {len(df)}")
            #
            #
            # else:
            #     df = st.session_state.df_traditional

            if st.session_state.sentiment_opinion == True:
                # Ensure the "Sentiment Opinion" column exists in both dataframes
                if 'Sentiment Opinion' not in st.session_state.unique_stories.columns:
                    df['Sentiment Opinion'] = None

                if 'Sentiment Opinion' not in st.session_state.df_traditional.columns:
                    df['Sentiment Opinion'] = None

            # df['Headline'] = df['Headline'].apply(remove_extra_spaces)
            # df['Snippet'] = df['Snippet'].apply(remove_extra_spaces)
            # df['Snippet'] = df['Snippet'].apply(clean_snippet)
            # df['Normalized Headline'] = df['Headline'].apply(normalize_text)
            # df['Normalized Snippet'] = df['Snippet'].apply(normalize_text)
            #
            # similarity_matrix = calculate_similarity(df)
            #
            # duplicate_groups = identify_duplicates(similarity_matrix)
            #
            # # Assign group IDs
            # group_ids = assign_group_ids(duplicate_groups)
            # df['Group ID'] = df.index.map(group_ids)
            #
            # # Drop the normalized columns
            # df = df.drop(columns=['Normalized Headline', 'Normalized Snippet'])
            #
            # st.session_state.df_traditional = df.copy()
            #
            # # Calculate group counts
            # group_counts = df.groupby('Group ID').size().reset_index(name='Group Count')
            #
            # # Group by 'Group ID' and keep the 'Group ID' column
            # unique_stories = df.groupby('Group ID').agg(lambda x: x.iloc[0]).reset_index()
            #
            # # Merge group counts with unique_stories
            # unique_stories_with_counts = unique_stories.merge(group_counts, on='Group ID')
            #
            # st.write('Unique Stories with Counts')
            # st.write(unique_stories_with_counts)  # NO GROUP ID here
            #
            # # Sort unique stories by group count in descending order
            # unique_stories_sorted = unique_stories_with_counts.sort_values(by='Group Count',
            #                                                                ascending=False).reset_index(drop=True)
            #
            # # Update the session state
            # st.session_state.unique_stories = unique_stories_sorted
            st.rerun()


    else:
        st.success('Configuration Completed!')
        st.write(f"Time taken: {st.session_state.elapsed_time:.2f} seconds")
        st.write(f"Full data size: {len(st.session_state.full_dataset)}")
        if 'sample_size' in st.session_state:
            st.write(f"Sample size used: {st.session_state.sample_size}")
        st.write(f"Unique stories in data: {len(st.session_state.unique_stories)}")
        st.dataframe(st.session_state.unique_stories)

        # st.success('Configuration Completed!')
        # st.write(f"Full data size: {len(st.session_state.full_dataset)}")
        # st.write(f"Calculated sample size: {st.session_state.sample_size}")
        # st.write(f"Unique stories in sample: {len(st.session_state.unique_stories)}")


        def reset_config():
            st.session_state.config_step = False
            # Reset other relevant session state variables as needed
            st.session_state.sentiment_opinion = None
            st.session_state.random_sample = None
            st.session_state.similarity_threshold = None
            st.session_state.sentiment_instruction = None
            st.session_state.df_traditional = st.session_state.full_dataset.copy()
            st.session_state.counter = 0


        # Add reset button
        if st.button("Reset Configuration"):
            reset_config()
            st.rerun()  # Rerun the script to reflect the reset state
