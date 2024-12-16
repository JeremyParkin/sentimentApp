import streamlit as st


def standard_sidebar():
    import streamlit as st
    st.sidebar.image('https://agilitypr.news/images/Agility-centered.svg', width=200)
    st.sidebar.header('MIG: Toning App')
    # st.sidebar.write("*Generate a random sample, group similar stories, and make toning easier*")
    st.sidebar.caption("Version: Dec 2024")

    # CSS to adjust sidebar
    adjust_nav = """
                            <style>
                            .eczjsme9 {
                                overflow: visible !important;
                                max-width: 244px !important;
                                }
                            .st-emotion-cache-a8w3f8 {
                                overflow: visible !important;
                                }
                            .st-emotion-cache-1cypcdb {
                                max-width: 244px !important;
                                }
                            </style>
                            """
    # Inject CSS with Markdown
    st.markdown(adjust_nav, unsafe_allow_html=True)

    # Add link to submit bug reports and feature requests
    st.sidebar.markdown(
        "[App Feedback](https://forms.office.com/Pages/ResponsePage.aspx?id=GvcJkLbBVUumZQrrWC6V07d2jCu79C5FsfEZJPZEfZxUNVlIVDRNNVBQVEgxQVFXNEM5VldUMkpXNS4u)")


def top_x_by_mentions(df, column_name):
    """Returns top 10 items by mention count"""
    if not df[column_name].notna().any():
        # If all values in the column are null, return an empty dataframe
        return
    top10 = df[[column_name, 'Mentions']].groupby(
        by=[column_name]).sum().sort_values(
        ['Mentions'], ascending=False)
    top10 = top10.rename(columns={"Mentions": "Hits"})

    return top10.head(10)


def fix_author(df, headline_text, new_author):
    """Updates all authors for a given headline"""
    df.loc[df["Headline"] == headline_text, "Author"] = new_author


def headline_authors(df, headline_text):
    """Returns the various authors for a given headline"""
    headline_authors = (df[df.Headline == headline_text].Author.value_counts().reset_index())
    return headline_authors


def normalize_text(text):
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
    # Convert to string in case the input is not a string
    text = str(text)
    # Remove extra spaces from the beginning and end
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text



def calculate_similarity(df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Normalized Headline'] + " " + df['Normalized Snippet'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def assign_group_ids(duplicates):
    group_id = 0
    group_ids = {}

    for i, similar_indices in duplicates.items():
        if i not in group_ids:
            group_ids[i] = group_id
            for index in similar_indices:
                group_ids[index] = group_id
            group_id += 1

    return group_ids



def identify_duplicates(similarity_matrix):
    duplicates = {}
    for i in range(similarity_matrix.shape[0]):
        duplicates[i] = []
        for j in range(similarity_matrix.shape[1]):
            if i != j and similarity_matrix[i][j] > st.session_state.similarity_threshold:
                duplicates[i].append(j)
    # st.write("Duplicates identified:", duplicates)
    return duplicates



def prompt_preview():
    prompt_text = f"""
                You are acting as an entity sentiment AI, indicating how a news story portrays {named_entity} based on these options: 
                \n{sentiment_rubric}
                \n{analysis_note}                
                \nHere is the news story...
            """
    return prompt_text


def generate_prompt(row, named_entity):
    prompt_text = f"""
        You are acting as an entity sentiment AI, indicating how a news story portrays {named_entity}.  Respond with only the ONE WORD label based on following options: 
        \n{sentiment_rubric}
        \n{analysis_note} 
        REMEMBER, respond ONLY with the ONE WORD SENTIMENT LABEL, nothing more.
        This is the news story:

        {row['Headline']}. {row['Snippet']}
        """
    return prompt_text


# Define function to generate sentiment prompt
def generate_sentiment_prompt(row, named_entity):
    return f"Please indicate the sentiment of the following news story as it relates to {named_entity}. Start with one word: Positive, Neutral, or Negative - followed by a colon then a one sentence rationale as to why that sentiment was chosen.\n\nThis is the news story:\n{row['Headline']}. {row['Example Snippet']}"


