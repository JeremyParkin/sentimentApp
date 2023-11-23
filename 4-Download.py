import streamlit as st
import pandas as pd
import mig_functions as mig
import io
import altair as alt


# Sidebar configuration
mig.standard_sidebar()

st.title("Download")

if not st.session_state.upload_step:
    st.error('Please upload a CSV before trying this step.')

elif not st.session_state.config_step:
    st.error('Please run the configuration step before trying this step.')
else:

    if pd.notna(st.session_state.df_traditional['Assigned Sentiment']).any():
        # Count the frequency of each sentiment and calculate the percentage
        sentiment_counts = st.session_state.df_traditional['Assigned Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        total = sentiment_counts['Count'].sum()
        sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total)

        if st.session_state.sentiment_type == '5-way':
            custom_order = ['VERY POSITIVE', 'SOMEWHAT POSITIVE', 'NEUTRAL', 'SOMEWHAT NEGATIVE', 'VERY NEGATIVE', 'NOT RELEVANT']
        else:
            custom_order = ['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'NOT RELEVANT']

        # Apply the custom order to 'Assigned Sentiment' in df_traditional
        st.session_state.df_traditional['Assigned Sentiment'] = pd.Categorical(
            st.session_state.df_traditional['Assigned Sentiment'], categories=custom_order, ordered=True)

        # Sort the DataFrame
        # st.session_state.df_traditional.sort_values(by='Assigned Sentiment', ascending=True, inplace=True)

        # Ensure all sentiments are in the dataframe (to avoid key errors in sorting)
        for sentiment in custom_order:
            if sentiment not in sentiment_counts['Sentiment'].values:
                # Create a new DataFrame for the missing sentiment and concatenate it
                missing_sentiment_df = pd.DataFrame({'Sentiment': [sentiment], 'Count': [0], 'Percentage': [0.0]})
                sentiment_counts = pd.concat([sentiment_counts, missing_sentiment_df], ignore_index=True)



        # Sort the dataframe based on the custom order
        sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=custom_order, ordered=True)

        st.subheader('Sentiment Statistics')

        col1, col2 = st.columns([3,2], gap='large')
        with col1:
            color_mapping = {
                "POSITIVE": "green",
                "NEUTRAL": "yellow",
                "NEGATIVE": "red",
                "VERY POSITIVE": "darkgreen",
                "SOMEWHAT POSITIVE": "limegreen",
                "SOMEWHAT NEGATIVE": "coral",
                "VERY NEGATIVE": "maroon",
                "NOT RELEVANT": "dimgray"
            }

            # Create the color domain and range from the sorted DataFrame
            color_domain = custom_order
            color_range = [color_mapping.get(sentiment, "grey") for sentiment in color_domain]

            # Create the color scale
            color_scale = alt.Scale(domain=color_domain, range=color_range)


            # Create the base chart for the horizontal bar chart
            base = alt.Chart(sentiment_counts).encode(
                y=alt.Y('Sentiment:N', sort=custom_order),
            )

            # Create the bar chart
            bar_chart = base.mark_bar().encode(
                x='Count:Q',
                color=alt.Color('Sentiment:N', scale=color_scale, legend=None),
                tooltip=['Sentiment', alt.Tooltip('Percentage', format='.1f', title='Percent'), 'Count']
            )

            # Create the text labels for the bars
            text = base.mark_text(
                align='left',
                baseline='middle',
                dx=3,  # Nudges text to the right so it doesn't appear on top of the bar
                color='ivory'  # Set text color to off-white
            ).encode(
                x=alt.X('Count:Q', axis=alt.Axis(title='Mentions')),
                text=alt.Text('Percentage:N', format='.1%')
            )

            # Combine the bar and text charts
            chart = (bar_chart + text).properties(
                # title='Sentiment Bar Chart',
                width=600,
                height=250
            ).configure_view(
                strokeWidth=0
            ).configure_axisLeft(
                labelLimit=180  # Increase label limit to accommodate longer labels
            )

            st.altair_chart(chart, use_container_width=True)



        with col2:

            # Adding percentage column to the stats table
            sentiment_stats = sentiment_counts.copy()


            # Convert 'Sentiment' to a categorical type with the custom order
            sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=custom_order,
                                                           ordered=True)

            # Sort 'sentiment_counts' by the categorical order
            sentiment_counts.sort_values(by='Sentiment', inplace=True)

            sentiment_counts['Percentage'] = (sentiment_counts['Percentage'] * 100).apply(lambda x: "{:.1f}%".format(x))


            # Display the table without the index
            st.dataframe(sentiment_counts, hide_index=True,)



    # Create a download link for the DataFrame as an Excel file
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    st.session_state.df_traditional.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()  # Use writer.close() instead of writer.save()
    output.seek(0)
    st.download_button(
        label="Download sentiment Excel",
        data=output,
        file_name=f"{st.session_state.client_name} - {st.session_state.focus} - Sentiment.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type='primary'
    )

    st.dataframe(st.session_state.df_traditional)