import nltk
import string
import pickle
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from nltk.stem.porter import PorterStemmer
from streamlit_option_menu import option_menu
from nltk.sentiment.vader import SentimentIntensityAnalyzer


st.set_page_config(page_title="Sentiment Analysis",
                   page_icon='browser-chrome',

                   )

selected = option_menu(
    menu_title=None,
    options=["Home", "Data Sentiments", "Dashboard"],
    icons=["house", "upload", "bar-chart-line-fill"],
    default_index=0,
    orientation="horizontal",
)

if selected == 'Home':
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        ps = PorterStemmer()
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)

    tfidf = pickle.load(open('vectorizer_mnb.pkl', 'rb'))
    model = pickle.load(open('model_mnb.pkl', 'rb'))

    st.title("Sentiment Classifier")
    placeholder = st.empty()
    input_review = placeholder.text_input(
        "Enter the reviews (required)", key=1)

    if st.button('Predict', key=3):
        nltk.download('punkt')
        transform_review = transform_text(input_review)
        score = TextBlob(input_review).sentiment.polarity
        score = round(abs(score*100))
        vector_input = tfidf.transform([transform_review])

        result = model.predict(vector_input)[0]

        if not input_review:
            st.warning("Please fill out the required field")
       
        elif result == 0:
            st.subheader("Our Model has Predicted Negative Sentiment")

        elif result == 1:
            st.subheader("Our Model has Predicted Neutral Sentiment")
        
        else:
            st.subheader("Our Model has Predicted Positive Sentiment")

        # input_review = placeholder.text_input("Enter the reviews (required)",value='',key=2)

        if not input_review:
            pass
        elif score == 0 or 'love' in input_review:
            st.subheader('100%')
        else:
            st.subheader(str(score) + '%')

if selected == 'Data Sentiments':

    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        ps = PorterStemmer()
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)

    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    def upload_file():
        file = st.file_uploader(label="Choose a file", type=['csv'])
        if file is not None:
            df = pd.read_csv(file, encoding='MacRoman')
            # if st.checkbox("Show raw data"):
            st.write('**Raw Data**')
            st.dataframe(df.head(3))
            if st.button('Predict', key=1):
                try:
                    df = df.head(1000)
                    df['transformed_text'] = df['Tweets'].apply(transform_text)
                    df['Polarity'] = df['transformed_text'].apply(
                        getPolarity)
                    a= []
                    for i in df['Polarity']:
                        a.append(round(abs(i*100)))
                        
                    df['Polarity_Score']= ''
                    df['Polarity_Score'] = a
                    df['Predicted_label'] = ''
                    df_input = tfidf.fit_transform(
                        df['transformed_text']).toarray()
                    new_result = model.predict(df_input)
                    df['Predicted_label'] = new_result
                    df['Predicted_label'] = df['Predicted_label'].replace(
                        [0, 1, 2], ['Negative', 'Neutral', 'Positive'])
                    predicted_df = df[['Predicted_label',
                                       'transformed_text', 'Date', 'Polarity_Score']]
                    st.write('**Predicted Dataframe**')
                    st.dataframe(predicted_df.head(5))
                    st.write('Negative', round(predicted_df['Predicted_label'].value_counts()[
                             'Negative']/len(predicted_df) * 100), '%')
                    st.write('Positive', round(predicted_df['Predicted_label'].value_counts()[
                             'Positive']/len(predicted_df) * 100), '%')
                    st.write('Neutral', round(predicted_df['Predicted_label'].value_counts()[
                             'Neutral']/len(predicted_df) * 100), '%')
                    pred_count = predicted_df['Predicted_label'].value_counts()
                    col3, col4 = st.columns([3, 1])
                    pred_fig = px.bar(
                        pred_count,
                        color=pred_count,
                    )
                    col3.markdown('**Number of Sentiments count**')
                    col3.plotly_chart(pred_fig)
                    col13, col14 = st.columns([3, 1])
                    line_fig = px.line(df, x='Date', y='Num_Of_Likes')
                    col13.markdown('**Line chart of Num of likes**')
                    col13.plotly_chart(line_fig)
                    st.write('**Line chart of Num of Retweets**')
                    line_fig1 = px.line(df, x='Date', y='Num_Of_Retwwet')
                    st.plotly_chart(line_fig1)
                    # st.write('**Line chart of Predicted Label**')
                    line_fig2 = px.line(predicted_df.head(300), x='Date', y=np.random.normal(
                        5.0, 1.0, 300), color='Predicted_label')
                    st.plotly_chart(line_fig2)

                except:
                    pass
                try:
                    df['transformed_text'] = df['text'].apply(transform_text)
                    df['Polarity'] = df['transformed_text'].apply(
                        getPolarity)
                    a= []
                    for i in df['Polarity']:
                        a.append(round(abs(i*100)))
                        
                    df['Polarity_Score']= ''
                    df['Polarity_Score'] = a
                    df_input = tfidf.fit_transform(
                        df['transformed_text']).toarray()
                    new_result = model.predict(df_input)
                    df['Predicted_label'] = new_result
                    df['Predicted_label'] = df['Predicted_label'].replace(
                        [0, 1, 2], ['Negative', 'Neutral', 'Positive'])
                    predicted_df = df[['Predicted_label',
                                       'transformed_text', 'Date', 'Polarity_Score']]
                    st.write('**Predicted Dataframe**')
                    st.dataframe(predicted_df.head(5))
                    st.write('Negative', round(predicted_df['Predicted_label'].value_counts()[
                             'Negative']/len(predicted_df) * 100), '%')
                    st.write('Positive', round(predicted_df['Predicted_label'].value_counts()[
                             'Positive']/len(predicted_df) * 100), '%')
                    st.write('Neutral', round(predicted_df['Predicted_label'].value_counts()[
                             'Neutral']/len(predicted_df) * 100), '%')
                    pred_count = predicted_df['Predicted_label'].value_counts()
                    col3, col4 = st.columns([3, 1])
                    pred_fig = px.bar(
                        pred_count,
                        color=pred_count,
                    )
                    col3.markdown('**Number of Sentiments count**')
                    col3.plotly_chart(pred_fig)
                    col13, col14 = st.columns([3, 1])
                    line_fig = px.line(df, x='Date', y='Num_Of_Likes')
                    col13.markdown('**Line chart of Num of likes**')
                    col13.plotly_chart(line_fig)
                    st.write('**Line chart of Num of Retweets**')
                    line_fig1 = px.line(df, x='Date', y='Num_Of_Retwwet')
                    st.plotly_chart(line_fig1)
                    st.write('**Line chart of Predicted Label**')
                    line_fig2 = px.line(predicted_df.head(500), x='Date', y=np.random.normal(
                        5.0, 1.0, 500), color='Predicted_label')
                    st.plotly_chart(line_fig2)
                    # scatter_fig = px.scatter(predicted_df.head(900), x = 'Date' , y = np.random.normal(5.0, 1.0, 900) ,color='Predicted_label')
                    # st.plotly_chart(scatter_fig)
                except:
                    pass

    upload_file()


if selected == "Dashboard":
    df = pd.read_csv('Tweets.csv')
    df = df.drop_duplicates(keep='first')
    st.subheader("Airline tweets by sentiment")
    # df=df.head(5000)
    # df['transformed_text'] = df['text'].apply(transform_text)
    nltk.download('vader_lexicon')
    df['date'] = pd.to_datetime(df['tweet_created'])

    choice = st.multiselect("Pick airlines", ('US Airways', 'United',
                            'American', 'Southwest', 'Delta', 'Virgin America'), key='0')

    if len(choice) > 0:
        choice_data = df[df.airline.isin(choice)]
        newfig = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count',
                              color='airline_sentiment',
                              facet_col='airline_sentiment', labels={'airline_sentiment': 'tweets'}, height=600, width=800)
        st.plotly_chart(newfig)

    sentiments = SentimentIntensityAnalyzer()
    df["positive"] = [sentiments.polarity_scores(
        i)["pos"] for i in df["text"]]
    df["negative"] = [-1 *
                      sentiments.polarity_scores(i)["neg"] for i in df["text"]]
    df["neutral"] = [sentiments.polarity_scores(
        i)["neu"] for i in df["text"]]

    def sentiment(d):
        return (d["positive"] + d["negative"] or d['negative'] - d['neutral'])

    df['sentiment'] = df.apply(lambda row: sentiment(row), axis=1)

    col16, col17 = st.columns([3, 1])
    fig8 = px.scatter(df, x='date',
                      y='sentiment', color='sentiment')
    fig8.update_yaxes(tickvals=[-1, 0, 1])
    fig8.add_hline(y=0, line_dash="dash", line_color="black")
    fig8.add_hline(y=0.5, line_dash="dash", line_color="green")
    fig8.add_hline(y=-0.4, line_dash="dash", line_color="purple")

    col16.markdown("**Density Distribution Scatter plot**")
    col16.plotly_chart(fig8)

    df['day_of_week'] = df['date'].dt.day_name()
    grp = df.groupby(df['day_of_week']).agg({'airline_sentiment': 'count'})
    grp.reset_index(inplace=True)
    cats = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']

    grp_sort = grp.groupby(['day_of_week']).sum().reindex(cats)

    grp_sort.reset_index(inplace=True)
    col12, col14 = st.columns([3, 1])
    fig5 = px.bar(x=grp_sort['day_of_week'], y=grp_sort['airline_sentiment'],
                  labels={
        "y": "Tweet Count",
        "x": "W Day",
    })
    col12.markdown("**Tweets by Week Day's**")
    col12.plotly_chart(fig5)
    # col10, col11 = st.columns([3, 1])
    # x = np.random.normal(5.0, 1.0, 14604)

    # fig4 = px.scatter(df, x, y=df['airline_sentiment_confidence'],
    #                   color='airline_sentiment', hover_data=['airline'])

    # # Second chart title
    # col10.markdown('**Distribution of Data**')

    # col10.plotly_chart(fig4)
    st.markdown('--------')
    df['month_of_year'] = df['date'].dt.month_name()
    grp_month = df.groupby('month_of_year').agg({'airline_sentiment': 'count'})

    grp_month.reset_index(inplace=True)
    col15, col16 = st.columns([3, 1])

    fig7 = px.bar(x=grp_month['month_of_year'], y=grp_month['airline_sentiment'],
                  labels={
        "y": "Tweet Count",
        "x": "Month",
    })

    col15.markdown("**Tweets by Month**")
    col15.plotly_chart(fig7)

    col3, col4 = st.columns([3, 1])
    airline_sentiment_count = df['airline_sentiment'].value_counts()
    fig1 = px.pie(
        df['airline_sentiment'],
        names="airline_sentiment",
    )

    # Second chart title
    col3.markdown('**Number of Sentiments count**')

    col3.plotly_chart(fig1)
    # col2.write(airline_count)
    st.markdown('--------')

    col1, col2 = st.columns([3, 1])
    airline_count = df['airline'].value_counts()
    fig = px.pie(
        df['airline'],
        names="airline",
    )

    # first chart title
    col1.markdown('**Number of airline counts**')

    col1.plotly_chart(fig)
    # col2.write(airline_count)
    # st.markdown('--------')

    # airline_sentiment.plot(kind='bar')
    # col5, col6 = st.columns([4, 1])
    # airline_sentiment_group = df.groupby(
    #     ['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
    # fig2 = px.bar(
    #     airline_sentiment_group
    # )
    # # Third chart title
    # col5.markdown('**Airline Sentiments**')
    # col5.plotly_chart(fig2)
    st.markdown('--------')

    new_df = df.drop(['user_timezone', 'tweet_location', 'tweet_coord',
                      'negativereason_gold', 'airline_sentiment_gold',
                      'negativereason_confidence', 'negativereason', 'tweet_id', 'name',
                      'retweet_count'], axis=1)

    st.markdown('**Detailed view of data**')
    st.dataframe(new_df)


hide_style = """
    <style>
    footer {visibility:hidden;}
    </style>
    """
st.markdown(hide_style, unsafe_allow_html=True)
