#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy

# Twitter API credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAADTkzQEAAAAAEQP4tE%2FOkRJmfpEH0M0prs4uKKQ%3DNpnAmfA3xxLre3nICzZ6AB0L5yRvivFIfbHR0pcVkaQY98HhBA"

# Authenticate with API v2
client = tweepy.Client(bearer_token=bearer_token)

# Search recent tweets (up to 100 tweets)
query = "Netflix -is:retweet lang:en"
tweets = client.search_recent_tweets(query=query, max_results=10)

# Print tweet text
for tweet in tweets.data:
    print(tweet.text)


# In[ ]:


import nltk
nltk.download('vader_lexicon')  # For sentiment analysis
nltk.download('punkt')          # For tokenization
nltk.download('stopwords')      # For stop words removal


# In[ ]:


import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import matplotlib

# 🔹 Fix Backend Issue
matplotlib.use("TkAgg")  # Try 'Qt5Agg' or 'Agg' if this doesn't work

# Sample Data
text_data = """
LF BUYERS PREMIUM ACCOUNT PH🌸 VIU WPS MS 365 VIVAONE NETFLIX QUIZLET YOUTUBE QUILLBOT CANVA PRO PRIME VIDEO 
GRAMMARLY CANVA LIFETIME TURNITIN STUDENT
NETFLIX DISNEY+ HOTSTAR VIU IQIYI WETV AMAZON PRIME CANVA VIDIO YOUTUBE SPOTIFY APPLE MUSIC CHATGPT CAPCUT PRO
"""

# ✅ Step 1: Data Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"@\w+", "", text)  
    text = re.sub(r"#\w+", "", text)  
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    return text

cleaned_text = clean_text(text_data)

# ✅ Step 2: Tokenization
words = cleaned_text.split()

# ✅ Step 3: Count Frequency
word_counts = Counter(words)

# ✅ Step 4: Extract Subscription Services
services_list = [
    "netflix", "canva", "spotify", "youtube", "viu", "grammarly", 
    "quizlet", "prime", "wps", "ms365", "apple", "iqiyi", "chatgpt"
]
services_count = {service: word_counts[service] for service in services_list if service in word_counts}

# ✅ Step 5: Bar Chart (Fixed FutureWarning)
plt.figure(figsize=(10, 5))
sns.barplot(
    x=list(services_count.keys()), 
    y=list(services_count.values()), 
    hue=list(services_count.keys()),  
    dodge=False,
    legend=False,
    palette="coolwarm"
)
plt.title("Most Mentioned Subscription Services", fontsize=14)
plt.xlabel("Subscription Services", fontsize=12)
plt.ylabel("Mentions Count", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.ion()  # Enable interactive mode
plt.show(block=True)  # Fix UserWarning

# ✅ Step 6: Word Cloud (Fixed UserWarning)
plt.figure(figsize=(10, 5))
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="coolwarm").generate(cleaned_text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Keywords", fontsize=14)

plt.show(block=True)  # Fix UserWarning


# In[ ]:


import re

def clean_tweet(tweet):
    tweet = re.sub(r"http\S+", "", tweet)  # Remove URLs
    tweet = re.sub(r"@\w+", "", tweet)  # Remove mentions (@user)
    tweet = re.sub(r"#\w+", "", tweet)  # Remove hashtags
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)  # Remove special characters
    return tweet.lower().strip()


# In[ ]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Initialize Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Sample text data
text_data = """
LF BUYERS PREMIUM ACCOUNT PH🌸 VIU WPS MS 365 VIVAONE NETFLIX QUIZLET YOUTUBE QUILLBOT CANVA PRO PRIME VIDEO 
GRAMMARLY CANVA LIFETIME TURNITIN STUDENT
NETFLIX DISNEY+ HOTSTAR VIU IQIYI WETV AMAZON PRIME CANVA VIDIO YOUTUBE SPOTIFY APPLE MUSIC CHATGPT CAPCUT PRO
"""

# Analyze Sentiment
sentiment_score = sia.polarity_scores(text_data)

# Extract Positive, Neutral, and Negative Scores
labels = ['Positive', 'Neutral', 'Negative']
scores = [sentiment_score['pos'], sentiment_score['neu'], sentiment_score['neg']]

# Plot Sentiment Scores
plt.figure(figsize=(6, 4))
plt.bar(labels, scores, color=['green', 'gray', 'red'])
plt.xlabel("Sentiment")
plt.ylabel("Score")
plt.title("Sentiment Analysis of Text Data")
plt.show()

# Print Sentiment Scores
print("Sentiment Scores:", sentiment_score)


# In[ ]:


from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer


# In[ ]:


# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiment for each tweet
tweet_data = []
for tweet in tweets.data:
    text = clean_tweet(tweet.text)
    blob_sentiment = TextBlob(text).sentiment.polarity  # TextBlob score
    vader_sentiment = sia.polarity_scores(text)["compound"]  # VADER score
    
    # Determine overall sentiment
    sentiment = "Neutral"
    if vader_sentiment > 0.05:
        sentiment = "Positive"
    elif vader_sentiment < -0.05:
        sentiment = "Negative"
    
    tweet_data.append([tweet.text, sentiment])

# Convert results into a DataFrame
import pandas as pd
df = pd.DataFrame(tweet_data, columns=["Tweet", "Sentiment"])
print(df)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(sentiments):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=sentiments, palette="coolwarm")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    plt.title("Sentiment Distribution")
    st.pyplot(plt)


# In[3]:


from wordcloud import WordCloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


# In[4]:


import plotly.express as px

def plot_pie_chart(sentiments):
    sentiment_counts = sentiments.value_counts()
    fig = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title="Sentiment Proportion",
        color_discrete_sequence=["green", "red", "blue"]
    )
    st.plotly_chart(fig)


# In[5]:


# -*- coding: utf-8 -*-
import streamlit as st

st.title("Twitter Sentiment Analysis")
st.write("This is a Streamlit app!")


# In[6]:


# import pandas as pd
# import streamlit as st

# # Sample Data (Replace with actual data)
# data = {
#     "Text": ["I love Netflix!", "Spotify is amazing", "YouTube is my favorite"],
#     "Sentiment": ["Positive", "Positive", "Positive"]
# }

# df = pd.DataFrame(data)

# st.title("Twitter Sentiment Analysis")
# st.write("Below is the sentiment analysis of Twitter data:")

# # Display DataFrame
# st.dataframe(df)


# In[7]:


import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

# ✅ Download required NLTK data
nltk.download('vader_lexicon')

# ✅ Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# 🎯 App Title
st.title("📊 Twitter Sentiment Analysis with Streamlit")

# 🔹 Description
st.markdown("""
This app analyzes the sentiment of text data using **VADER Sentiment Analysis**.  
You can use this tool to gauge the sentiment of tweets, reviews, or any text data.
""")

# ✅ Upload CSV File
uploaded_file = st.file_uploader("📂 Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file is not None:
    # ✅ Read CSV
    df = pd.read_csv(uploaded_file)
    
    # ✅ Display first 5 rows
    st.subheader("📌 Sample Data")
    st.write(df.head())

    # ✅ Check if 'text' column exists
    if 'text' in df.columns:
        # ✅ Perform Sentiment Analysis
        df["sentiment_score"] = df["text"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
        df["sentiment"] = df["sentiment_score"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

        # ✅ Display sentiment analysis results
        st.subheader("📊 Sentiment Analysis Results")
        st.write(df[["text", "sentiment", "sentiment_score"]])

        # ✅ Plot Pie Chart
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Proportion",
            color_discrete_sequence=["green", "red", "blue"]
        )
        st.plotly_chart(fig)

        # ✅ Download Processed Data
        st.subheader("📥 Download Sentiment Analysis Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="sentiment_analysis_results.csv", mime="text/csv")
    else:
        st.error("❌ The uploaded file must contain a 'text' column!")
else:
    st.info("📌 Please upload a CSV file to analyze sentiment.")

# ✅ Footer
st.markdown("""
---
🚀 **Project by Amisha Kashyap**  
🌟 **GitHub:** [Click Here](https://github.com/Amishakashyap/streamlit-sentiment-analysis)
""")


# In[8]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Sample text data (Replace this with Twitter data if needed)
text_data = """
LF BUYERS PREMIUM ACCOUNT PH🌸 VIU WPS MS 365 VIVAONE NETFLIX QUIZLET YOUTUBE QUILLBOT CANVA PRO PRIME VIDEO 
GRAMMARLY CANVA LIFETIME TURNITIN STUDENT NETFLIX DISNEY+ HOTSTAR VIU IQIYI WETV AMAZON PRIME CANVA VIDIO YOUTUBE SPOTIFY APPLE MUSIC CHATGPT CAPCUT PRO
"""

# Analyze Sentiment
sentiment_score = sia.polarity_scores(text_data)

# Extract Positive, Neutral, and Negative Scores
labels = ['Positive', 'Neutral', 'Negative']
scores = [sentiment_score['pos'], sentiment_score['neu'], sentiment_score['neg']]

# Print Sentiment Scores in the Console
print("🔹 Sentiment Analysis Results:")
print(f"✅ Positive: {sentiment_score['pos']}")
print(f"⚪ Neutral: {sentiment_score['neu']}")
print(f"❌ Negative: {sentiment_score['neg']}")

# Plot Sentiment Scores
plt.figure(figsize=(6, 4))
plt.bar(labels, scores, color=['green', 'gray', 'red'])
plt.xlabel("Sentiment")
plt.ylabel("Score")
plt.title("Sentiment Analysis of Text Data")
plt.show()


# In[9]:


import nltk
nltk.download('vader_lexicon')  # For sentiment analysis
nltk.download('punkt')  # For tokenization (optional)



# In[10]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Sample text data
text_data = """
LF BUYERS PREMIUM ACCOUNT PH🌸 VIU WPS MS 365 VIVAONE NETFLIX QUIZLET YOUTUBE QUILLBOT CANVA PRO PRIME VIDEO 
GRAMMARLY CANVA LIFETIME TURNITIN STUDENT NETFLIX DISNEY+ HOTSTAR VIU IQIYI WETV AMAZON PRIME CANVA VIDIO YOUTUBE SPOTIFY APPLE MUSIC CHATGPT CAPCUT PRO
"""

# Analyze Sentiment
sentiment_score = sia.polarity_scores(text_data)

# Extract Positive, Neutral, and Negative Scores
labels = ['Positive', 'Neutral', 'Negative']
scores = [sentiment_score['pos'], sentiment_score['neu'], sentiment_score['neg']]

# Create visualization
plt.figure(figsize=(6, 4), dpi=300)
plt.bar(labels, scores, color=['green', 'gray', 'red'])
plt.xlabel("Sentiment", fontsize=12, fontweight='bold')
plt.ylabel("Score", fontsize=12, fontweight='bold')
plt.title("Sentiment Analysis of Text Data", fontsize=14, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1)

# Save the image for LinkedIn post
image_path = "/mnt/data/sentiment_analysis_plot.png"
plt.savefig(image_path, bbox_inches='tight', dpi=300)
plt.show()

# Return image path for download
image_path


# In[ ]:


# Retry saving the image and ensure it's available for download

# Create visualization again
plt.figure(figsize=(6, 4), dpi=300)
plt.bar(labels, scores, color=['green', 'gray', 'red'])
plt.xlabel("Sentiment", fontsize=12, fontweight='bold')
plt.ylabel("Score", fontsize=12, fontweight='bold')
plt.title("Sentiment Analysis of Text Data", fontsize=14, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1)

# Save the image for LinkedIn post
image_path = "/mnt/data/sentiment_analysis_chart.png"
plt.savefig(image_path, bbox_inches='tight', dpi=300)
plt.show()

# Return image path for download
image_path


# In[ ]:


# Retry saving the image and ensure it's available for download

# Create visualization again
plt.figure(figsize=(6, 4), dpi=300)
plt.bar(labels, scores, color=['green', 'gray', 'red'])
plt.xlabel("Sentiment", fontsize=12, fontweight='bold')
plt.ylabel("Score", fontsize=12, fontweight='bold')
plt.title("Sentiment Analysis of Text Data", fontsize=14, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1)

# Save the image for LinkedIn post
image_path = "/mnt/data/sentiment_analysis_chart.png"
plt.savefig(image_path, bbox_inches='tight', dpi=300)
plt.show()

# Return image path for download
image_path


# In[ ]:


image_path = "C:\\Users\\amish\\Desktop\\sentiment_analysis_plot.png"
plt.savefig(image_path, bbox_inches='tight', dpi=300)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




