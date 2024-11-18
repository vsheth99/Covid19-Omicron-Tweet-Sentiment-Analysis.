import pandas as pd
import glob
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define folder path to your CSV files and month mapping
folder_path = "/Users/vivek/Downloads/Tweets"  
month_map = {
    "TweetIDs_November.csv": "November",
    "TweetIDs_December.csv": "December",
    "TweetIDs_January.csv": "January",
    "TweetIDs_February.csv": "February",
    "TweetIDs_March.csv": "March",
    "TweetIDs_April.csv": "April",
    "TweetIDs_May.csv": "May"
}

# Load each CSV file, add a 'month' column, and combine them
dfs = []
for file_path in glob.glob(folder_path + "/*.csv"):
    file_name = file_path.split("/")[-1]
    month = month_map.get(file_name, None)
    
    if month:
        df = pd.read_csv(file_path)
        df['month'] = month  # Add month column
        dfs.append(df)
    else:
        print(f"Warning: {file_name} not found in month map. Skipping.")

# Concatenate DataFrames if any files were successfully loaded
if dfs:
    tweets_df = pd.concat(dfs, ignore_index=True)
    print("Data loaded successfully.")
else:
    print("No DataFrames to concatenate. Check if CSV files are empty or improperly formatted.")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to categorize sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis and add 'sentiment' column
tweets_df['sentiment'] = tweets_df['text'].apply(get_sentiment)

# Define colors for each sentiment
colors = {
    'Positive': '#66c2a5',  # Green
    'Negative': '#fc8d62',  # Red
    'Neutral': '#8da0cb'    # Blue
}

# Create and save pie charts for each month
for month in tweets_df['month'].unique():
    # Filter data for the specific month
    month_data = tweets_df[tweets_df['month'] == month]
    
    # Get sentiment counts
    sentiment_counts = month_data['sentiment'].value_counts()
    
    # Define colors for the current pie chart based on sentiment order
    pie_colors = [colors[sentiment] for sentiment in sentiment_counts.index]
    
    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=pie_colors
    )
    plt.title(f'Sentiment Distribution for {month}')
    
    # Save plot as PNG
    plt.savefig(f'sentiment_distribution_{month}_pie.png')
    plt.show()  # Show the plot
