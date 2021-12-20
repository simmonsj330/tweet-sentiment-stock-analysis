import pandas as pd
from pandas import DataFrame
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

'''
This file aims to analyze the tweet data located in tweets.csv
'''

def remove_tagged(tweet_string):
    marker = False
    removal_list = []
    tmp = ''
    for i in range(len(tweet_string)):
        if tweet_string[i] == '@':
            marker = True
        if marker == True and tweet_string[i] != ' ':
            tmp = tmp + tweet_string[i]
        if marker == True and tweet_string[i] == ' ':
            removal_list.append(tmp)
            tmp = ''
            marker = False
    

    #print(tweet)
    #print(removal_list)
    for i in range(len(removal_list)):
        tweet_string = tweet_string.replace(removal_list[i], '')

    return tweet_string

if __name__ == "__main__":

    #Reading in tweet data csv
    df = pd.read_csv ('tweets.csv')

    #Iterating through data to find sentiment of tweet text
    sentiment_list = []
    for i in range(len(df)):


        tweet = str(df['text'].iloc[i])
        tweet = remove_tagged(tweet)
        testimonial = TextBlob(tweet)
        sentiment_list.append(testimonial.sentiment.polarity)

    #Storing sentiment onto new data frame column
    df['sentiment'] = sentiment_list
    outfile = DataFrame(df,columns=["id","created_at","favorite_count","retweet_count","text","sentiment"])
    df.to_csv('with_sentiment.csv',index=False)

    # get financial data
    fin_d = pd.read_csv('tsla.csv')
    fin_d = fin_d[::-1].reset_index(drop = True)
    fin_d['change'] = fin_d['Close'] - fin_d['Open']
    print(fin_d)
    
    #get only date of tweet
    for i in range(len(df)):
        df['created_at'][i] = df['created_at'][i].partition(' ')[0]

    #iterate through tsla, see if tweet for date, if not drop
    df['change'] = ""
    for i in range(len(fin_d)):
        truth = df[df['created_at'] == fin_d['Date'][i]]
        if truth.empty:
            df.drop(i)
        else:
            for element in truth.index:
                df['change'][element] = fin_d['change'][i]
            
    df = df.where(df['change'] != '').dropna()
    df.to_csv('with_financials.csv', index=False)

    # make model and predict
    sentiment = df['sentiment'].to_numpy()
    close = df['change'].to_numpy()
    X = [sentiment, close]
    print('sentiment', sentiment)
    print('change', close)
    with np.printoptions(threshold=np.inf):
        print(X)

    train_sent, train_close, test_sent, test_close = train_test_split(sentiment, close, random_state=425)
    model = KNeighborsClassifier('distance').fit(train_sent, train_close)