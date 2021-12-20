import pandas as pd
from pandas import DataFrame
from pandas._libs.missing import NA
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import seaborn as sns
from sklearn.metrics import classification_report

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
    df = df.drop(['text'], axis=1)
    df['sentiment'] = sentiment_list
    outfile = DataFrame(df,columns=["id","created_at","favorite_count","retweet_count","sentiment"])
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
    
    df = df.loc[(df['sentiment'] != 0)]
    df.dropna(inplace=True)
    df['change'].replace('', np.nan, inplace=True)
    df.dropna(subset=['change'], inplace=True)
    df.to_csv('with_financials.csv', index=False)
    print(df.isnull())

    # make model and predict
    vars = ['favorite_count', 'retweet_count', 'sentiment']
    # make 1's 0's for positive negative
    Y = np.zeros(shape=df['change'].shape)
    change = df['change'].to_numpy()
    for i in range(np.size(change)):
        if change[i] >= 0:
            Y[i] = 1
        else:
            Y[i] = 0
    n_neighbors = 2
    for weights in ['distance', 'uniform']:
        print(weights)
        print('------------')
        var1 = 'sentiment'
        for var2 in vars:
            print(var2)
            df = pd.read_csv('with_financials.csv', quoting=csv.QUOTE_NONNUMERIC)._get_numeric_data()
            var1_data = df[var1].to_numpy()
            var2_data = df[var2].to_numpy()
            
            # make X vector
            X = np.column_stack((var1_data, var2_data))
            st = StandardScaler().fit(X)
            X = st.transform(X)
            # make model and predict
            train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=425)
            #print('trainx: ', train_X)
            #print('trainy: ', train_Y)
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights).fit(train_X, train_Y)
            # model = tree.DecisionTreeClassifier().fit(train_X, train_Y)
            y_pred = model.predict(test_X)
            print(accuracy_score(test_Y, y_pred))
            predictions = model.predict(test_X)

            print(classification_report(test_Y, predictions))
            '''
            # Create color maps
            h = 0.02
            cmap_light = ListedColormap(["orange","cornflowerblue"])
            cmap_bold = ["darkorange", "darkblue"]
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            sns.scatterplot(
                x=X[:, 0],
                y=X[:, 1],
                hue=Y,
                palette=cmap_bold,
                alpha=1.0,
                edgecolor="black",
                s=25,
            )
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title(
                "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
            )
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.show()
            # for i in range(np.size(Y)):
            #     if Y[i] == 1: 
            #         plt.scatter(var1_data[i], var2_data[i], c='cyan')
            #     else:
            #         plt.scatter(var1_data[i], var2_data[i], c='magenta')
            '''