import pandas as pd
from pandas import DataFrame
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    print('Hello')


    pos_neg_list = []
    data = pd.read_csv('with_financials.csv')

    for i in range(len(data['change'])):
        if(data['change'][i] > 0):
            pos_neg_list.append(1)
        else:
            pos_neg_list.append(0)

    data['trade'] = pos_neg_list
    
    data = data.drop(columns=['id', 'created_at', 'text', 'change'])


    y = data['trade']
    X = data.drop(columns=['trade'])
    print(X)
    print(X.shape)


    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.25)

    model = KNeighborsClassifier(n_neighbors = 2)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    print(classification_report(Y_test, predictions))