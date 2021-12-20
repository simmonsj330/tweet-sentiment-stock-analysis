import matplotlib.pyplot as plt
import pandas as pd

def graph_likes(v):
    datelist = []
    avglist = []
    for i in v:
        total = 0
        count = 0
        date1 = i[0]
        date1 = date1[0:10]
        if date1 in datelist: continue
        else: datelist.append(date1)
        for j in v:
            date2 = j[0]
            date2 = date2[0:10]
            if date2 != date1: continue
            total += j[1]
            count += 1
        avglist.append(total/count)

    plt.scatter(datelist, avglist, s=0.20, c='r')
    plt.title("Average Favorite Count Over Time")
    plt.xticks([0,144,288,432,576,720,864,1008],["2014","2015","2016","2017","2018","2019","2020","2021"])
    plt.xlabel("Days")
    plt.ylabel("Average Number of Favorites")
    plt.savefig("graphLikes.jpeg")
    plt.close()
    return

def graph_rts(v):
    datelist = []
    avglist = []
    for i in v:
        total = 0
        count = 0
        date1 = i[0]
        date1 = date1[0:10]
        if date1 in datelist: continue
        else: datelist.append(date1)
        for j in v:
            date2 = j[0]
            date2 = date2[0:10]
            if date2 != date1: continue
            total += j[2]
            count += 1
        avglist.append(total/count)

    plt.scatter(datelist, avglist, s=0.20, c='r')
    plt.title("Average Retweet Count Over Time")
    plt.xticks([0,144,288,432,576,720,864,1008],["2014","2015","2016","2017","2018","2019","2020","2021"])
    plt.xlabel("Days")
    plt.ylabel("Average Number of Retweets")
    plt.savefig("graphRts.jpeg")
    plt.close()
    return

def graph_length(v):
    datelist = []
    avglist = []
    for i in v:
        total = 0
        count = 0
        date1 = i[0]
        date1 = date1[0:10]
        if date1 in datelist: continue
        else: datelist.append(date1)
        for j in v:
            date2 = j[0]
            date2 = date2[0:10]
            if date2 != date1: continue
            total += len(j[3])
            count += 1
        avglist.append(total/count)

    plt.scatter(datelist, avglist, s=0.20, c='r')
    plt.title("Average Length of Tweets Over Time")
    plt.xticks([0,144,288,432,576,720,864,1008],["2014","2015","2016","2017","2018","2019","2020","2021"])
    plt.xlabel("Days")
    plt.ylabel("Average Length of Tweets")
    plt.savefig("graphLength.jpeg")
    plt.close()
    return

def graph_sent(v):
    datelist = []
    avglist = []
    for i in v:
        total = 0
        count = 0
        date1 = i[0]
        date1 = date1[0:10]
        if date1 in datelist: continue
        else: datelist.append(date1)
        for j in v:
            date2 = j[0]
            date2 = date2[0:10]
            if date2 != date1: continue
            total += j[4]
            count += 1
        avglist.append(total/count)

    plt.scatter(datelist, avglist, s=0.20, c='r')
    plt.title("Average Sentiment Over Time")
    plt.xticks([0,144,288,432,576,720,864,1008],["2014","2015","2016","2017","2018","2019","2020","2021"])
    plt.xlabel("Days")
    plt.ylabel("Average Sentiment")
    plt.savefig("graphSent.jpeg")
    plt.close()
    return

csvdata = pd.read_csv('with_sentiment.csv')
v = csvdata.to_numpy()

graph_likes(v[:,1:])
graph_rts(v[:,1:])
graph_length(v[:,1:])
graph_sent(v[:,1:])



