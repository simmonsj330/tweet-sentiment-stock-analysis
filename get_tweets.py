import tweepy
import sys

CONSUMER_KEY = "x"
CONSUMER_SECRET = "x"
ACCESS_KEY = "x"
ACCESS_SECRET = "x"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)

api = tweepy.API(auth)

all = []

for i in range(1, len(sys.argv)):
    tweets = api.user_timeline(screen_name=sys.argv[i], count=200, include_rts = False, tweet_mode = 'extended')
    all.extend(tweets)
    old_id = tweets[-1].id
    while True:
        tweets = api.user_timeline(screen_name=sys.argv[i], count=200, include_rts = False, max_id = old_id - 1, tweet_mode = 'extended')
        if len(tweets) == 0: break
        old_id = tweets[-1].id
        all.extend(tweets)
    print(sys.argv[i])

from pandas import DataFrame
outtweets = [[tweet.id_str, tweet.created_at, tweet.favorite_count, tweet.retweet_count, tweet.full_text.encode("utf-8").decode("utf-8")] 
            for idx,tweet in enumerate(all)]
df = DataFrame(outtweets,columns=["id","created_at","favorite_count","retweet_count", "text"])
df.to_csv('tweets.csv',index=False)
