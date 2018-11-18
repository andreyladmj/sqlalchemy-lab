import json

from rx import Observable

# https://www.safaribooksonline.com/videos/reactive-python-for/9781491979006/9781491979006-video294985
def tweets_for(topics):

    def observe_tweets(observer):
        class TweetListener(StreamListener):
            def on_data(self, data):
                observer.on_next(data)
                return True

            def on_error(self, status):
                observer.on_error(status)


        i = TweetListener()
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        stream = Stream(auith, 1)
        stream.filter(track=topics)

    return Observable.create(observe_tweets).share()

topics = ['Britian', 'France']

tweets_for(topics).map(lambda d: json.loads(d))\
    .filter(lambda map: 'text' in map)\
    .map(lambda map: map['text'].strip())\
    .subscribe(lambda s: print(s))