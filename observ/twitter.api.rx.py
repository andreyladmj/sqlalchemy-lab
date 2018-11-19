# https://www.safaribooksonline.com/videos/reactive-python-for/9781491979006/9781491979006-video294997?autoplay=false

def tweets_for(topics):
    def observe_tweets(observer):
        class TweeterListener(StreamListener):
            def on_data(self, data):
                pass