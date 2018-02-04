import emoji
import twitter
import numpy as np

#  (3.3.1 supports 280 characters)


class TwitterTools(object):
    # Subclass to hold tweets
    class Tweet:
        def __init__(self, text, hashtags, label='unknown'):
            self.text = text
            self.hashtags = hashtags
            self.label = label

    def __init__(self, numberToPull, query, textFile):

        """
        :param numberToPull: Tweets to be saved
        :param query: Term to be searched
        :param textFile: Text file in which tweets are saved in form of RAWtextFile.txt and FilteredtextFile.txt
        """
        # Twitter API keys apps.twitter1.com
        CONSUMER_KEY = 'wvkgw5Uw8uHzbj9svzZRSxrib'
        CONSUMER_SECRET = 'CPvoQQ1OtxQdWLkSJB90dswkNISvR73myhhYrSG37OlIm5wcsm'
        ACCESS_TOKEN_KEY = '40386305-tMsOvDim0fVpLHRgeYeFgpwCh8CsWOxNsnOaRyHpA'
        ACCESS_TOKEN_SECRET = '9r9VooUTljgdreqj6F3cSeMBP9CD8UYE5HLqXEwOg6iHR'
        self.numberToPull = numberToPull
        self.query = query
        self.textFile = textFile
        self.rawTextFile = 'RAW' + textFile
        self.filteredTextFile = 'Filtered' + textFile
        # Initialize api class containing pull functions
        self.api = twitter.Api(consumer_key=CONSUMER_KEY,
                               consumer_secret=CONSUMER_SECRET,
                               access_token_key=ACCESS_TOKEN_KEY,
                               access_token_secret=ACCESS_TOKEN_SECRET,
                               tweet_mode='extended',
                               sleep_on_rate_limit=True)

    def searchTweets(self):
        """
        :return results: List of tweet objects and list of tweets texts that have been screen for triviality
        """
        #  w+ create & write - a+ append - r reads - w write
        file = open(self.rawTextFile, "w+", encoding='utf-8')
        # ID of last fetched tweet ID
        batchID = 0
        # Text of tweets that have been screened
        listOfTweets = []
        # Count of Tweets saved
        numTweets = 0
        # Tweet objects that have been screened
        results = []
        while numTweets < self.numberToPull:
            try:
                search = self.api.GetSearch(term=self.query, count=100, since_id=batchID, include_entities=True)
                for tweet in search:
                    if not self.is_trivial_tweet(tweet, listOfTweets):
                        results.append(tweet)
                        listOfTweets.append(tweet.full_text)
                        # print(tweet.full_text)
                        # print(
                        #    "********************************************************************************************************************************************\n")
                        #  ###### Dont need to write full tweet to file anymore ######
                        # file.write(tweet.full_text + '\n')
                        # file.write(
                        #    '********************************************************************************************************************************************\n')  # 180 stars for length of Tweet/Tweet Separator
                        numTweets += 1
                        print("Tweet #" + str(numTweets))
                    batchID = tweet.id
                print('Last BatchID: ', batchID)
            except Exception as ex:
                print(ex)
        file.close()
        return results, listOfTweets

    @staticmethod
    def is_trivial_tweet(tweet, list_of_tweets):
        """
            Test if the given tweet is trivial. To be nontrivial tweets should contain at least one real word
            and should not be a retweet
        :param list_of_tweets: list of all tweets that have been currently saved
        :param tweet: tweet object
        :return: true or false
        """
        if tweet.full_text[:3] != 'RT ' and tweet.full_text not in list_of_tweets and tweet.lang == 'en':
            urls = tweet.urls
            media = tweet.media
            users = tweet.user_mentions
            temp = tweet.full_text
            # Checks for array of url objects that contains urls to remove
            if urls:
                for url_obj in urls:
                    temp = temp.replace(url_obj.url, "URL")
                    # print(url_obj.url, "changed to URL")
            # Checks for array of media objects for various media urls to remove
            if media:
                for media_obj in media:
                    # print('media.url ' + media_obj.url)
                    temp = temp.replace(media_obj.url, "")
                    if media_obj.media_url:
                        temp = temp.replace(media_obj.media_url, "")
                    if media_obj.media_url_https:
                        temp = temp.replace(media_obj.media_url_https, "")
            # Checks for array of user mention objects that contain screen names to remove
            if users:
                for user in users:
                    # print('user.screen_name ' + user.screen_name)
                    temp = temp.replace('@' + user.screen_name, "")
            # Checks if ONLY emojis and or hyperlinks are contained within tweet text
            for character in temp:
                if character in emoji.UNICODE_EMOJI:
                    temp = temp.replace(character, "")
            # Checks for newline
            temp = temp.replace("\n", "")
            # Removes all whitespace
            temp = temp.replace(" ", "")
            # Check if tweet doesn't contain pure text character
            if temp == "":
                # print("Pointless Tweet")
                return True
            return False
        return True

    def filterTweets(self, rawTweets):
        """
            Takes in list of tweets objects and edits the fields so that they are more easily interpreted at vectors
            Converts data for use in the neural network
        :param rawTweets: Parsed tweets to be filtered
        :return
        """
        removeCharacters = ['\r', '\n', ',', '*', '"', '`', "'"]
        spaceCharacters = ['_', '-', ',', ':', ';', '(', ')', '[', ']', '{', '}', '/' ]
        file = open(self.filteredTextFile, "w+", encoding='utf-8')
        numTweets = 0
        for tweet in rawTweets:
            print(type(tweet))
            urls = tweet.urls
            media = tweet.media
            users = tweet.user_mentions
            hashtags = tweet.hashtags
            print("RAW TWEET:")
            print(tweet.full_text)
            filteredTweet = tweet.full_text
            filteredTweet = filteredTweet.replace(self.query, "COMPANY")  # Company
            if urls:
                for object in urls:
                    filteredTweet = filteredTweet.replace(object.url, "URL")
                    print(object.url, "changed to URL")
            if media:
                for object in media:
                    print('media.url ' + object.url)
                    filteredTweet = filteredTweet.replace(object.url, "")
                    if object.media_url:
                        filteredTweet = filteredTweet.replace(object.media_url, "")
                    if object.media_url_https:
                        filteredTweet = filteredTweet.replace(object.media_url_https, "")
            if users:
                count = 0
                for user in users:
                    print('user.screen_name ' + user.screen_name)
                    if count == 0:
                        filteredTweet = filteredTweet.replace('@' + user.screen_name, "SOMEONE")
                    if count > 0:
                        filteredTweet = filteredTweet.replace('@' + user.screen_name, "")
            if hashtags:
                for hashtag in hashtags:
                    print('hashtag ' + hashtag.text)
                    filteredTweet = filteredTweet.replace('#' + hashtag.text, "")
            for c in removeCharacters:
                filteredTweet = filteredTweet.replace(c, "")
            for c in spaceCharacters:
                filteredTweet = filteredTweet.replace(c, " ")
            filteredTweet = filteredTweet.replace("!", ".")
            filteredTweet = filteredTweet.replace("?", ".")
            filteredTweet = filteredTweet.replace("@", "at")
            filteredTweet = filteredTweet.replace("&amp", "and")
            filteredTweet = filteredTweet.replace("%", "percent")
            filteredTweet = filteredTweet.replace("$", "dollar")
            filteredTweet = filteredTweet.replace("+", "plus")

            for i, character in enumerate(filteredTweet):
                if character in emoji.UNICODE_EMOJI:
                    filteredTweet = filteredTweet.replace(character, " EMOJI ")
            # Remove adjacent duplicate words "SOMEONE"
            print(filteredTweet)
            words = filteredTweet.split()
            i = 0
            while i < len(words) - 1:
                if words[i + 1] == words[i] and words[i] == "SOMEONE":
                    words.pop(i)
                else:
                    i += 1
            # Remove adjacent duplicate words "EMOJI"
            i = 0
            while i < len(words) - 1:
                if words[i + 1] == words[i] and words[i] == "EMOJI":
                    words.pop(i)
                else:
                    i += 1
            filteredTweet = " ".join(words)
            # emoji.emojize can be used to convert the unicode into emoji description, (:grinning:) semicolons have to be removed though
            print("FILTERED TWEET:")
            print(filteredTweet)
            print(
                "********************************************************************************************************************************************\n")
            file.write(filteredTweet + '\n')
            file.write(
                '********************************************************************************************************************************************\n')
            numTweets += 1
            print("Tweet #" + str(numTweets))
        file.close()

    @staticmethod
    def convert_to_tensor(filtered_tweets):
        """
            converts a list of tweets to a 3D tensor with dimensions [number of tweets, charactrs, max binary per character]
        That is, we converte each character in the list of tweets into a binary vector
        for example a = [1, 0, 0, 0, 0]
        :param filtered_tweets:
        :return:
        """
        # Vocabulary convention: tensor == 3D, matrix == 2D, vector == 1D
        # Dimensions of our tensor
        tweet_size = 140
        num_tweets = len(filtered_tweets)
        binary_vector_size = 5

        # Tensor to hold all tweets
        tweet_tensor = np.array([num_tweets, tweet_size, binary_vector_size])
        i = 0
        for tweet in filtered_tweets:
            j = 0
            for character in tweet:
                value = 0
                # Spaces should be [0, 0, ..., 0]
                if character != ' ':
                    # Stores the ASCII value by 96 so that 'a' == 1
                    value = ord(character.lower()) - 96
                # Converts the character's value to binary and stores that value as a string of 1s and 0s
                binary_string = str("{0:b}".format(value))
                k = 0
                for b in binary_string:
                    # Stores the kth binary digit of the jth character or the ith tweet into our 3D tensor
                    tweet_tensor[i][j][k] = int(b)
                    k += 1
                j += 1
            i += 1
        return tweet_tensor


