from TwitterTools import TwitterTools


# Tweets to pull
numberToPull = 10

# Companies of interest
companies = {
    "Wendys": "@Wendys",
    "Mcdonalds": "@McDonalds",
    "BurgerKing": "@BurgerKing",
    "TacoBell": "@tacobell"}

people = {"person": "@Person"}

# TwitterInterface Objects to pull from
queryList = []

# Raw Tweet Objects
rawTweets = []


# List companies to search
def companiesToSearch():
    for name, screen_name in companies.items():
        queryList.append(TwitterTools(numberToPull, screen_name, name + ".txt"))


# List people to search
def peopleToSearch():
    for name, screen_name in people:
        queryList.append(TwitterTools(numberToPull, screen_name, name + ".txt"))


# Execute queries contained in list
def collectData():
    for query in queryList:
        raw = query.searchTweets()
        query.filterTweets(raw)
        rawTweets.append(raw)


peopleToSearch()
collectData()
