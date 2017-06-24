import requests

API_KEY = 'c95d6d1cb0174dfeaaa259afbf8d0057'

BASE_URL = 'http://api.nytimes.com/svc/{area}/v2/'
TOP_STORIES_URL = BASE_URL.format(area='topstories') + '{section}.json'
ARTICLE_SEARCH_URL = BASE_URL.format(area='search') + 'articlesearch.json'

TOP_STORIES_SECTIONS = [
    'home',
    'opinion',
    'world',
    'national',
    'politics',
    'upshot',
    'nyregion',
    'business',
    'technology',
    'science',
    'health',
    'sports'
]


# http://api.nytimes.com/svc/search/v2/articlesearch.json?q=new+york+times&page=2&sort=oldest&api-key=####

# http://api.nytimes.com/svc/topstories/v2/{section}.{response-format}?api-key={api-key}

def article_search(query):
    url = ARTICLE_SEARCH_URL
    res = requests.get(url, params={'api-key': API_KEY,
                                    'q': query,
                                    'document_type': 'article'})
    return res.json()



def get_top_stories(section):
    url = TOP_STORIES_URL.format(section=section)
    res = requests.get(url, params={'api-key': API_KEY})
    print(res.json())


if __name__ == '__main__':
    article_search('trump')
