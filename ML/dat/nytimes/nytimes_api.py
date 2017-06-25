import json
import requests
import time

API_KEY = 'c95d6d1cb0174dfeaaa259afbf8d0057'

BASE_URL = 'http://api.nytimes.com/svc/{area}/v2/'
TOP_STORIES_URL = BASE_URL.format(area='topstories') + '{section}.json'
ARTICLE_SEARCH_URL = BASE_URL.format(area='search') + 'articlesearch.json'
PAGE_SIZE = 10

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


def get_params():
    return {'api-key': API_KEY}


def article_search(news_desk, num_results=10):
    url = ARTICLE_SEARCH_URL
    filter_val = 'news_desk:("{}") AND document_type:article'.format(news_desk)
    params = get_params()
    params.update({'fq': filter_val})

    results = []
    has_results = True
    page_index = 0

    while has_results:
        page_index += 1
        params.update({'page': page_index})
        print('hitting search api', news_desk, page_index)
        res = requests.get(url, params=params)
        if res.status_code != 200:  # might be rate-limited. max 1 call/sec
            page_index -= 1
            continue

        res = res.json()

        # Append results
        results.extend(res['response']['docs'])

        # Decide if we need more data
        pages = res['response']['meta']['hits']
        print('Total pages:', pages)
        exceeded_pages = pages == page_index
        exceeded_num_results = page_index >= num_results / PAGE_SIZE
        has_results = not (exceeded_pages or exceeded_num_results)

        time.sleep(1)

    return results


def get_top_stories(section):
    url = TOP_STORIES_URL.format(section=section)
    res = requests.get(url, params={'api-key': API_KEY})
    print(res.json())


if __name__ == '__main__':
    f = open('data.json', 'w')
    sections = ['U.S.', 'Politics']

    results = []
    for section in sections:
        r = article_search(section, num_results=500)
        results.extend(r)

    json.dump(results, f)
    f.close()
