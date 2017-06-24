import json
import requests

API_KEY = 'eec9a6ff-3377-4819-b319-9f52eb494c60'

BASE_URL = 'https://content.guardianapis.com/'
SEARCH_URL = BASE_URL + 'search'
SECTIONS_URL = BASE_URL + 'sections'
PAGE_SIZE = 50

SECTIONS = ['artanddesign', 'australia-news', 'better-business', 'books', 'business', 'cardiff', 'childrens-books-site', 'cities', 'commentisfree', 'community', 'crosswords', 'culture', 'culture-network', 'culture-professionals-network', 'edinburgh', 'education', 'enterprise-network', 'environment', 'extra', 'fashion', 'film', 'football', 'global-development', 'global-development-professionals-network', 'government-computing-network', 'guardian-professional', 'healthcare-network', 'help', 'higher-education-network', 'housing-network', 'inequality', 'info', 'jobsadvice', 'katine', 'law', 'leeds', 'lifeandstyle', 'local', 'local-government-network', 'media', 'media-network', 'membership', 'money', 'music', 'news', 'politics', 'public-leaders-network', 'science', 'search', 'small-business-network', 'social-care-network', 'social-enterprise-network', 'society', 'society-professionals', 'sport', 'stage', 'teacher-network', 'technology', 'theguardian', 'theobserver', 'travel', 'travel/offers', 'tv-and-radio', 'uk-news', 'us-news', 'voluntary-sector-network', 'weather', 'women-in-leadership', 'world']

def get_params():
 return {'api-key': API_KEY,
         'page-size': PAGE_SIZE,
         'format': 'json'}


def content_search(section, num_results=50, query=None):
    params = get_params()

    results = []

    has_results = True
    page_index = 0

    while has_results:
        page_index += 1
        params.update({'page': page_index, 'section': section})
        if query:
            params.update({'q': query})
        print('hitting search api', section, page_index)
        res = requests.get(SEARCH_URL, params=params)
        res = res.json()

        # Append results
        results.extend(res['response']['results'])

        # Decide if we need more data
        pages = res['response']['pages']
        print('Total pages:', pages)
        exceeded_pages = pages == page_index
        exceeded_num_results = page_index >= num_results / 50
        has_results = not (exceeded_pages or exceeded_num_results)

    return results


def get_sections():
    params = get_params()
    print('hitting sections api')
    res = requests.get(SECTIONS_URL, params=params)
    res = res.json()
    return res


if __name__ == '__main__':
    f = open('data2.json', 'w')
    sections = ['us-news', 'politics']

    results = []
    for section in sections:
        r = content_search(section, num_results=10000)
        results.extend(r)

    json.dump(results, f)
    f.close()
