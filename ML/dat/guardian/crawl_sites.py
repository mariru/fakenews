import json
import logging

from gevent.pool import Pool

import justext
import requests

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT_SECONDS = 5

def get_content(url):
    """ Use justext to get only non-boilerplate text from url. """
    print('crawling', url)
    try:
        response = requests.get(url, allow_redirects=True,
                                timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as e:
        print(e)
        return ''
    if response.status_code != 200:
        print('bad code', response.status_code)
        return ''
    try:
        paragraphs = justext.justext(response.content,
                                     justext.get_stoplist('English'))
    except Exception as e:
        logger.warning('Error calling justext: %s', url, exc_info=True)
        print(e)
        return ''
    not_boiler = '\n'.join([i.text for i in paragraphs
                            if not i.is_boilerplate])
    return not_boiler



if __name__ == '__main__':
    f = open('data2.json')
    articles = json.load(f)

    out_f = open('fulltext.json', 'w')
    crawled_urls = set()

    def extract_text(article):
        url = article['webUrl']
        if url in crawled_urls:
            return

        crawled_urls.add(url)
        text = get_content(url)
    
        article['fullText'] = text
        out_f.write(json.dumps(article))
        out_f.write('\n')
        out_f.flush()

    pool = Pool(size=5)
    pool.map(extract_text, articles)
    out_f.close()
