import logging

from bs4 import BeautifulSoup
from urllib import parse

import justext
import requests

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT_SECONDS = 5


def clean_url(url):
    # Remove query params from url.
    p = parse.urlparse(url)
    return parse.urlunparse((p.scheme, p.netloc, p.path, '', '', ''))


def extract_text(content):
    try:
        paragraphs = justext.justext(content,
                                     justext.get_stoplist('English'))
    except Exception as e:
        logger.warning('Error calling justext', exc_info=True)
        print(e)
        return ''
    not_boiler = '\n'.join([i.text for i in paragraphs
                            if not i.is_boilerplate])
    return not_boiler


def extract_image(content):
    soup = BeautifulSoup(content)
    image_tag = soup.find('meta', property='og:image')
    if image_tag:
        return image_tag.get('content', '')
    else:
        return None


def extract_text_and_image(url):
    """ Use justext to get only non-boilerplate text from url. """
    print('crawling', url)
    try:
        response = requests.get(url, allow_redirects=True,
                                timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as e:
        print(e)
        return '', ''
    if response.status_code != 200:
        print('bad code', response.status_code)
        return '', ''

    text = extract_text(response.content)
    image = extract_image(response.content)
    return text, image
