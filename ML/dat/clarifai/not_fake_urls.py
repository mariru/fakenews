import json

from urllib.parse import urljoin

IN_FILE = '../nytimes/data.json'
OUT_FILE = 'not_fake_photos.txt'

BASE_URL = 'http://www.nytimes.com'

def get_urls(filename):
    f = open(filename)
    data = json.load(f)
    print(len(data))

    urls = []
    for d in data:
        images = d['multimedia']
        for image in images:
            if image['subtype'] == 'xlarge':
                url = image['url']
                if not url.startswith('http'):
                    url = urljoin(BASE_URL, url)
                urls.append(url)
                break
    return urls

f = open('not_fake_photos.txt', 'w')
urls = get_urls(IN_FILE)
f.write('\n'.join(urls))
f.close()
