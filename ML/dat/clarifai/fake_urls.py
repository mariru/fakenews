import csv

IN_FILE = 'fake.csv'
OUT_FILE = 'fake_photos.txt'


def get_urls(filename):
    urls = set()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            url = row['main_img_url']
            if url:
                urls.add(url)
    return list(urls)

csv.field_size_limit(200000)
f = open('fake_photos.txt', 'w')
urls = get_urls(IN_FILE)
f.write('\n'.join(urls))
f.close()
