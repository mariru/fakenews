from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

CLIENT_ID = 'TcT6Ab1S0ynJfrUmRN7dWLcYoTmrDBvCRpNoubEG'
CLIENT_SECRET = 'srNpkyHd1iaX7CpXcoqgwVgB7wvvSVHR2XiJONF1'

def get_clarifai_api():
    return ClarifaiApp(CLIENT_ID, CLIENT_SECRET)


def bulk_upload(urls, concepts, not_concepts, app):
    # Bulk Import Images API
    images = []
    for url in urls:
        img = ClImage(url=url, concepts=concepts, not_concepts=not_concepts)
        images.append(img)

    app.inputs.bulk_create_images(images)

def train(fake_urls, not_fake_urls, app):
    bulk_upload(fake_urls[0:50], ['bias'], ['neutral'], app)
    bulk_upload(not_fake_urls[0:50], ['neutral'], ['bias'], app)
    model = app.models.get('news-photos')
    model.train()

def predict(url):
    app = get_clarifai_api()
    model = app.models.get('news-photos')
    image = ClImage(url=url)
    data = model.predict([image])
    return get_concept_scores(data)

def get_concept_scores(data):
    concepts = {}
    for concept in data['outputs'][0]['data']['concepts']:
        concepts[concept['name']] = concept['value']
    return concepts


if __name__ == '__main__':
    fake = open('fake_photos.txt')
    fake_urls = fake.readlines()
    fake.close()

    not_fake = open('not_fake_photos.txt')
    not_fake_urls = not_fake.readlines()
    not_fake.close()

    app = get_clarifai_api()
    # train(fake_urls, not_fake_urls, app)

    # Test
    test_fake_urls = fake_urls[51:60]
    test_not_fake_urls = not_fake_urls[51:60]

    model = app.models.get('news-photos')
    for url in test_fake_urls:
        image = ClImage(url=url)
        data = model.predict([image])
        print(url)
        print(get_concept_scores(data))

    for url in test_not_fake_urls:
        image = ClImage(url=url)
        data = model.predict([image])
        print(url)
        print(get_concept_scores(data))
