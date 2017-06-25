from django.conf import settings

from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage


def get_clarifai_api():
    return ClarifaiApp(settings.CLARIFAI_CLIENT_ID,
                       settings.CLARIFAI_CLIENT_SECRET)


def get_concept_scores(data):
    concepts = {}
    for concept in data['outputs'][0]['data']['concepts']:
        concepts[concept['name']] = concept['value']
    return concepts


def predict_image_bias(url):
    app = get_clarifai_api()
    model = app.models.get('news-photos')
    image = ClImage(url=url)
    data = model.predict([image])
    return get_concept_scores(data)
