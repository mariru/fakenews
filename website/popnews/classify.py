from django.conf import settings

from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

IMAGE_WEIGHT = 0.3

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
    concepts = get_concept_scores(data)

    if concepts['bias'] > concepts['neutral']:
        return 'bias', concepts['bias']
    else:
        return 'neutral', concepts['neutral']


def combine_text_and_image(bias, concept, concept_value):
    bias = bias - 0.5
    if concept == 'neutral':
        bias = bias * (1 - IMAGE_WEIGHT * concept_value)  # decrease bias.
    else:
        bias = bias * (1 + IMAGE_WEIGHT * concept_value)  # increase bias.
    return bias + 0.5
