import json

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from popnews.models import ArticleSave


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

@csrf_exempt
@login_required
def save_article(request):
    user = request.user
    data = json.loads(request.body.decode())
    url = data.get('url')
    if not url:
        return HttpResponseBadRequest('Missing url')

    # Clean url
    url = clean_url(url)

    return JsonResponse({'bias': -10})

@login_required
def user_stats(request):
    user = request.user
    articles_saved = ArticleSave.objects.filter(user=user)
    return JsonResponse({'articles': len(articles_saved)})
