import json

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from popnews.helpers import clean_url
from popnews.models import Article
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

    # Classify url
    # TODO(maja)

    # Save url
    article, _ = Article.objects.get_or_create(url=url)
    article_save, _ = ArticleSave.objects.get_or_create(user=user,
                                                        article=article)
    return JsonResponse({'bias': -10})


@login_required
def user_stats(request):
    user = request.user
    articles_saved = ArticleSave.objects.filter(user=user).values('article__url')
    return JsonResponse({'articles': list(articles_saved)})


@login_required
def test_save(request):
    return render(request, 'popnews/test_save.html')
