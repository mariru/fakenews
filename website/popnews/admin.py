from django.contrib import admin

from popnews.models import Article
from popnews.models import UserProfile

admin.site.register(Article)
admin.site.register(UserProfile)
