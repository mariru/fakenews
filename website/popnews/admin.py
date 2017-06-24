from django.contrib import admin

from popnews.models import Article
from popnews.models import ArticleSave
from popnews.models import UserProfile

admin.site.register(Article)

@admin.register(ArticleSave)
class ArticleSaveAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'article', 'datetime')

admin.site.register(UserProfile)
