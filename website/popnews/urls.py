from django.conf.urls import url

from popnews import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'save', views.save_article, name='save_article'),
    url(r'stats', views.user_stats, name='user_stats'),
    url(r'test', views.test_save, name='test_save')
]
