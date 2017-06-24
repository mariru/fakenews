from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone


class UserProfile(models.Model):
    user = models.OneToOneField(User, related_name='profile')


class Article(models.Model):
    url = models.URLField()
    users = models.ManyToManyField(User, through='ArticleSave')


class ArticleSave(models.Model):
    user = models.ForeignKey(User)
    article = models.ForeignKey(Article)
    datetime = models.DateTimeField(default=timezone.now)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()
