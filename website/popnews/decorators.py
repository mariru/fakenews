import uuid
from functools import wraps
from django.contrib.auth import login
from django.contrib.auth.models import User


def autologin(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.user.is_anonymous:
            # Create new user account and login.
            user = User.objects.create(username=str(uuid.uuid1()))
            login(request, user)
        return view_func(request, *args, **kwargs)
    return wrapper
