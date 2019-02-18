from django.db import models
from django.contrib import auth
# Create your models here.

class User(auth.models.User, auth.models.PermissionsMixin):
    profile_pic = models.ImageField(upload_to='accounts/static/profile_pics',blank=True)
    def __str__(self):
        return "@{}".format(self.username)


   # pip install pillow to use this!
    # Optional: pip install pillow --global-option="build_ext" --global-option="--disable-jpeg"
