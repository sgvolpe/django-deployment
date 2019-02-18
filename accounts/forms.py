from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm
from django.db import models


class UserCreateForm(UserCreationForm):
    class Meta:
        fields = ("username", "email", "password1", "password2", )
        model = get_user_model()
        profile_pic = models.ImageField(upload_to='basic_app/profile_pics',blank=True)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].label = "Display name"
        self.fields["email"].label = "Email address"
