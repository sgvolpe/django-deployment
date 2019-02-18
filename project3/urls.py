"""project3 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from . import views


#if setting.DEBUG:
    #import debug_toolbar
    #TODO:
urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.HomePage.as_view(), name='index'),

    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('accounts/', include('django.contrib.auth.urls')),
    path(r'test/', views.TestPage.as_view(), name='test'),
    path(r'thanks/', views.ThanksPage.as_view(), name='thanks'),


    path(r'other', views.other, name='other'),

    
    path('new_dash/', include('new_dash.urls', namespace='new_dash')),
    path('django_plotly_dash/', include('django_plotly_dash.urls'),name='the_django_plotly_dash'),
]
