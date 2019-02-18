
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from django.conf.urls import url, include
from django.conf import settings
from django.conf.urls.static import static


app_name = 'new_dash'



urlpatterns = [
    #path('', views.index, name='index'),

    #
    #path('dash2', views.dash2, name='dash2'),
    path('django_plotly_dash/', include('django_plotly_dash.urls'),name='the_django_plotly_dash'),



    #Parse BFMRQ
    path('CreateBFM_Parse', views.CreateBFM_Parse.as_view(), name='CreateBFM_Parse'),
    path('BFM_ParseDetail/<int:pk>/', views.BFM_ParseDetail.as_view(), name='BFM_ParseDetail'),

]
