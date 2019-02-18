
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

    path('fare_forecast', views.Createfare_forecast.as_view(), name='fare_forecast'),
    path('fare_forecast_run', views.fare_forecast_run, name='fare_forecast_run'),
    #BFM
    path('bfm_list', views.BFMListView.as_view(), name='bfm_list'),
    path('bfm_new', views.CreateBFM.as_view(), name='bfm_new'),
    path("bfm_view/<int:pk>/",views.BFMDetail.as_view(),name="bfm_view"),
    path("bfm_decompress_rs/<int:pk>", views.bfm_decompress_rs, name="bfm_decompress_rs"),

    #several_same_BFM
    path('several_same_BFM_new', views.CreateSeveral_same_BFM.as_view(), name='several_same_BFM_new'),
    path('several_same_BFMDetail/<int:pk>/', views.several_same_BFMDetail.as_view(), name='several_same_BFMDetail'),
    path('several_same_BFM_list', views.several_same_BFMListView.as_view(), name='several_same_BFM_list'),
    path('several_same_BFM/<int:pk>/process_summary', views.process_summary, name='process_summary'),
    path('several_same_BFM/<int:pk>/send_again', views.send_again, name='send_again'),
    path('several_same_BFM/<int:pk>/parse_bfmrs_to_df', views.parse_bfmrs_to_df, name='parse_bfmrs_to_df'),


    #OceanAnalysis
    path('ocean_analysis_new', views.CreateOceanAnalysis.as_view(), name='ocean_analysis_new'),
    path('ocean_analysis_view/<int:pk>/', views.OceanAnalysisDetail.as_view(), name='ocean_analysis_view'),
    #path("bfm_view/<int:pk>/",views.BFMDetail.as_view(),name="bfm_view"),
    path("ocean_analysis_run/<int:pk>", views.ocean_analysis_run, name="ocean_analysis_run"),


    #VirtualInterlining
    path('virtualinterlining_new', views.CreateVirtualInterlining.as_view(), name='virtualinterlining_new'),

    #Parse BFMRQ
    path('CreateBFM_Parse', views.CreateBFM_Parse.as_view(), name='CreateBFM_Parse'),
    path('BFM_ParseDetail/<int:pk>/', views.BFM_ParseDetail.as_view(), name='BFM_ParseDetail'),

]
