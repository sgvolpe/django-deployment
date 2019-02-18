


import os, datetime, json
#import networkx as nx
import pandas as pd

from django.db import models
from django.conf import settings
from django.core.validators import RegexValidator
from django.urls import reverse
from django.utils import timezone
from django import forms




DEBUG=True


class BFM_Parse(models.Model):
    bfm_rs_file = models.FileField(blank=False)
    #bfm_rs_df = pd.DataFrame()#models.CharField(max_length=255, blank=True)

    def parse(self):
        bfmrs_json = BargainFinderMaxRQ.bfm_from_file(self.bfm_rs_file)
        #self.bfm_rs_df = bfmrs_json# BargainFinderMaxRQ.bfm_rs_to_df(bfmrs_json, RET='ROCK')
        return bfmrs_json #BargainFinderMaxRQ.bfm_rs_to_df(bfmrs_json, RET='ROCK')


    def get_absolute_url(self):
        return reverse("new_dash:BFM_ParseDetail",kwargs={"pk": self.pk})


class StatelessApp(models.Model):
    '''
    A stateless Dash app.

    An instance of this model represents a dash app without any specific state
    '''

    app_name = models.CharField(max_length=100, blank=False, null=False, unique=True)
    slug = models.SlugField(max_length=110, unique=True, blank=True)

    def as_dash_app(self):
        '''
        Return a DjangoDash instance of the dash application
        '''
        app = DjangoDash('dash2', external_stylesheets='https://codepen.io/amyoshino/pen/jzXypZ.css',serve_locally=False)
        #app.append_css({'external_stylesheets':'https://codepen.io/amyoshino/pen/jzXypZ.css'})
        app.layout = html.Div([
            html.Div([],className='row', id="header"),

            html.Div([
                html.Div([html.H4('Itinerary Table')],className='six columns'),
                html.Div([html.H4('Itinerary Table')],className='six columns'),
                html.Div(["a"],className='six.columns'),
                html.Div(["a"],className='six.columns'),
            ],className='row justify-content-md-center', id="middle"),

            html.Div([],className='row', id="footer"),


        ] ,className='ten columns' )
        return app


class DashApp(models.Model):
    '''
    An instance of this model represents a Dash application and its internal state
    '''
    stateless_app = models.ForeignKey(StatelessApp, on_delete=models.PROTECT,
                                      unique=False, null=False, blank=False)
    instance_name = models.CharField(max_length=100, unique=True, blank=True, null=False)
    slug = models.SlugField(max_length=110, unique=True, blank=True)
    base_state = models.TextField(null=False, default="{}")
    creation = models.DateTimeField(auto_now_add=True)
    update = models.DateTimeField(auto_now=True)
    save_on_change = models.BooleanField(null=False,default=False)



    def current_state(self):
        '''
        Return the current internal state of the model instance
        '''

    def update_current_state(self, wid, key, value):
        '''
        Update the current internal state, ignorning non-tracked objects
        '''

    def populate_values(self):
        '''
        Add values from the underlying dash layout configuration
        '''
