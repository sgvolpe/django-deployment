
from __future__ import unicode_literals

import base64 ,datetime, io, json,plotly, dash, os, re, io, urllib
from datetime import datetime
from collections import OrderedDict

from braces.views import SelectRelatedMixin
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http.response import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views import generic
from django.views.generic import (TemplateView, ListView)


from . import assistant as AS
from . import BargainFinderMaxRQ
from . import models
from . models import StatelessApp, BFM_Parse
#from . import dashboard_app
from . import dashboard_app0



import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



class CreateBFM_Parse(LoginRequiredMixin,SelectRelatedMixin,generic.CreateView):
    model = models.BFM_Parse
    template_name = "BFM_Parse/BFM_Parse_form.html"
    fields = ('bfm_rs_file',)

    #def form_valid(self, form):

    #    self.object = form.save(commit=False)
    #    self.object.save()
    #    self.object.bfm_rs_df = parse()
    #    self.object.save()
    #    return super().form_valid(form)



class BFM_ParseDetail(SelectRelatedMixin, generic.DetailView):
    model = models.BFM_Parse
    select_related = ()
    template_name = "BFM_Parse/BFM_Parse_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        print ('----')

        #context['app'] = dashboard_app.getapp(df_path)
        bfm_rs_json = self.object.parse()
        context['app'] = dashboard_app0.getitin('getitin', BargainFinderMaxRQ.bfm_from_file(self.object.bfm_rs_file))
        context['bfm_rs'] =bfm_rs_json
        return context
