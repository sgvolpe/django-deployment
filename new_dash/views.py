
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
from . models import StatelessApp, BFM, several_same_BFM, OceanAnalysis, VirtualInterlining, BFM_Parse
from . import dashboard_app
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





class several_same_BFMListView(ListView):
    model = several_same_BFM
    template_name = "several_same_BFM/several_same_BFM_list.html" #"new_dash/several_same_BFM/several_same_BFM_list.html"
    select_related = ()
    #context['app'] = dashboard_app.getapp(str(context['shoppingcomparison']))
    def get_queryset(self):
        return several_same_BFM.objects.all()


class CreateSeveral_same_BFM(LoginRequiredMixin,SelectRelatedMixin,generic.CreateView):
    model = models.several_same_BFM
    template_name = "several_same_BFM/several_same_BFM_form.html"
    fields = ("title", 'description','ap','los','onds','repeats','bfm_template_1')

    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.author = self.request.user
        self.object.save()
        self.object.send_rq()
        return super().form_valid(form)


class several_same_BFMDetail(SelectRelatedMixin, generic.DetailView):
    model = models.several_same_BFM
    select_related = ()
    template_name = "several_same_BFM/several_same_BFM_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        #print ('----')
        #df_path = 'dashboard/static/bfm/'+str(context['bfm'].pk) + '/df.csv'
        #print (df_path)
        #print (context['bfm'].df_path)
        #context['app'] = dashboard_app.getapp(df_path)
        return context



def process_summary(request, pk):
    several_same_BFM  = get_object_or_404(models.several_same_BFM, pk=pk)
    several_same_BFM.process_summary()
    #BFM_obj = AS.bfm_rs(file_path=bfm.bfm_rs_file)
    #if DEBUG: print ('Trying to decompress your BFM')
    return HttpResponse('Ok.')

def send_again(request, pk):
    several_same_BFM  = get_object_or_404(models.several_same_BFM, pk=pk)
    several_same_BFM.send_rq()
    #BFM_obj = AS.bfm_rs(file_path=bfm.bfm_rs_file)
    #if DEBUG: print ('Trying to decompress your BFM')
    return HttpResponse('Again sent: Ok.')

def parse_bfmrs_to_df(request, pk):
    several_same_BFM  = get_object_or_404(models.several_same_BFM, pk=pk)
    several_same_BFM.generate_dfs()
    return HttpResponse('Again sent: Ok.')

class Createfare_forecast(LoginRequiredMixin,SelectRelatedMixin,generic.CreateView):
    model = models.fare_forecast
    template_name = "fare_forecast/fare_forecast_form.html"
    fields = ("ond", 'departure_date','return_date','predict_date')

    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.author = self.request.user
        self.object.save()
        self.object.run()
        return super().form_valid(form)

def fare_forecast_run(request, pk):
    fare_forecast  = get_object_or_404(models.fare_forecast, pk=pk)
    fare_forecast.run()

class CreateBFM(LoginRequiredMixin,SelectRelatedMixin,generic.CreateView):
    model = models.BFM
    template_name = "bfm/bfm_form.html"
    fields = ("title", 'description','bfm_rq_txt')

    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.author = self.request.user
        self.object.save()
        self.object.send_rq()
        return super().form_valid(form)

class BFMDetail(SelectRelatedMixin, generic.DetailView):
    model = models.BFM
    select_related = ()
    template_name = "bfm/bfm_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        print ('----')
        df_path = 'dashboard/static/bfm/'+str(context['bfm'].pk) + '/df.csv'
        #print (df_path)
        #print (context['bfm'].df_path)
        #context['app'] = dashboard_app.getapp(df_path)
        return context


class BFMListView(ListView):
    model = BFM
    template_name = "bfm/bfm_list.html"
    select_related = ()



    #context['app'] = dashboard_app.getapp(str(context['shoppingcomparison']))
    def get_queryset(self):
        return BFM.objects.all()


def bfm_decompress_rs(request, pk):
    bfm  = get_object_or_404(BFM, pk=pk)
    #BFM_obj = AS.bfm_rs(file_path=bfm.bfm_rs_file)
    #if DEBUG: print ('Trying to decompress your BFM')
    return HttpResponse(bfm.decompress_rs() , content_type="application/xhtml+xml")

class CreateVirtualInterlining(LoginRequiredMixin,SelectRelatedMixin,generic.CreateView):
    model = models.VirtualInterlining
    template_name = "virtualinterlining/virtualinterlining_form.html"
    fields = ('ori','des','ddate','rdate','stayover', 'night_penaly')


    def form_valid(self, form):
        self.object = form.save(commit=False)
        #self.object.author = self.request.user
        self.object.save()
        self.object.run()
        return super().form_valid(form)


class CreateOceanAnalysis(LoginRequiredMixin,SelectRelatedMixin,generic.CreateView):
    model = models.OceanAnalysis
    template_name = "ocean_analysis/new_form.html"
    fields = ("title", 'description')

    def form_valid(self, form):
        self.object = form.save(commit=False)
        #self.object.author = self.request.user
        self.object.save()
        self.object.run()
        return super().form_valid(form)

class OceanAnalysisDetail(SelectRelatedMixin, generic.DetailView):
    model = models.OceanAnalysis
    select_related = ()
    template_name = "ocean_analysis/ocean_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        print ('----')
        #df_path = 'dashboard/static/bfm/'+str(context['bfm'].pk) + '/df.csv'
        #print (df_path)
        #print (context['bfm'].df_path)
        #context['app'] = dashboard_app.getapp(df_path)
        return context


def ocean_analysis_run(request, pk):
    OA  = get_object_or_404(OceanAnalysis, pk=pk)
    OA.run()
    SD = dashboard.models.SummaryDashboard(csv_file=OA.path+'/summary.csv',title='TEST')
    SD.save()

    #BFM_obj = AS.bfm_rs(file_path=bfm.bfm_rs_file)
    #if DEBUG: print ('Trying to decompress your BFM')

    return HttpResponse('TEST PENDING')
    return redirect(SD)
