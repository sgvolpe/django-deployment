from django.views.generic import TemplateView
from django.template import Template
from django.shortcuts import render

class HomePage(TemplateView):
    template_name = 'index.html'


class TestPage(TemplateView):
    template_name = 'test.html'

class ThanksPage(TemplateView):
    template_name = 'thanks.html'


############################
def other(request):
    template_name = 'other.html'
    print ( 'Other' )
    return render(request,'other.html')

def date_generator_form(request):
    template_name = 'date_generator_form.html'
    print ( 'Other' )
    return render(request,'date_generator_form.html')
