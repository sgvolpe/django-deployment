
import dash, random, plotly
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import plotly.graph_objs as go

from . import assistant as AS
from plotly.graph_objs import *
from django_plotly_dash import DjangoDash
from dash.dependencies import Input, Output, State

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
from sklearn import preprocessing
#from . import BargainFinderMaxRQ

def dict_to_table(dic):
    k = list( dic.keys() )
    v = list( dic.values() )
    return html.Table(
        [html.Tr([html.Th('LFE Metric')]+[html.Th('Value')])] +
        [
            html.Tr([
                html.Td(k[i]), html.Td(v[i])
            ])
            for i in range(len(k))
        ]
    )

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th('col')]+[html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([html.Td(dataframe.index[i])] + [
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

DEBUG = True
def getitin(x , bfm_rs_json):
    print ('==================YOU ROCK \/ I PARSE==================')
    app = DjangoDash('getitin')
    app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
    external_js =[]# ["https://code.jquery.com/jquery-3.2.1.min.js","https://codepen.io/bcd/pen/YaXojL.js"]
    for js in external_js:
        app.scripts.append_script({"external_url": js})
    prices,itinerary_times = [],[]
    body = bfm_rs_json
    OriginDestinationOptions = body['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary']
    if DEBUG: print('1')
    OriginDestinationOptions_list = [(OriginDestinationOptions, 'RT')]
    #Multi-Ticket Treatment SOW
    if 'OneWayItineraries' in   body['OTA_AirLowFareSearchRS']:
        if DEBUG: print('OneWayItineraries')
        one_ways = body['OTA_AirLowFareSearchRS']['OneWayItineraries']
        [OneWay_ob, OneWay_ib] = one_ways['SimpleOneWayItineraries']
        if OneWay_ob is not None and OneWay_ib is not None:
            OriginDestinationOptions_list += [ (OneWay_ob['PricedItinerary'], 'OB'),(OneWay_ib['PricedItinerary'], 'IB')]


            if DEBUG: print (f'Number of Bounds: {len(OriginDestinationOptions_list )}')

    for leg_OriginDestinationOptions in OriginDestinationOptions_list:
        (OriginDestinationOptions, bound) = leg_OriginDestinationOptions
        if DEBUG: print (f'Reading Bound: {bound}')
        for option in OriginDestinationOptions:
            try: AdditionalFares = option['TPA_Extensions']['AdditionalFares']
            except: AdditionalFares = []
            options = option['AirItineraryPricingInfo'] + [fare['AirItineraryPricingInfo'] for fare in AdditionalFares]

            prices.append( min( [AirItineraryPricingInfo['ItinTotalFare']['BaseFare']['Amount']
                            for AirItineraryPricingInfo in options ] ))

            legs = option['AirItinerary']['OriginDestinationOptions']['OriginDestinationOption']
            times = [ leg['ElapsedTime'] for leg in legs]
            itinerary_times.append(sum(times))
    print (itinerary_times)
    print (prices)
    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }
    app.layout = html.Div([
        dcc.Graph(
            id='basic-interactions',
            figure={
                'data': [
                    {
                        'x': prices,
                        'y': itinerary_times,
                        'text': ['a', 'b', 'c', 'd'],
                        'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                        'name': 'Trace 1',
                        'mode': 'markers',
                        'marker': {'size': 12}
                    }
                ],
                'layout': {
                    'clickmode': 'event+select'
                }
            }
        ),


    ])
