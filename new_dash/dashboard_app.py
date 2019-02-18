
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

def getitin(x,bfm_rs_json):
    app = DjangoDash('ancillaries_app')
    app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
    external_js = ["https://code.jquery.com/jquery-3.2.1.min.js","https://codepen.io/bcd/pen/YaXojL.js"]
    for js in external_js:
        app.scripts.append_script({"external_url": js})

    app = DjangoDash('getitin')
    app.layout = html.Div(
        html.Div([
            html.Div([ dcc.Graph(id='results_table') ], className= 'four columns offset-by-one'),
            html.Div([ dcc.Graph(id='chart2') ], className= 'four columns offset-by-one')

            ])
        )


    @app.callback(dash.dependencies.Output('results_table', 'children'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_results_table(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]

        new_df = pd.DataFrame( [lfe_dict,quickest_dict,cxr_div_dict,alliance_div,response_time_win], index=['LFE','Quickest','CXR Div', 'Alliance Div','Response Time'] )

        return generate_table(new_df)

    @app.callback( dash.dependencies.Output('chart1', 'figure'),[Input('results_table', 'rows')] )
    def update_carriers_count_graph(rows):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Carrier Count per Provider',  ))
        return fig




    return app

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


def get_ancillaries(file_path=r'', ancillaries_object=None):
    app = DjangoDash('ancillaries_app')
    app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
    external_js = ["https://code.jquery.com/jquery-3.2.1.min.js","https://codepen.io/bcd/pen/YaXojL.js"]
    for js in external_js:
        app.scripts.append_script({"external_url": js})

    df = pd.read_csv(file_path) ## bookings df


    app.layout = html.Div(
        html.Div([
            html.Div([ dcc.Graph(id='carriers_count-graph') ], className= 'four columns offset-by-one'),
            html.Div([ dcc.Graph(id='bookings_count-graph') ], className= 'four columns offset-by-one'),
            html.Div([ dcc.Graph(id='carriers_coverage-graph') ], className= 'four columns offset-by-one'),
            html.Div([ dcc.Graph(id='bookings_coverage-graph') ], className= 'four columns offset-by-one'),
            html.Div([
                dt.DataTable( rows=df.to_dict('records'), row_selectable=True, filterable=True, sortable=True, selected_row_indices=[],
                id='datatable')
            ], className= 'ten columns offset-by-one')
        ])
    )

    @app.callback(dash.dependencies.Output('carriers_count-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_carriers_count_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Carrier Count per Provider',  ))
        df = ancillaries_object.get_carrier_count()
        p = ['sabre','travelport','amadeus'] #TODO: gget gds list auto
        c = ['red', 'green','blue']
        for x in range(len(p)):
            x_ = df[df['GDS'] == p[x] ]['ANC']
            y_ = df[df['GDS'] == p[x] ]['count']
            fig.append_trace({
                'x': x_,
                'y': y_,
                'type': 'bar',
                'opacity':0.5,
                'marker': {'color':c[x], 'line':{'width':2}},
                'name': p[x],
            }, 1, 1)

        return fig

    @app.callback(dash.dependencies.Output('bookings_count-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_bookings_count_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Bookings Count per Provider',  ))
        df = ancillaries_object.get_booking_count()
        p = ['sabre','travelport','amadeus'] #TODO: gget gds list auto
        c = ['red', 'green','blue']
        for x in range(len(p)):
            x_ = df[df['GDS'] == p[x] ]['ANC']
            y_ = df[df['GDS'] == p[x] ]['count']
            fig.append_trace({
                'x': x_,
                'y': y_,
                'type': 'bar',
                'opacity':0.5,
                'marker': {'color':c[x], 'line':{'width':2}},
                'name': p[x],
            }, 1, 1)

        return fig

    @app.callback(dash.dependencies.Output('carriers_coverage-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def carriers_coverage_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Carriers Percentage per Provider',  ))
        df = ancillaries_object.get_carrier_count(percentage=True)
        p = ['sabre','travelport','amadeus'] #TODO: gget gds list auto
        c = ['red', 'green','blue']
        for x in range(len(p)):
            x_ = df[df['GDS'] == p[x] ]['ANC']
            y_ = df[df['GDS'] == p[x] ]['count']
            fig.append_trace({
                'x': x_,
                'y': y_,
                'type': 'bar',
                'opacity':0.5,
                'marker': {'color':c[x], 'line':{'width':2}},
                'name': p[x],
            }, 1, 1)

        return fig

    @app.callback(dash.dependencies.Output('bookings_coverage-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def bookings_coverage_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Bookings Percentage per Provider',  ))
        df = ancillaries_object.get_booking_count(percentage=True)
        p = ['sabre','travelport','amadeus'] #TODO: gget gds list auto
        c = ['red', 'green','blue']
        for x in range(len(p)):
            x_ = df[df['GDS'] == p[x] ]['ANC']
            y_ = df[df['GDS'] == p[x] ]['count']
            fig.append_trace({
                'x': x_,
                'y': y_,
                'type': 'bar',
                'opacity':0.5,
                'marker': {'color':c[x], 'line':{'width':2}},
                'name': p[x],
            }, 1, 1)

        return fig


    if __name__ == '__main__':
            app.run_server(debug=True)

    return app



def get_summary_app(file_path=r'', providers = ['sabre','competitor']):
    app = DjangoDash('summary_app')
    app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
    external_js = ["https://code.jquery.com/jquery-3.2.1.min.js","https://codepen.io/bcd/pen/YaXojL.js"]
    for js in external_js:
        app.scripts.append_script({"external_url": js})

    df = pd.read_csv(file_path)
    p = [x+j for j in ['_cheapest','_cheapest','_only','_only','_time_min','_fare_mean','_fare_std'] for x in providers] # ,'payload_size','response_time'
    columns = ['itinerary_distance','ond','response_time_win','rq1_top50','rq2_top50','lfe','quickest','cxr_div','both', 'alliance_div','lfe_fare_difference','ob_date','ib_date','repeat']+p # 'search_date','ap_los',




    app.layout = html.Div(
        html.Div([
            html.Div([
                dt.DataTable( rows=df[columns].to_dict('records'), row_selectable=True, filterable=True, sortable=True, selected_row_indices=[],
                id='datatable')
            ], className= 'ten columns offset-by-one',),

            html.Div([dcc.Graph(id='lfe_cross-graph')], className= 'three columns offset-by-one', style={'height':'500px'}),
            html.Div([dcc.Graph(id='overlap-graph_across')], className= 'three columns offset-by-one', style={'height':'500px'}),
            html.Div([dcc.Graph(id='polar-graph')], className= 'three columns offset-by-one', style={'height':'500px'}),
            html.Div([dcc.Graph(id='lfe-quick-graph')], className= 'ten columns offset-by-one'),
            html.Div([dcc.Graph(id='fare_time-graph')], className= 'ten columns offset-by-one'),
            html.Div([dcc.Graph(id='overlap-graph')], className= 'four columns offset-by-one', style={'height':'500px'}),
            html.Div([dcc.Graph(id='overlap-ond-graph')], className= 'four columns offset-by-one', style={'height':'500px'}),
            html.Div(id='results_table', className= 'four columns'),
            html.Div(id='lfe_table', className= 'four columns'),

            #html.Div([dcc.Graph(id='quick-graph')], className= 'ten columns', style={'height':'500px'}),

        ],className='row')


    )


    @app.callback(dash.dependencies.Output('results_table', 'children'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_results_table(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        lfe_dict = dict((df['lfe'].value_counts()))
        quickest_dict = dict((df['quickest'].value_counts()))
        cxr_div_dict = dict((df['cxr_div'].value_counts()))
        alliance_div = dict((df['alliance_div'].value_counts()))
        response_time_win = dict((df['response_time_win'].value_counts()))

        new_df = pd.DataFrame( [lfe_dict,quickest_dict,cxr_div_dict,alliance_div,response_time_win], index=['LFE','Quickest','CXR Div', 'Alliance Div','Response Time'] )

        return generate_table(new_df)


    @app.callback(dash.dependencies.Output('lfe_table', 'children'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_lfe_table(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]



        return dict_to_table(dict( pd.DataFrame(rows)['lfe_fare_difference'].describe() ))




    @app.callback(dash.dependencies.Output('lfe_cross-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_lfe_cross_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Overlap',  ))
        fig.layout = go.Layout(barmode='stack')
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        lfe_dict = dict((df['lfe'].value_counts()))
        data = Data([go.Pie(values=list(lfe_dict.values()), labels=list(lfe_dict.keys()))])
        layout = go.Layout( title='LFE')
        return go.Figure(data=data,layout=layout)

    @app.callback(dash.dependencies.Output('overlap-graph_across', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_overlap_graph(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        p = ['both']+[p + '_only' for p in providers]
        c = ['green', 'red', 'blue']
        p_list = [df[x].sum() for x in p ]
        layout = go.Layout( title='Overlap Across')
        return go.Figure(data=[go.Pie(values=p_list, labels=p)], layout=layout)

    @app.callback(dash.dependencies.Output('polar-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_polar_graph(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        lfe_dict = dict((df['lfe'].value_counts()))
        quickest_dict = dict((df['quickest'].value_counts()))
        cxr_div_dict = dict((df['cxr_div'].value_counts()))
        alliance_div = dict((df['alliance_div'].value_counts()))
        response_time = dict((df['response_time_win'].value_counts()))

        provider_list = providers #['sabre', 'competitor']
        data = []
        for x in range(len(provider_list)):
            if provider_list[x] not in lfe_dict: lfe_dict[provider_list[x]] = 0
            if provider_list[x] not in quickest_dict: quickest_dict[provider_list[x]] = 0
            if provider_list[x] not in cxr_div_dict: cxr_div_dict[provider_list[x]] = 0
            if provider_list[x] not in alliance_div: alliance_div[provider_list[x]] = 0
            if provider_list[x] not in response_time: response_time[provider_list[x]] = 0
            data.append(
                go.Scatterpolar(
                  r = [ lfe_dict[provider_list[x]], quickest_dict[provider_list[x]], cxr_div_dict[provider_list[x]], response_time[provider_list[x]], alliance_div[provider_list[x]] ],
                  theta = ['lfe','quickest','cxr_div','response_time','alliance_div'],
                  fill = 'toself',
                  name = provider_list[x]
              )
            )
        layout = go.Layout(
                title = "Winner Metrics", #font = dict(size = 15),
                polar = dict( #bgcolor = "rgb(223, 223, 223)",
                  angularaxis = dict(
                    tickwidth = 2,
                    linewidth = 3,
                    layer = "below traces"
                  )
                )
        )
        return go.Figure(data=data,layout=layout)


    @app.callback(dash.dependencies.Output('fare_time-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_fare_time_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Fare vs Time',  ))
        fig.layout = go.Layout(barmode='stack',title = "Price vs Time",)
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)

        p = providers
        c = ['red', 'blue']
        for x in range(len(p)):
            fig.append_trace({
                'y': df[p[x]+'_cheapest' ],
                'x': df[p[x] + '_time_min' ],
                'type': 'scatter',
                'opacity':0.5,
                'mode':'markers',
                'marker': {'color':c[x],'size': 15, 'line':{'width':2}},
                'name': p[x],
                'orientation': 'h',
            }, 1, 1)
        return fig




    @app.callback(dash.dependencies.Output('overlap-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_overlap_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, shared_xaxes=True,shared_yaxes=False)
        fig.layout = go.Layout(barmode='stack',title='Overlap per Query')
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        p = ['both']+[p + '_only' for p in providers]
        c = ['green', 'red', 'blue']
        for x in range(len(p)):
            fig.append_trace({
                'y': df['ond'].astype(str)+df['ob_date']+df['ib_date'],#df['ap_los'].astype(str), #+df['search_date'].astype(str)
                'x': df[p[x]].astype(float) / (df['both'].astype(float)+df[p[1]].astype(float)+df[p[2]].astype(float)),
                'type': 'bar',
                'opacity':0.5,
                'marker': {'color':c[x],'line':{'color': c[x],'width':1.5}},
                'name': p[x],
                'orientation': 'h',
            }, 1, 1)

        return fig

    @app.callback(dash.dependencies.Output('overlap-ond-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_overlap_ond_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=2, subplot_titles=('Overlap per Query','Overlap per OnD'  ), shared_xaxes=True,shared_yaxes=False)
        fig.layout = go.Layout(barmode='stack', title='Overlap per Ond')
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        p = ['both']+[p + '_only' for p in providers]
        df = df.groupby(['ond'],as_index=True)[p].sum()

        c = ['green', 'red', 'blue']
        for x in range(len(p)):
            fig.append_trace({
                'y': df.index,
                'x': df[p[x]].astype(float)/(df['both'].astype(float)+df[p[1]].astype(float)+df[p[2]].astype(float)),
                'type': 'bar',
                'opacity':0.5,
                'marker': {'color':c[x],'line':{'color': c[x],'width':1.5}},
                'name': p[x],
                'orientation': 'h',
            }, 1, 1)


        return fig


    @app.callback(dash.dependencies.Output('lfe-quick-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_lfe_quick_graph(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=3, cols=1, subplot_titles=('Quickest Option', 'Cheapest Option', 'Fare Mean | STDEV' ), shared_xaxes=True,shared_yaxes=False)
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        provider_list = providers #['sabre', 'competitor']
        colors = ['red', 'blue']
        x_list = df['ond'].astype(str)+df['ob_date']+df['ib_date']
        for i in range(len(provider_list)):
            fig.append_trace({
                'y': df[ provider_list[i] +'_time_min'],
                'x': x_list,
                'type':'scatter',
                'name':provider_list[i]+'_time_min',
                'marker': {'color': colors[i]},
                'opacity':0.5,
                'orientation': 'h',
            } ,1, 1)

            fig.append_trace({
                'y': df[provider_list[i] + '_cheapest'],
                'x': x_list,
                'type':'bar',
                'name': provider_list[i]+ '_cheapest',
                'marker':{'color': colors[i]},
                'opacity':0.5,
            } ,2, 1)

            fig.append_trace({
                'x': x_list,
                'y': df[provider_list[i] + '_fare_mean'],
                'type':'scatter',
                'name': provider_list[i]+'Mean Fare | Stdev',
                'marker':{'color': colors[i]},
                'opacity':0.5,
                'error_y':{'type':'data', 'array':df[provider_list[i] + '_fare_std'],'visible':True,},
            } ,3, 1)

        return fig




    if __name__ == '__main__':
            app.run_server(debug=True)

    return app



def getapp(file_path=r''):
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
    app = DjangoDash('dash2')
    df = pd.read_csv(file_path)
    app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

    ond = df['market'][0]
    date = df['search_date'][0]
    ap_los = df['ap_los'][0]
    provider_count = dict((df['provider'].value_counts()))
    external_js = ["https://code.jquery.com/jquery-3.2.1.min.js","https://codepen.io/bcd/pen/YaXojL.js"]

    for js in external_js:
        app.scripts.append_script({"external_url": js})

    app.layout = html.Div(
        html.Div([
            html.Div([
                html.Div(children=[
                    html.H1(children=ond),
                    html.H3(children=date),
                    html.H4(children=str(provider_count)),
                    html.H5(children=str(ap_los)),
                    #generate_table(df['is_duplicate'].value_counts())


                ],className='row'),
                html.Div([
                    html.Div([
                        html.A(['Print PDF'],className="button no-print print",style={}),
                        html.H3('Choose Provider:'),
                        html.Div([
                        dcc.Checklist(id = 'Provider', values=['sabre', 'competitor','overlap'], labelStyle={'display': 'inline-block'},
                            options=[{'label': 'Sabre', 'value': 'sabre'},{'label': 'Competitor', 'value': 'competitor'},{'label': 'Overlap', 'value': 'overlap'}]
                        )],className='row'),
                    ],className='six columns'),
                    html.Div([dcc.Graph(id='time_vs_fare-graph')], className= 'twelve columns', style={'height':'500px'}),
                    html.Div([dcc.Graph(id='overlap-graph')], className= 'twelve columns'),
                    html.Div([dcc.Graph(id='fare-graph')], className= 'twelve columns'),

                ],className='row', style={'margin-top': '10'}),

            ], className="row"),

            html.Div(
                [
                html.Div([dcc.Graph(figure=go.Figure(),id='overlap_pie-graph')], className= 'six columns'),
                html.Div([dcc.Graph(id='violin-graph')], className= 'six columns'),
            ], className="ten columns"),

            html.Div([
                html.Div([
                        html.H3('Itinerary Table'),
                        dt.DataTable( rows=df.to_dict('records'), row_selectable=True, filterable=True, sortable=True, selected_row_indices=[],
                        id='datatable-gapminder')
                    ], className="twelve columns"),
            ], style={'width':'100%','font-size':'0.8em'})

        ], className='ten columns offset-by-one')
    )

    @app.callback(
        dash.dependencies.Output('violin-graph', 'figure'),
        [dash.dependencies.Input('Provider', 'values'),Input('datatable-gapminder', 'rows'), Input('datatable-gapminder', 'selected_row_indices')])
    def update_image_src2(selector,rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Fare Distribution by Provider',  ))

        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        df = df[ df['provider'].isin( selector ) ]

        providers = pd.unique(df['provider'])
        for provider in providers:
            fig.append_trace({
                "x": df['provider'][df['provider'] == provider],
                'y': df['fare'][df['provider'] == provider],
                "name": provider,
                'type': 'violin',
                "box": {"visible": True},
                "meanline": {"visible": True},
                #'color': {'competitor':'blue','sabre':'red'}[provider],
            }, 1, 1)
        return fig


    @app.callback(
        dash.dependencies.Output('fare-graph', 'figure'),[dash.dependencies.Input('Provider', 'values'),Input('datatable-gapminder', 'rows'), Input('datatable-gapminder', 'selected_row_indices')])
    def update_image_src(selector,rows, selected_row_indices):

        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Fare Comparison',  ), shared_xaxes=False,shared_yaxes=False)

        sdf = [float(x) for x in list(df[df['provider']=='sabre']['fare'])]
        cdf = [float(x) for x in  list(df[df['provider']=='competitor']['fare']) ]

        fig.append_trace({
            'y': sdf,
            'x': [i for i in range(len(sdf))],
            'type':'scatter','name':'Sabre',
            'marker': {'color':'red'}

        } ,1, 1)
        fig.append_trace({
            'y': cdf,
            'x': [i for i in range(len(cdf))],
            'type':'scatter','name':'Competitor',
            'marker': {'color':'blue'}

        } ,1, 1)
        fig.append_trace({
            'y': [s - c for s, c in zip(sdf,cdf)],
            'x': [i for i in range(min (len(cdf) , len(sdf)) )],
            'type':'bar','name':'Sabre - Competitor',
            'marker': {'color':'green'}

        } ,1, 1)

        return fig

    @app.callback(
        dash.dependencies.Output('time_vs_fare-graph', 'figure'),
        [dash.dependencies.Input('Provider', 'values'),Input('datatable-gapminder', 'rows'), Input('datatable-gapminder', 'selected_row_indices')])
    def update_image_src(selector,rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Time vs Fare',  ), shared_xaxes=False,shared_yaxes=False)

        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]

        df = pd.DataFrame(rows)
        df = df[ df['provider'].isin( selector ) ]
        try:
            for provider in selector:
                fig.append_trace({
                    'x': df[(df['provider'] == provider ) & (df['is_duplicate']=='N')]['fare'],
                    'y': df[(df['provider'] == provider ) & (df['is_duplicate']=='N')]['travel_time'],
                    'type': 'scatter', 'mode': 'markers', 'name':provider, 'marker': {'color': {'sabre':'red','competitor':'blue','overlap':'green'}[provider] }
                }, 1, 1)
            fig.append_trace({
                'x': df[(df['is_duplicate']=='Y')]['fare'],
                'y': df[(df['is_duplicate']=='Y')]['travel_time'],
                'type': 'scatter', 'mode': 'markers', 'name':'overlap', 'marker': {'color':'green'},
            }, 1, 1)
        except : pass
        return fig


    @app.callback(dash.dependencies.Output('overlap-graph', 'figure'),
        [dash.dependencies.Input('Provider', 'values'),Input('datatable-gapminder', 'rows'), Input('datatable-gapminder', 'selected_row_indices')])
    def update_image_src(selector, rows, selected_row_indices):

        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Options Distribution by Carrier',  ), shared_xaxes=False,shared_yaxes=False)

        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        df = df[ df['provider'].isin( selector ) ]
        providers = pd.unique(df['provider'])
        aux_count = 0
        for provider in providers:
            print (provider)
            aux_df = df[df['provider']==provider]
            fig.append_trace({
                'x': aux_df['idx'], #'x': [i for i in range(aux_count, len(aux_df) + aux_count)],#'x':df[df['provider']==provider]['id'],
                'y': [1 for i in list(range(aux_df.shape[0]))],
                'type': 'bar',
                'name':provider,
                'marker': {'color': {'competitor':'blue','sabre':'red'}[provider]}
            } ,1, 1)
            aux_count += len(aux_df)
        try:
            aux_df = df[df['is_duplicate']=='Y']
            fig.append_trace({
                'x': aux_df['id'],# 'x': [i for i in range(aux_count, len(aux_df) + aux_count)],  #'x':df[df['is_duplicate']=='Y']['id'],
                'y': [1 for i in list(range(aux_df.shape[0]))],
                'type': 'bar',
                'name':'overlap',
                'marker': {'color': 'green'}
            } ,1, 1)
        except: pass
        return fig

    @app.callback(dash.dependencies.Output('overlap_pie-graph', 'figure'),
    [dash.dependencies.Input('time_vs_fare-graph', 'figure'),Input('datatable-gapminder', 'rows'),Input('datatable-gapminder', 'selected_row_indices')])
    def update_image_src2(selector, rows, selected_row_indices):


        fig = plotly.tools.make_subplots( rows=4, cols=1, subplot_titles=('Time vs Fare',  ), shared_xaxes=False,shared_yaxes=False)
        try:
            if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
            df = pd.DataFrame(rows)
            duplicate_dict = dict((df['is_duplicate'].value_counts()))

            data = Data([
                go.Pie(values=list(duplicate_dict.values()), labels=list(duplicate_dict.keys()))
             ])

            return go.Figure(data=data)

        except: return fig



    def test():
        return 'TEST'

    if __name__ == '__main__':
            app.run_server(debug=True)

    return app


def get_bfm_rs_app(df):
    app = DjangoDash('bfm_rs_app')
    app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
    external_js = ["https://code.jquery.com/jquery-3.2.1.min.js","https://codepen.io/bcd/pen/YaXojL.js"]
    for js in external_js:
        app.scripts.append_script({"external_url": js})

    app.layout = html.Div(
        html.Div([
            html.Div([dcc.Graph(id='map-graph')], className= 'ten columns offset-by-one'),
            html.Div([dcc.Graph(id='time_vs_fare-graph')], className= 'ten columns offset-by-one'),
            html.Div([dcc.Graph(id='fare_histogram-graph')], className= 'four columns offset-by-one'),
            html.Div([dcc.Graph(id='fare_histogram-graph2')], className= 'four columns offset-by-one'),
            html.Div([dcc.Graph(id='time_histogram-graph')], className= 'ten columns offset-by-one'),

            #html.Div([dcc.Graph(id='heatmap-graph')], className= 'ten columns offset-by-one'),



            html.Div([dt.DataTable( rows=df.to_dict('records'), row_selectable=True, filterable=True, sortable=True, selected_row_indices=[],id='datatable')
            ], className= 'ten columns offset-by-one')
        ],className='row')
    )



    @app.callback(dash.dependencies.Output('time_vs_fare-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def update_image_src(rows, selected_row_indices):
        fig = plotly.tools.make_subplots( rows=1, cols=1, subplot_titles=('Time vs Fare',  ), shared_xaxes=False,shared_yaxes=False)
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)

        fig.append_trace({
            'x': df['price'],
            'y': df['travel_time'],
            'type': 'scatter',
            'opacity':0.5,
            'mode':'markers',
            'marker': {'color':'red','size': 15, 'line':{'width':2}},
            #'name': str(df['itinerary']),
            #'orientation': 'h',
        }, 1, 1)


        return fig

    @app.callback(dash.dependencies.Output('fare_histogram-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def fare_histogram_graph(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        nrows = df.shape[0]
        data = Data([go.Histogram(x=df['price'], nbinsx=10)]) #,nbinsx = nrows // 20
        return go.Figure(data=data)

    @app.callback(dash.dependencies.Output('fare_histogram-graph2', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def fare_histogram_graph2(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        fl_count = df['flight_count'].unique()
        x = []
        group_labels = []
        for fl in fl_count:
            x.append(df[df['flight_count']==fl]['price'].astype(float))
            group_labels.append(str(fl))

        #fig['layout'].update(title='Fare Distribution by Number of Flights')
        data = Data([go.Histogram(x=x, nbinsx=1)])
        colors = ['#3A4750', '#F64E8B', '#c7f9f1','blue']
        fig = ff.create_distplot(x, group_labels, bin_size=.8, curve_type='normal', colors=colors) #

        return fig

    @app.callback(dash.dependencies.Output('time_histogram-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def time_histogram_graph(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        nrows = df.shape[0]
        data = Data([go.Histogram(x=df['travel_time'], nbinsx=15)]) #,nbinsx = nrows // 20

        #TODO:
        print (df[['travel_time','price']].astype(float).describe() )
        print(df[['travel_time','price']].astype(float).var())
        return go.Figure(data=data)


    @app.callback(dash.dependencies.Output('map-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def map(rows, selected_row_indices, sep='|'):
        print ('CREATING YOUR MAP')
        print (sep)
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        stopover = len(df['itinerary'][0].split(sep)[0].split(sep))
        DepartureAirports=df['DepartureAirports'][0].split(sep)
        #ArrivalAirports=df['DepartureAirports'][0].split(sep)
        origin_iata, destination_iata = str(DepartureAirports[0]), str(DepartureAirports[stopover])
        print (origin_iata)
        print (destination_iata)
        airport_details = AS.get_airport_details([origin_iata, destination_iata])
        lat = [airport_details[iata]['lat'] for iata in [origin_iata, destination_iata]]
        lon = [airport_details[iata]['lon'] for iata  in [origin_iata, destination_iata]]
        min_lat, max_lat, min_lon, max_lon = min(lat)-10, max(lat)+10, min(lon)-10, max(lon)+10
        print (lat)
        aux = set(zip(list(df['DepartureAirports']),list(df['ArrivalAirports'])   ))
        cities = []
        for onds in aux:
            color = 'rgb'+str( (int(random.random() * 256)%256,int(random.random() * 256)%256,int(random.random() * 256)%256) )

            (x,y) = onds
            segments = ( list ( zip ( x.split(sep) , y.split(sep) ) ) )
            for segment in segments:
                (ori, dest) = segment
                airports_iata_list = [ori, dest]
                airport_details = AS.get_airport_details(airports_iata_list)
                lat = [airport_details[iata]['lat'] for iata in airports_iata_list]
                lon = [airport_details[iata]['lon'] for iata in airports_iata_list]
                ap_names = [airport_details[iata]['city']+'_'+airport_details[iata]['country'] for iata in airports_iata_list]
                cities.append(
                    dict(
                        type = 'scattergeo',
                        lat = lat,lon = lon,
                        hoverinfo = 'text',
                        text = ap_names,
                        mode = 'lines',
                        line = dict(
                            width = 2,
                            color = color,
                        )
                    )
                )

        layout = dict(
                title = ' '.join(airports_iata_list),
                showlegend = False,
                geo = dict(
                    resolution = 50,
                    showland = True,
                    landcolor = 'rgb(243, 243, 243)',
                    countrycolor = 'rgb(204, 204, 204)',
                    lakecolor = 'rgb(255, 255, 255)',
                    projection = dict( type="equirectangular" ),
                    coastlinewidth = 2,
                    lataxis = dict(
                        range =[0,0],# [ min_lat, max_lat ],
                        showgrid = True,
                        tickmode = "linear",
                        dtick = 10
                    ),
                    lonaxis = dict(
                        range = [0,0],#[ min_lon, max_lon ],
                        showgrid = True,
                        tickmode = "linear",
                        dtick = 20
                    ),
                )
            )

        fig = dict( data=cities, layout=layout )

        return fig



    #@app.callback(dash.dependencies.Output('heatmap-graph', 'figure'),[Input('datatable', 'rows'), Input('datatable', 'selected_row_indices')])
    def heatmap(rows, selected_row_indices):
        if len(selected_row_indices) != 0: rows = [rows[x] for x in selected_row_indices]
        df = pd.DataFrame(rows)
        its = list(df['itinerary'])
        dist_m = AS.get_matrix_distance(its)
        print ('CALCULATED')
        return go.Figure(data=Data([go.Heatmap(z=dist_m )] ) )



    if __name__ == '__main__':
        app.run_server(debug=True)

    return app
