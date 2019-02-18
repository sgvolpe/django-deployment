

import networkx as nx
import os, datetime, json
import pandas as pd

from django.db import models
from django.conf import settings
from django.core.validators import RegexValidator
from django.urls import reverse
from django.utils import timezone
from django import forms
from django.forms.widgets import CheckboxSelectMultiple

from multiselectfield import MultiSelectField
from sklearn.linear_model import LinearRegression
from . import assistant as AS
from . import SWS as SWS
from . import BargainFinderMaxRQ



DEBUG=True



def get_bfm_airports(f_name):
    airports = set()
    rs = json.loads(open(f_name).read())
    OriginDestinationOptions_list = rs['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary']
    for OriginDestinationOptions in OriginDestinationOptions_list:
        legs = OriginDestinationOptions['AirItinerary']['OriginDestinationOptions']['OriginDestinationOption']
        for leg in legs:
            flights = leg['FlightSegment']
            for flight in flights:
                ArrivalAirport= flight['ArrivalAirport']['LocationCode']
                DepartureAirport= flight['DepartureAirport']['LocationCode']
                airports.add(ArrivalAirport)
                airports.add(DepartureAirport)


    return airports

def get_all_airports():
    a = set()
    directory = 'new_dash/static/virtualinterlining/'
    for f_name in os.listdir(directory+'responses/'):
        a = a.union( get_bfm_airports(f'{directory}responses/{f_name}') )
    return list (a)



def extract_db_nodes(DB,it_ori,it_des,d_date,r_date,min_p=.1, rand_airport_p=.05):
    'Input: origin, destination and dats of journey'
    'Output: list of airports with higher likelihood to return cheaper fares'
    #Filter DB
    DB = DB
    DB = DB[ (DB['it_ori']==it_ori) & (DB['it_des']==it_des)
             # & (DB['d_date']==d_date) & (DB['r_date']==r_date)
    ]
    total_rows = DB.shape[0]
    print (f'We have {total_rows} inputs for {it_ori}-{it_des}.')
    #Extract Efficiency
    DB['cheaper_int'] = DB['cheaper'].astype(int)
    efficiency = DB.groupby(['path'])['cheaper_int'].mean().to_dict()
    print (f'Efficiency Vector: {efficiency}')

    #Extact not repeating Airports
    ALL_AIRPORTS = get_all_airports() #'MVD,BUE,SAO,LIM,SCL,MEX,MIA,NYC,MAD,LON,ROM,LIS,PAR,JNB,DXB,SIN,BKK,SYD'.split(',')
    paths = []
    for p, e in efficiency.items():
        if float(e) + .1 > random.random():
            paths.append(p)



    nodes = set()
    for p in paths:
        airports = p.split('-')
        for a in airports:
            nodes.add(a)

    # Include some variance
    for a in ALL_AIRPORTS:
        if rand_airport_p > random.random():
            nodes.add(a)

    return list(nodes)

class Network:
    ori,des,ddate,rdate,stopovers, stayover,G,airport_id,airports,DB =None,None,None,None,None,None,None,None,None,None
    itineraries = {}
    def __init__(self, ori='MVD', des='LON', ddate='2019-06-01',rdate='2019-06-08',

                   stopovers=['BUE','SAO','MAD','LIS'], DB_path='new_dash/static/virtualinterlining/DB.csv',stayover=1, penalty_per_night=100):
        self.ori = ori
        self.des = des
        self.ddate = ddate
        self.rdate = rdate
        self.stayover = stayover
        self.G = nx.DiGraph()
        self.airport_id = {}
        self.airports = [ori,des] + stopovers
        self.session = SWS.Rest_Handler()
        self.DB = self.pull_DB(DB_path)
        self.penalty_per_night = penalty_per_night


        for idx, a in enumerate(self.airports): self.airport_id[a]=idx
        e = [(a,b, 999999) for a in self.airports for b in self.airports
            # if a != b
            ]

        self.G.add_weighted_edges_from(e)
        if DEBUG: print (f'### NETWORK CREATED ###\n AIRPORTS: {self.airports}\n ')

    def edit_edge(self, o, d, w):
        self.G[o][d]['weight'] = w

    def change_date(self, date, p_days=1):
        return str(datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=p_days))[:10]



    def fill_edges(self):
        if DEBUG: print (f'### Filling Edges ###\n ')
        for o in self.airports:
            for d in self.airports:
                if o != d and ( o==self.ori or d==self.des) :
                    if o == self.ori: dep_date = self.ddate
                    else: dep_date = str(self.change_date(self.ddate, self.stayover) )

                    if d == self.des: ret_date = self.rdate
                    else: ret_date = str(self.change_date(self.rdate, self.stayover) )

                    f_name = f"bfm_{o}_{d}_{dep_date}_{ret_date}_adrs.json"        #TODO: USE db to retrieve these
                    directory='new_dash/static/virtualinterlining/'
                    stats_file = open(f'{directory}stats.txt','r')
                    stats = json.loads(stats_file.read())
                    stats_file.close()

                    df = self.DB
                    df = df[
                          (df['it_ori']==o)
                        & (df['it_des']==d)
                        & (df['d_date']==dep_date)
                        & (df['r_date']==ret_date)
                        ]


                    if df.shape[0] > 0: # if f_name in os.listdir('responses/'):  #Cache
                        if DEBUG: print ('No Need to run BFM.')
                        stats['cache_hit'] += 1
                        w = df.iloc[0]['price']
                        self.itineraries[o+d]= df.iloc[0]['itinerary']
                    else:
                        bfm_results = self.session.bfmad(o, d, dep_date, ret_date)
                        w = bfm_results['price']
                        print (w)
                        self.itineraries[o+d]=bfm_results['itin']

                        #Use generated data to also fill Database (CAche)
                        for data in bfm_results['bfm_itineraries']:
                            self.DB_add(data)


                    stats['total_hits'] += 1
                    stats_file = open('stats.txt','w')
                    stats_file.write(json.dumps(stats))
                    stats_file.close()


                    if DEBUG: print (f'-> Checking Price for: {o}-{d}: ${w}')
                    self.edit_edge(o,d, w)






    def draw(self):
        pos = nx.spring_layout(self.G)
        labels = nx.get_edge_attributes(self.G,'weight')
        nx.draw(self.G, with_labels=True, edge_labels=labels,pos=pos)
        nx.draw_networkx_edge_labels(self.G,pos,edge_labels=labels)
        plt.show()

    def get_path_itinerary(self, path_x):
        inbound=[]
        retorno = []
        while(len(path_x)>1):
            #print (path_x)
            #print(path_x[0]+path_x[1], retorno)
            retorno.append(self.itineraries[path_x[0]+path_x[1]])
            inbound.append(path_x[0])
            path_x.pop(0)
        return retorno

    def path_weight(self, path, W=0):
        'Given a path calculates the total Weight of the same as the sum of all the edges'
        path_copy = [p for p in path]
        #if DEBUG:print (f'-- {path}, {graph},{S}')
        while len(path_copy) > 1:
            W += self.G[path_copy[0]][path_copy[1]]['weight']
            path_copy.pop(0)
        return W

    def shortest_path(self, ori, des, maxstops=1):
        '''Given two Nodes returns the path with the less weight.
            Brute Force Algorithm:
                Generate all possible paths, return the first one with minimum weight
        '''
        #if DEBUG: print (f'From {ori} to {des}:')
        min_weight, min_path = 999999, []

        # Generate the List with All simple paths from ori to des
        try:
            paths = list(nx.all_simple_paths(self.G, ori, des, maxstops+1 ) )
        except Exception as e:
            print ('ERROR', str(e))
            paths = []

        if DEBUG: print (f'The algorithm found {len( paths )} Possible Paths.')
        for p in paths: print(f'- {p}')

        # Keep the one with minimum weight
        self.all_possible_paths = {}
        for actual_path in paths:
            self.all_possible_paths['-'.join(actual_path)] = self.path_weight(actual_path)

        min_weight = 999999
        min_path = []
        for k, v in self.all_possible_paths.items():

            ts = datetime.datetime.now()
            itinerary = self.get_path_itinerary(k.split(','))

            nsp=self.all_possible_paths['-'.join([ori,des])]
            data={'ts':ts,'it_ori':ori,'it_des':des,'d_date':self.ddate,'r_date':self.rdate,'path':k
                       ,'price':v,'non_stopover_price':nsp,'itinerary':itinerary,
                  'cheaper': nsp > v + self.penalty_per_night * self.stayover}
            self.DB_add(data)

        try:
            min_path = min(self.all_possible_paths, key=self.all_possible_paths.get)
            min_weight = self.all_possible_paths[min_path]
            itinerary = self.get_path_itinerary(min_path.split('-'))

        except:
            min_path, min_weight, itinerary = None, 999999, ''

        #Return
        self.solution={'min_path':min_path, 'min_weight':min_weight, 'itinerary':itinerary}
        return min_weight, min_path


    def to_string(self):
        retorno = {}
        for k, v in self.__dict__.items():
            if k !='DB': retorno[k]=v
        return str(retorno)


    def log(self, log_path='log.txt', to_log=None):
        l = open(log_path, 'a')
        if to_log is None: to_log = self.to_string()
        l.write(str(datetime.datetime.now() ) +','+ to_log+'\n')
        l.close()


    def pull_DB(self,DB_path):
        df = pd.read_csv(DB_path)
        return df

    def push_DB(self):
        self.DB.to_csv('DB.csv',index=False)

    def DB_add(self,data):
        'Adds new row into DB and then saves it locally'
        self.DB=self.DB.append(data,ignore_index=True)
        self.push_DB()


    def DB_health(self):
        # Health of DB
        #DB = pd.read_csv('DB.csv')
        DB = self.DB
        DB['ond'] = DB['it_ori'] + DB['it_des']
        rows = DB.shape[0]
        origins, destinations, onds = DB['it_ori'].unique(), DB['it_des'].unique(), DB['ond'].unique()

        print (DB.groupby(['ond'])['it_ori'].count().plot() )

        print (f'Number of Rows: {rows}')
        print (f'Number of Origins: {len(origins)}')
        print (f'Number of Destinantions: {len(destinations)}')
        print (f'Number of OnDs: {len(onds)}')

class VirtualInterlining(models.Model):

    ori = models.CharField(max_length=3, blank=False, default='MVD')
    des = models.CharField(max_length=3, blank=False, default='LON')
    ddate = models.CharField(max_length=20, blank=False, default='2019-03-31')
    rdate = models.CharField(max_length=20, blank=False, default='2019-04-15')
    night_penaly = models.IntegerField(default=100)
    stayover = models.IntegerField(default=2)
    analysis_finished = models.DateTimeField(blank=True, null=True)

    'virtualinterlining'

    def run(self):
        #Load DB
        DB_path='new_dash/static/virtualinterlining/DB.csv'
        DB = pd.read_csv(DB_path)

        #Get Stopovers (Network Nodes)
        STOPOVERS=extract_db_nodes(DB, self.ori, self.des, self.ddate, self.rdate, min_p=.5, rand_airport_p=.05)
        stopovers = [ s for s in STOPOVERS if s != self.ori and s != self.des ]

        #Generate the Network
        N = Network(self.ori, self.des,stopovers=stopovers,stayover=self.stayover)

        #Calculate the Edges (Prices)
        N.fill_edges()

        #Get the Output: - All possible solutions and the sabre_cheapest
        N.shortest_path(ORI,DES,maxstops=1)

        N.log()




class BFM_Parse(models.Model):
    bfm_rs_file = models.FileField(blank=False)
    #bfm_rs_df = pd.DataFrame()#models.CharField(max_length=255, blank=True)

    def parse(self):
        bfmrs_json = BargainFinderMaxRQ.bfm_from_file(self.bfm_rs_file)
        #self.bfm_rs_df = bfmrs_json# BargainFinderMaxRQ.bfm_rs_to_df(bfmrs_json, RET='ROCK')
        return bfmrs_json #BargainFinderMaxRQ.bfm_rs_to_df(bfmrs_json, RET='ROCK')


    def get_absolute_url(self):
        return reverse("new_dash:BFM_ParseDetail",kwargs={"pk": self.pk})


'''
\\LTXW0396.sgdcelab.sabre.com\Jobs
'''
class OceanAnalysis(models.Model):
    title = models.CharField(max_length=50, blank=True)
    description = models.CharField(max_length=255, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def get_absolute_url(self):
        return reverse("new_dash:ocean_analysis_view",kwargs={"pk": self.pk})

    def __str__(self):
        return str(self.pk)

    def create_directories(self):
        directory = 'new_dash/static/ocean_analysis/'+str(self.pk)
        [os.makedirs(dir) for dir in [directory,directory+'/input',directory+'/query_files',
         directory+'/output' ]
            if not os.path.exists(dir)]
        self.path = directory
        self.save()

    def log(self, text):
        log_path = self.path + '/log.txt'
        log_file = open(log_path, 'a')
        log_file.write(str( datetime.datetime.now() ) + ',' + str(text) + '\n')
        log_file.close()

    def open_dfs(self, s_path=None,c_path=None ):
        if DEBUG: print ('''Receives a 2 Ocean raw data file set
        Returns 2 Normalized DFs one per Provider
        Open and creates extra needed columns: fare,itinerary travel time''' )

        if s_path is None:
            sdf = self.sdf
        else:
            s_cols = ['AP','LOS','market','DCXRFLIGHT','RCXRFLIGHT','TotaltravelTime','fare']
            sdf = pd.read_csv(s_path, usecols=s_cols)
        sdf['itinerary'] = sdf[[u'DCXRFLIGHT','RCXRFLIGHT']].apply(self.get_itinerary_sabre,axis=1)

        if c_path is None:
            cdf = self.sdf
        else:
            c_cols=['AP','LOS','MARKET','KEY_OUT1','KEY_OUT2','KEY_OUT3','KEY_IN1','KEY_IN2','KEY_IN3','TOTALTRAVELTIME','PRICE']
            cdf = pd.read_csv(c_path, sep=';', usecols=c_cols)
        cdf = cdf.fillna('')
        if 'KEY_OUT3' in cdf.keys() and 'KEY_IN3' in cdf.keys() :
            cdf['itinerary'] = cdf[['KEY_OUT1','KEY_OUT2','KEY_OUT3','KEY_IN1','KEY_IN2','KEY_IN3']].apply(self.get_itinerary_comp,axis=1)
        else:
            cdf['itinerary'] = cdf[['KEY_OUT1','KEY_OUT2','KEY_IN1','KEY_IN2']].apply(self.get_itinerary_comp,axis=1)


        #Cleaning of Columns and renaming
        sdf.dropna(subset=['TotaltravelTime'], inplace=True)
        sdf['travel_time'] = sdf[u'TotaltravelTime'].astype(str)
        cdf['travel_time'] = cdf[u'TOTALTRAVELTIME']
        cdf.rename(columns={'PRICE':'fare'}, inplace=True)
        cdf.rename(columns={'MARKET':'ond'}, inplace=True)
        sdf.rename(columns={'market':'ond'}, inplace=True)
        sdf['ap_los'] = sdf['AP'].astype(str) +'_'+sdf['LOS'].astype(str)
        cdf['ap_los'] = cdf['AP'].astype(str) +'_'+cdf['LOS'].astype(str)
        sdf = sdf[['itinerary','ap_los','ond','travel_time','fare']]
        cdf = cdf[['itinerary','ap_los','ond','travel_time','fare']]

        return [sdf,cdf]

    def get_alliance(self, itinerary ):
        retorno = []
        al2all = pd.read_csv('dashboard/static/alliances.csv',index_col='airline').to_dict()
        airlines = [flight[:2] for flight in itinerary.split('-')]
        for airline in airlines:

            if airline in al2all['alliance']: retorno.append( al2all['alliance'][airline] )
        else: retorno.append('na')
        return retorno

    def get_itinerary_sabre(self, legs):
        'Builds the Itinerary from Flight columns for Sabre raw data'
        ob,ib = legs[0],legs[1]
        try:
            return '-'.join([f[:2]+ (6-len(f))*'0'+f[2:] for f in ob.split(' - ')+ib.split(' - ')])
        except Exception as e:
            print ('ERROR generating ITinerary',str(e) ,legs )

    def get_itinerary_comp(self, flights):
        'Builds the Itinerary from Flight columns for competitor raw data'
        return  '-'.join([f[:2]+ (6-len(f))*'0'+f[2:] for f in flights if f!=''])


    def filter_df(self, sdf, cdf, ap_los=None, ond=None, num_options=None, truncate_to_min=False):
        if ap_los is not None:
            sdf = sdf[(sdf['ap_los'] == ap_los ) ]
            cdf = cdf[(cdf['ap_los'] == ap_los ) ]
        if ond is not None:
            sdf = sdf[(sdf['ond'] == ond) ]
            cdf = cdf[(cdf['ond'] == ond ) ]
        if truncate_to_min:
            num_options = min(sdf.shape[0], cdf.shape[0])
        if num_options is not None:
            sdf=sdf[:num_options]
            cdf=cdf[:num_options]
        if DEBUG: print (sdf.shape[0], cdf.shape[0] )
        return [sdf,cdf]

    def generate_id_list(self, configuration_file_name='configuration.csv' ):
        if DEBUG: print (self, configuration_file_name  )
        qid_list=[]

        DF = pd.read_csv(self.path+'/input/'+configuration_file_name, sep=',') # Used to extract ap,los, onds
        if DEBUG: print ('Generate a concatenation of valid pos, search_dates, ap, los and onds' )
        try:
            if configuration_file_name.split('_')[0] == 'sabre':
                if DEBUG: print (' '*4 + 'Configuration file is abre')
                DF['qid'] = DF['searchdate'].astype(str) + '_' + DF['market']+ '_' + DF['AP'].astype(str) + '_' + DF['LOS'].astype(str) #TODO: if not done
            else:
                if DEBUG: print (' '*4 + 'Configuration file is not sabre')
                print (DF.columns)
                DF['qid'] = configuration_file_name.split('_')[1] + '_' + DF['MARKET']+ '_' + DF['AP'].astype(str) + '_' + DF['LOS'].astype(str)
                #DF['qid'] = DF['SEARCHDATE'].astype(str) + '_' + DF['MARKET']+ '_' + DF['AP'].astype(str) + '_' + DF['LOS'].astype(str)
                if DEBUG: print(DF.head())
        except Exception as e:
            print (f'ERROR READING CONF FILE: {configuration_file_name}: {str(e)}')
        try:
            qid_list = list ( DF['qid'].unique() )
            if DEBUG: print (' I will need to generate ' + str(len( qid_list )) + ' files.')
            self.log(' I will need to generate ' + str(len( qid_list )) + ' files.')
        except Exception as e:
            qid_list=[]
            print (str(e))
        return qid_list

    def generate_ind_files(self, override_files=False):
        if DEBUG: print ('Individual files?')
        directory = 'new_dash/static/ocean_analysis/' + str(self.pk)
        i_folder = directory + '/input/'
        files_created = 0
        control_date = ''

        print (i_folder)
        print (os.listdir(i_folder))
        try:
            id_list = self.generate_id_list(os.listdir(i_folder)[0])
        except Exception as e:
            if DEBUG: print (f'ERROR reading input files: {os.listdir(i_folder)[0]}. {str(e)}')
            id_list= []

        print ('POSSIBLE FILES: ' + str(len(id_list)) )


        for q_id in id_list:
            s_date, ond, ap, los = q_id.split('_')
            folder = directory + '/input/'
            ap_los = ap + '_' + los
            #try:
            file_name = s_date + '_' + ond + '_' + ap + '_' + los + '.csv'
            if file_name not in os.listdir(directory + '/query_files/'):
                if ( s_date !=  control_date) :                                 # Control to avoid open df is not necessary
                    [s_df, c_df] = self.open_dfs( i_folder+'sabre_' + s_date , i_folder+'competitor_' + s_date ) # +'.csv' +'.csv'
                control_date = s_date                                           # Update control Variable

                # Filter dataframes based on id
                [S, C] = self.filter_df(s_df, c_df, ap_los, ond, truncate_to_min=True)

                #Create new DF which will be saved as to new individual file
                S['provider'], S['search_date'], S['ond'] = 'sabre', s_date, ond
                C['provider'], C['search_date'], C['ond'] = 'competitor', s_date, ond
                SC = pd.concat([S,C])

                output_path = directory + '/query_files/'+ file_name

                #Save csv only if doesn't exist or override_files == True
                if override_files:
                    pd.DataFrame(SC).to_csv(output_path, sep=',')
                elif not os.path.exists(output_path):
                    pd.DataFrame(SC).to_csv(output_path, sep=',')

                directory + '/query_files/' + file_name


    def advanced_processing(self):
        if DEBUG: print ('Advanced Processing START: ')
        self.log(' Start of Advanced Processing')

        folder = self.path + '/query_files/'
        output_path = self.path + '/output/'
        for f in os.listdir(folder):
            f_name = f.split('.')[0]
            if DEBUG: print (f_name)

            if f_name + '-processed.csv' in os.listdir(output_path): pass
            else:
                df = pd.read_csv(folder + f )
                if df.shape[0] != 0:

                    df["is_duplicate"] = df.duplicated(subset=['itinerary'],keep=False)
                    df['price_rank_abs'] = df['fare'].rank(ascending=1,method='dense')
                    df['time_rank_abs'] = df['travel_time'].rank(ascending=1,method='dense')
                    df['price_rank'] = df.groupby('provider')['fare'].rank(ascending=1,method='dense')
                    df['time_rank'] = df.groupby('provider')['travel_time'].rank(ascending=1,method='dense')

                    # Split based on provider to later merge
                    sdf = df[ df['provider'] == 'sabre' ]
                    cdf = df[ df['provider'] != 'sabre' ]

                    #Merge of content from both providers based on itinerary
                    result = pd.merge(sdf, cdf, how='outer', on=['itinerary', 'itinerary'] ,suffixes=['_sabre','_competitor'] )

                    #CLEANING:
                    sdf = cdf =  None

                    result.fillna('')
                    result['fare_difference'] = (result['fare_sabre']-result['fare_competitor'])/result['fare_sabre']
                    result['search_date'] = result[['search_date_sabre', 'search_date_competitor']].astype(str).apply(lambda x: ''.join(x).replace('nan','')[:8], axis=1)
                    result['ap_los'] = result[['ap_los_sabre', 'ap_los_competitor']].astype(str).apply(lambda x: ''.join(x).replace('nan',''), axis=1)
                    result['ond'] = result[['ond_sabre', 'ond_competitor']].astype(str).apply(lambda x: ''.join(x).replace('nan','')[:6], axis=1)
                    result['provider'] = result[['provider_sabre', 'provider_competitor']].astype(str).apply(lambda x: '_'.join(x), axis=1)
                    result.drop(['provider_competitor','provider_sabre' ],axis=1,inplace=True)
                    result['provider'] = result['provider'].apply(lambda x: {'sabre_competitor':'both','sabre_nan':'sabre_unique','nan_competitor':'competitor_unique'}[x])
                    result['alliance'] = result['itinerary'].apply(lambda x: self.get_alliance(x))
                    result['itinerary_carrier'] = result['itinerary'].apply(lambda x: '-'.join (sorted(list(set([flight[:2] for flight in x.split('-')]))) ))
                    result['time_rank_abs'] = result['time_rank_abs_sabre']

                    result2= result[['search_date','ond','ap_los','provider','itinerary',
                                     'fare_sabre','fare_competitor','fare_difference',
                                    'travel_time_sabre','travel_time_competitor',
                                     'price_rank_abs_sabre','price_rank_abs_competitor',
                                     'time_rank_sabre', 'time_rank_competitor',
                                     'time_rank_abs_sabre','itinerary_carrier'
                                    ]]

                    result2.to_csv(output_path + f_name + '-processed.csv', sep=',')

                    #print (output_path + f_name + '-processed.csv')

        self.log(' End  of Advanced Processing')
        if DEBUG: print (' End  of Advanced Processing')


    def process_summary(self):
        dicts = []
        input_dir = self.path + '/output/'
        output_path = self.path
        print (self.path)
        for file_path in os.listdir(input_dir):
            file_name = file_path.split('.')[0]
            di = {'file_name': file_name}
            print ('Processing: ', file_name)
            df = pd.read_csv(input_dir + file_path, sep=',')

            #Split based on provider
            sabre_options = df[df['provider'].str.contains('sabre')]
            competitor_options = df[df['provider'].str.contains('competitor')]

            di['ond'] = df['ond'][0]
            di['ap_los'] = df['ap_los'][0]
            di['search_date'] = df['search_date'][0]

            # Mean and STD Fare
            di['competitor_fare_mean']=competitor_options['fare_competitor'].mean()
            di['sabre_fare_mean']=sabre_options['fare_sabre'].mean()
            di['competitor_fare_std']=competitor_options['fare_competitor'].std()
            di['sabre_fare_std']=sabre_options['fare_sabre'].std()


            #Get cheapest option per provider
            try:di['competitor_cheapest']=competitor_options.loc[competitor_options['fare_competitor'].idxmin()]['fare_competitor']
            except: di['competitor_cheapest'] = 1.0
            try:di['sabre_cheapest']=sabre_options.loc[sabre_options['fare_sabre'].idxmin()]['fare_sabre']
            except: di['sabre_cheapest'] = 1.0

            di['lfe_fare_difference'] = (di['competitor_cheapest'] - di['sabre_cheapest'] ) / di['competitor_cheapest']

            if di['lfe_fare_difference'] == 0:  di['lfe'] = 'tie'
            elif di['lfe_fare_difference'] > 0: di['lfe'] = 'sabre'
            elif di['lfe_fare_difference'] < 0: di['lfe'] = 'competitor'



             #Get travel time mean and std
            di['competitor_time_mean']=competitor_options['travel_time_competitor'].mean()
            di['sabre_time_mean_sabre'] = sabre_options['travel_time_sabre'].mean()
            di['competitor_time_std']=competitor_options['travel_time_competitor'].std()
            di['sabre_time_std']=sabre_options['travel_time_sabre'].std()


            di['sabre_time_max'] = sabre_options['travel_time_sabre'].max()
            di['sabre_time_min'] = sabre_options['travel_time_sabre'].min()
            di['competitor_time_max']  = competitor_options['travel_time_competitor'].max()
            di['competitor_time_min']  = competitor_options['travel_time_competitor'].min()

            if di['competitor_time_min']- di['sabre_time_min'] == 0:  di['quickest'] = 'tie'
            elif di['competitor_time_min']- di['sabre_time_min'] > 0: di['quickest'] = 'sabre'
            elif di['competitor_time_min']- di['sabre_time_min'] < 0: di['quickest'] = 'competitor'


            di['sabre_only'] = 0
            di['competitor_only'] = 0
            di['both'] = 0
            try:
                di['sabre_only'] = dict(df['provider'].value_counts() )['sabre_unique']
            except Exception: pass
            try:
                di['competitor_only'] = dict( df['provider'].value_counts() )['competitor_unique']
            except Exception: pass
            try: di['both'] = dict( df['provider'].value_counts() )['both']
            except Exception: pass
            try:di['overlap']= di['both']*100.0 / (di['sabre_only'] + di['competitor_only'] - di['both']*2)
            except Exception: di['overlap'] = 100

            di['sabre_min_carrier'] = dict( sabre_options.groupby(['itinerary_carrier'])['fare_sabre'].min() )
            di['competitor_min_carrier'] = dict( competitor_options.groupby(['itinerary_carrier'])['fare_competitor'].min() )


            di['sabre_cxr_div'] = len (di['sabre_min_carrier'].keys() )
            di['competitor_cxr_div'] = len (di['competitor_min_carrier'].keys() )




            if di['competitor_cxr_div']- di['sabre_cxr_div'] == 0:  di['cxr_div'] = 'tie'
            elif di['competitor_cxr_div']- di['sabre_cxr_div'] > 0: di['cxr_div'] = 'sabre'
            elif di['competitor_cxr_div']- di['sabre_cxr_div'] < 0: di['cxr_div'] = 'competitor'

            dicts.append(di)


        out_df = pd.DataFrame(dicts)
        out_df.to_csv(output_path + '/summary.csv' )


    def run(self):
        log = ['OA Running']
        try:
            self.create_directories()
            log.append('Directories Created')
        except Exception as e:
            log.append(str(e) + 'ERROR CREATING Directories!')

        self.generate_ind_files()
        log.append('IND Files')
        self.advanced_processing()
        log.append('ADV Files')
        self.process_summary()
        log.append('SUM Files')
        print (log)



class fare_forecast(models.Model):
    ond_val = RegexValidator(r'^([A-Z]{3}-[A-Z]{3},)*([A-Z]{3}-[A-Z]{3})+$','Check!')
    author = models.ForeignKey('auth.User',on_delete=models.CASCADE)
    title = models.CharField(max_length=50, blank=True)
    ond = models.CharField(max_length=20, blank=True, validators=[ond_val] )
    departure_date = models.CharField(max_length=20, blank=True)
    return_date = models.CharField(max_length=20, blank=True )
    predict_date = models.CharField(max_length=20, blank=True )

    def get_forecast_and_history(self, parameters):
        RH = SWS.Rest_Handler()
        RH.token
        url_parameters = '&'.join( [k+'='+v for k,v in parameters.items()] )
        if DEBUG: print (url_parameters)
        fare_forecast =  RH.LowFareForecats(url_parameters).text
        try:
            fare_forecast = json.loads(fare_forecast)
        except Exception as e: print (str(e))

        fare_history =  RH.LowFareHistory(url_parameters).text
        try:
            fare_history = json.loads(fare_history)
        except Exception as e: print (str(e))

        return {'fare_forecast':fare_forecast, 'fare_history':fare_history}

    def get_forecast_from_history(self, fare_history, ap_of_fare):
        try:
            dates = [(datetime.datetime.strptime(x['ShopDateTime'][:10],"%Y-%m-%d") - datetime.datetime.now() ).days
             for x in fare_history['FareInfo']]
        except Exception as e:
            print ('ERROR. On get_forecast_from_history I')
            print (str(e))
            print (fare_history)
        prices = [float(x['LowestFare']) for x in fare_history['FareInfo']]
        if DEBUG: print (dates, prices)

        # Create dataframe
        dataset=pd.DataFrame(data=prices,index=dates)
        d = {'prices': prices, 'dates': dates}
        df = pd.DataFrame(data=d)

        #Train the model
        regressor = LinearRegression()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 1].values
        regressor.fit(X, y)
        print('Intercept, Coef: ', regressor.intercept_,regressor.coef_)

        #Plot DataFrame
        #df.plot(x='dates', y='prices', style='o')
        #plt.title('Price vs Days in the Past')
        #plt.xlabel('Days in the past')
        #plt.ylabel('Price')
        #plt.show()

        fare_predicted = regressor.predict(ap_of_fare)
        return fare_predicted

    def run(self):
        ond = self.ond.split('-')
        today_date = datetime.datetime.now()
        ap_of_fare = (datetime.datetime.strptime(self.predict_date,"%Y-%m-%d") - today_date ).days
        parameters = {'origin':ond[0], 'destination':ond[1], 'departuredate':self.departure_date, 'returndate':self.return_date }

        forecast_and_history = self.get_forecast_and_history(parameters)
        fare_forecast = forecast_and_history['fare_forecast']
        fare_history = forecast_and_history['fare_history']
        fare_predicted = self.get_forecast_from_history(fare_history, ap_of_fare)

        print ('The fare prediction on ', f_date, ' is ',str(fare_predicted[0]) )
        return ('The fare prediction on ', f_date, ' is ',str(fare_predicted[0]) )

    def __str__(self):
        return '' # 'new_dash/static/several_same_BFM/'+str(self.pk)+ '/summary.csv'

class several_same_BFM(models.Model):
    author = models.ForeignKey('auth.User',on_delete=models.CASCADE)
    title = models.CharField(max_length=50, blank=True)
    description = models.CharField(max_length=255, blank=True)

    ap_val = RegexValidator(r'^([0-9]+,{0,1})+$','Check!')
    ond_val = RegexValidator(r'^([A-Z]{3}-[A-Z]{3},)*([A-Z]{3}-[A-Z]{3})+$','Check!')
    ap = models.CharField(max_length=500, blank=True, validators=[ap_val] )
    los = models.CharField(max_length=500, blank=True, validators=[ap_val] )
    onds = models.CharField(max_length=1000, blank=True, validators=[ond_val] )
    bfm_template_1 =  models.TextField(null=True, blank=True)

    repeats = models.IntegerField()
    total_queries= models.IntegerField(default=0)
    analysis_finished = models.DateTimeField(blank=True, null=True)

    def log(self, text):
        log_path = self.directory + '/log.txt'
        log_file = open(log_path, 'a')
        log_file.write(str( datetime.datetime.now() ) + ',' + str(text) + '\n')
        log_file.close()


    def __str__(self):
        return 'new_dash/static/several_same_BFM/'+str(self.pk)+ '/summary.csv'

    def get_summary_name(self):
            return 'new_dash/static/several_same_BFM/'+str(self.pk)+ '/summary.csv'

    def get_absolute_url(self):
        self.total_queries = len(self.ap.split(','))*len(self.los.split(','))*(self.repeats)
        self.save()
        self.send_rq()
        return reverse("new_dash:several_same_BFMDetail",kwargs={"pk": self.pk})

    def process_summary_by_carrier(self,sep = '|'):
        directory = 'new_dash/static/several_same_BFM/'+str(self.pk)
        o_path = directory + '/summary/summary_per_airline/'
        i_path = directory + '/dataframes/'
        DF = None
        di = [] # list of dicts to convert to DF
        for df_path in os.listdir(i_path):
            d = {}
            df = pd.read_csv( i_path + df_path )

            df['airlines']= df['itinerary'].apply(lambda x: sep.join( set ( [flight[:2] for flight in x.replace(sep*2, sep).split(sep)] ) ))
            g_df = df[['airlines','ANCvsMFxI']].groupby(['airlines','ANCvsMFxI']).size()
            g_df.to_csv(o_path+df_path, sep=',')

        i_path = o_path
        categories = 'NAN,MAIN_FARE_INCLUDE_BAG,COVERAGE_INCREASE,BUNDLED_CHEAPER,UNBUNDLED_CHEAPER,TIE,NO_ADD_FARE_WITH_BAG_OFFERED'.split(',')
        frames = [pd.DataFrame(columns=categories),]+[pd.read_csv(i_path+df_path, header=None) for df_path in os.listdir(i_path) ]
        L = [len(df) for df in frames]
        DF = pd.concat( frames , axis=0)
        DF.rename(index=str, columns={0: "Airlines", 1:"Won",2: "Count"}, inplace=True)
        G_DF = DF[['Airlines','Won',"Count"]].groupby(['Airlines','Won']).sum()

        o_path = directory + '/summary/summary_per_airline.csv'
        G_DF.to_csv(o_path, sep=',')

        #version2
        if False:
            o_path = directory + '/summary/summary_per_airline_v2.csv'
            l = {}
            DF = DF.fillna(0)

            for x in DF.iterrows():
                carrier = x[1][0]

                if carrier not in l:
                    l[carrier] = {}
                    for CAT in categories:
                        l[carrier][CAT]=0
                    #l[carrier] = {'main_fare_included_bag':0, 'add_fare':0,'anc_fare':0,'ANC_NOT_OFFERED':0,'ANC_NOT_OFFERED_NO_ADD_FARE_WITH_BAG':0, 'ANC_NOT_OFFEREDCOVERAGE_INCREASE':0 }

                l[carrier][x[1][1]] += int(x[1][2])
            G_DF2 = pd.DataFrame(data=list(l.values()), index=list(l.keys()) )
            G_DF2.to_csv(o_path, sep=',')


    def process_summary(self, type=['ANCvsMFxI']):
        directory = 'new_dash/static/several_same_BFM/'+str(self.pk)
        o_path = directory + '/summary/summary.csv'
        i_path = directory + '/dataframes/'
        DF = None
        di = [] # list of dicts to convert to DF
        ###type='ANCvsMFxI'
        for df_path in os.listdir(i_path):
            d = {}
            df = pd.read_csv( i_path + df_path )

            categories = 'NAN,MAIN_FARE_INCLUDE_BAG,COVERAGE_INCREASE,BUNDLED_CHEAPER,UNBUNDLED_CHEAPER,TIE,NO_ADD_FARE_WITH_BAG_OFFERED'.split(',')

            if 'ANCvsMFxI' in type :
                for CAT in categories:
                    try:
                        d[CAT] = dict(df['ANCvsMFxI'].value_counts() )[CAT]
                    except Exception as e:
                        d[CAT]=0

                d['TOTAL_OPTIONS'] = df.shape[0]
                try: d['origin'] = df['origin'][0]
                except: d['origin']='n/a'
                try: d['destination'] = df['destination'][0]
                except: d['destination']='n/a'
                try: d['dep_date'] = df['dep_date'][0]
                except: d['dep_date']='n/a'
                try: d['ret_date'] = df['ret_date'][0]
                except: d['ret_date']='n/a'

                #d['bag_incl_fare_diff'] = float(d['bag_incl_main_fare'] ) - float( d['bag_incl_add_fare'] )
                # Fare difference where there's a difference positive or negative
                try:
                    df_x = df[(df['bag_incl_fare_diff'].notnull() & ( df['bag_incl_fare_diff'].astype(float) != 0 ) )]
                    df_x['FARE_DIFF'] = df_x['bag_incl_fare_diff']/df_x['bag_incl_main_fare']
                    d['fare_difference_mean_percentage'] = df_x['FARE_DIFF'].mean()
                except: d['fare_difference_mean_percentage'] = 'n/a'

                #Fare difference only when bundled cheaper (fare diff positive)
                try:
                    df_y = df[(df['bag_incl_fare_diff'].notnull() & ( df['bag_incl_fare_diff'].astype(float) > 0 ) )]
                    df_y['FARE_DIFF'] = df_y['bag_incl_fare_diff']/df_y['bag_incl_main_fare']
                    d['FARE_IMPROVEMENT_MEAN'] = df_y['FARE_DIFF'].mean()
                except: d['FARE_IMPROVEMENT_MEAN'] = 'n/a'


            if False:
                try: d['main_fare_included_bag'] = dict(df['cheaper_bag_incl'].value_counts() )['main_fare_included_bag']
                except: d['main_fare_included_bag']=0
                try: d['TIE'] = dict(df['cheaper_bag_incl'].value_counts() )['TIE']
                except: d['TIE']=0
                try: d['add_fare'] = dict(df['cheaper_bag_incl'].value_counts() )['add_fare']
                except: d['add_fare']=0
                try: d['anc_fare'] = dict(df['cheaper_bag_incl'].value_counts() )['anc_fare']
                except: d['anc_fare']=0
                try: d['ANC_NOT_OFFERED'] = dict(df['cheaper_bag_incl'].value_counts() )['ANC_NOT_OFFERED']
                except: d['ANC_NOT_OFFERED']=0
                try: d['ANC_NOT_OFFERED_NO_ADD_FARE_WITH_BAG'] = dict(df['cheaper_bag_incl'].value_counts() )['ANC_NOT_OFFERED_NO_ADD_FARE_WITH_BAG']
                except: d['ANC_NOT_OFFERED_NO_ADD_FARE_WITH_BAG']=0
                try: d['ANC_NOT_OFFEREDCOVERAGE_INCREASE'] = dict(df['cheaper_bag_incl'].value_counts() )['ANC_NOT_OFFEREDCOVERAGE_INCREASE']
                except: d['ANC_NOT_OFFEREDCOVERAGE_INCREASE']=0



            di.append(d)
        df = pd.DataFrame(di,index=range(len(di)))
        #df = df[['dep_date','ret_date''origin','destination','fare_difference',']]
        df.to_csv(o_path, sep=',')

        self.process_summary_by_carrier()


    def send_rq(self, override=False):
        if DEBUG: print ('*'*10)
        if DEBUG: print ('Sending RQ')
            # Create directories
        directory = 'new_dash/static/several_same_BFM/'+str(self.pk)
        directories = [directory + '/requests', directory + '/responses', directory + '/dataframes', directory + '/summary', directory + '/responses_decompressed', directory+'/summary_per_airline']
        [os.makedirs(dir) for dir in directories if not os.path.exists(dir)]
        if DEBUG: print ('  -> Directories Created.')
        repeats = self.repeats
        ond_list = self.onds.split(',')
        dates = AS.generate_dates(self.ap.split(','), self.los.split(','))
        parameters_list = [{'repeat': str(repeat),'origin': ond.split('-')[0],'destination': ond.split('-')[1],'dep_date': date_comb.split('/')[0]
        ,'ret_date': date_comb.split('/')[1]} for  repeat in range(repeats) for ond in ond_list  for date_comb in dates]

        for params in parameters_list:
                # Change BFM
            new_payload = AS.payload_change(self.bfm_template_1, params)
                # Save RQ
            f_name = '_'.join(params.values()).replace('/','_')
            f = open( directory + '/requests/' + f_name + '.xml', 'w')
            f.write(new_payload)
            f.close()
            if DEBUG: print ('    '+directory + '/requests/' + f_name + '.xml')
        if DEBUG: print ('  -> Requests Created.')

        for f_name in os.listdir(directory + '/requests'):

            rq_payload = open(directory + '/requests/'+f_name).read()
            [repeat, origin, destination, dep_date, ret_date ] = f_name.split('.')[0].split('_')
            params = {'repeat':repeat, 'origin':origin, 'destination':destination, 'dep_date':dep_date, 'ret_date':ret_date }

            rs_f_name = '_'.join(params.values()).replace('/','_')
            rs_name = directory + '/responses/' + rs_f_name + '.xml'
            print (rs_f_name + '.xml')
            if rs_f_name + '.xml' not in os.listdir(directory + '/responses/') and override:
                # Send RQ
                bfm_rs = SWS.send_BFM(rq_payload, params)
                print (rs_name)
                if DEBUG: print ('    RQ Sent' )
                #rs_name = directory + '/responses/' + f_name + '.xml'
                f = open( rs_name , 'w')
                f.write(bfm_rs['rs_text']) #response = {'rs_text':rs.text,'response_time':response_time,'payload_size':payload_size}
                f.close()


            BFMRS = AS.bfm_rs(rs_name) # Convert to object
            print (params)
            #params =  {'origin':params['origin'],'destination': params['destination'],'dep_date': params['dep_date'],'ret_date': params['ret_date']}
            other = {'response_time':0,'payload_size':0}
            decomp_xml = BFMRS.bfm_rs_to_df( save=True, output_path=directory+ '/responses_decompressed/' + f_name + '-decompressed.xml', parameters=params, other=other, RET='bfm_rs' )
            rs_name = directory+ '/responses_decompressed/' + f_name + '-decompressed.xml'
            f = open( rs_name , 'w')
            f.write(decomp_xml)
            f.close()
            BFMRS.bfm_rs_to_df( save=True, output_path=directory+ '/dataframes/' + f_name + '.csv', parameters=params, other=other, RET='' )



        self.process_summary()

    def generate_dfs(self):
        directory = 'new_dash/static/several_same_BFM/'+str(self.pk)
        for rs_name in os.listdir( directory + '/responses_decompressed/'):
            f_name = rs_name
            [repeat, ori,des,ddate,rdate] = rs_name.split('.')[0].split('_')
            params =  {'origin':ori,'destination': des,'dep_date': ddate,'ret_date': rdate}
            BFMRS = AS.bfm_rs(directory + '/responses_decompressed/'+rs_name) # Convert to object
            other = {'response_time':0,'payload_size':0}

            decomp_xml = BFMRS.bfm_rs_to_df( save=True, output_path=directory+ '/responses_decompressed/' + f_name + '-decompressed.xml', parameters=params, other=other, RET='bfm_rs' )
            decomp_rs_name = directory+ '/responses_decompressed/' + f_name + '-decompressed.xml'
            f = open( decomp_rs_name , 'w')
            f.write(decomp_xml)
            f.close()

            BFMRS.bfm_rs_to_df( save=True, output_path=directory+ '/dataframes/' + f_name + '.csv', parameters=params, other=other, RET='', truncate=1 )

class BFM(models.Model):
    #author = models.ForeignKey('auth.User',on_delete=models.CASCADE)
    #benchmark = models.ForeignKey('Benchmark',on_delete=models.CASCADE, null=True)
    STATUS = ( ('Finished','Finished'),('Running','Running'),('NotStarted','NotStarted') )
    title = models.CharField(max_length=50, blank=True)
    description = models.CharField(max_length=255, blank=True)
    bfm_rq_txt =  models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    status = MultiSelectField(choices=STATUS)
    response_time = models.IntegerField(null=True)
    payload_size = models.IntegerField(null=True)
    bfm_rq_file = models.FileField(blank=True)
    bfm_rs_file = models.FileField(blank=False)
    df_path = models.CharField(max_length=100, blank=True)
    #TODO: Env

    def get_df(self):
        return open(self.df_path,'r')

    def send_rq(self):
        directory = 'new_dash/static/bfm/'+str(self.pk)
        self.directory=directory
        self.save()

        [os.makedirs(dir) for dir in [directory] if not os.path.exists(dir)]

        f = open(directory+'/bfm_rq.xml','w')    #Save RQ XML
        f.write(self.bfm_rq_txt)
        f.close()


        #Send BFMRQ
        payload = self.bfm_rq_txt#bfm_rq_file.read()

        bfm_rs = SWS.send_BFM(payload, params={})
        self.response_time = bfm_rs['response_time']
        self.payload_size = bfm_rs['payload_size']
        other = {'response_time':bfm_rs['response_time'],'payload_size':bfm_rs['payload_size'], 'ond':'AAA-BBB', 'search_date':str(self.timestamp)}
        bfm_rs = bfm_rs['rs_text'].replace('<?xml version="1.0" encoding="UTF-8"?>', '')

        f = open(directory+'/bfm_rs.xml', 'w')
        f.write(bfm_rs)
        f.close()

        f = open(directory+'/bfm_rs-decompressed.xml', 'w')
        f.write(self.decompress_rs())
        f.close()

        bfmdf = AS.bfm_rs(directory + '/bfm_rs.xml')
        params = {'origin':'xxx','destination':'yyy'}

        bfmdf.bfm_rs_to_df(save=True, output_path=directory+ '/df.csv', parameters=params,other=other,RET='')

    def resend_rq(self):

        self.save_rq()
        payload = self.bfm_rq_txt#self.bfm_rq_file.read()
        print (payload)
        bfm_rs = SWS.send_BFM(payload, params={})
        self.response_time = bfm_rs['response_time']
        self.payload_size = bfm_rs['payload_size']
        bfm_rs = bfm_rs['rs_text'].replace('<?xml version="1.0" encoding="UTF-8"?>', '')
        o_folder = 'dashboard/static/bfm/' + str(self.pk) + '/'
        f_loc = o_folder + 'bfm_rs.xml'
        f = open(f_loc, 'w')
        f.write(bfm_rs)
        f.close()
        self.timestamp=datetime.datetime.now()
        self.save()

    def decompress_rs(self):
        BFM_obj = AS.bfm_rs('new_dash/static/bfm/' + str(self.pk) + '/bfm_rs.xml')
        BFM_pay = BFM_obj.bfm_rs_to_df(RET='bfm_rs')

        return str(BFM_pay)

    def __str__(self):
        return ''#self.df_path

    def get_absolute_url(self):
        return reverse("new_dash:bfm_view",kwargs={"pk": self.pk})


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
