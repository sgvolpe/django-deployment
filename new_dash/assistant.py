###
# Imports
import os, re, itertools, urllib, io, urllib,datetime,zlib, base64, re, datetime
import lxml.etree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn import preprocessing

from collections import OrderedDict

from matplotlib.ticker import FuncFormatter



DEBUG = False

import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return run_time
    return wrapper_timer

@timer
def log(func, out_path):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log_file = open(out_path, 'a')
        value = out_path()
        log_file.write(value)
        log_file.close()
    return wrapper



def it_distance(it1, it2,sep='-'):
    f_count = max( len(it1.replace(sep*2,sep).split(sep)), len(it2.replace(sep*2,sep).split(sep)) )
    legs1, legs2 = it1.split(sep*2),it2.split(sep*2)
    for i in range(len(legs1)):
        l = zip(legs1[i].split(sep),legs2[i].split(sep))
        f_count=f_count - sum([a==b for (a,b) in l])
    return f_count


def get_matrix_distance(it1,it2,sep='-'):
    #it1, it2 = it1.replace(sep*2,sep).split(sep), it2.replace(sep*2,sep).split(sep)
    m, n =len(it1), len(it2)
    M = [[0 for w in range(n)] for w in range(m)]
    for i in range(m):
        for j in range(n):
            M[i][j] = it_distance(it1[i],it2[j],sep=sep)
    return M

def itineraries_distance(it1, it2,sep='-'):
    M = get_matrix_distance(it1, it2,sep=sep)
    result = 0
    for r in M:
        result += min(r)
    return result

def generate_dates(aps,loss,base_date=None):
    if base_date==None: base_date=datetime.datetime.now()
    dates = []
    for ap in aps:
        for los in loss:
            dep_date = base_date + datetime.timedelta(days=int(ap))
            ret_date = dep_date + datetime.timedelta(days=int(los))
            dates.append(str(dep_date.strftime("%Y-%m-%d"))+'/'+str(ret_date.strftime("%Y-%m-%d")))
    return dates


def payload_change(payload,parameters):
    time = 'T11:00:00'
    namespaces = {'ota': 'http://www.opentravel.org/OTA/2003/05'}
    tree_in = ET.fromstring(payload)
    OriginDestinationInformationList = tree_in.findall('ota:OriginDestinationInformation', namespaces=namespaces)
    [ob,ib] = OriginDestinationInformationList[:25]
    if 'pcc' in parameters:pass
    if 'origin' in parameters: ob.find('ota:OriginLocation', namespaces=namespaces).attrib["LocationCode"] = ib.find('ota:DestinationLocation', namespaces=namespaces).attrib["LocationCode"] = parameters['origin']
    if 'destination' in parameters:  ob.find('ota:DestinationLocation', namespaces=namespaces).attrib["LocationCode"] = ib.find('ota:OriginLocation', namespaces=namespaces).attrib["LocationCode"] =parameters['destination']
    if 'dep_date' in parameters: ob.find('ota:DepartureDateTime', namespaces=namespaces).text = parameters['dep_date']+time
    if 'ret_date' in parameters: ib.find('ota:DepartureDateTime', namespaces=namespaces).text = parameters['ret_date']+time
    return ET.tostring(tree_in,pretty_print=True, encoding='unicode')


def get_airport_details(airport_iata_list, airports_file='static/airports.csv'):
    df = pd.read_csv(airports_file, index_col=4)
    retorno = {}
    for airport_iata in airport_iata_list:
        try: retorno[airport_iata] = df.loc[airport_iata].to_dict()
        except: pass
    return retorno





class bfm_rs:

    def __init__(self, file_path):
        #if DEBUG:print ('creating BFM RS: '+file_path)
        self.file_path = file_path


    def get_chunks(self, bfm_rs):
        #Get Chunks
        chunk_list = [x.replace('--','') for x in bfm_rs.split('--StreamingChunkBreak')]
        return chunk_list


    def save_bfm_df(self, dataframe, output_path):
        dataframe.to_csv(output_path, sep=',')


    def get_xml_from_path(self, file_path ):
        bfm_rs = open(file_path,'r').read()
        tree_in = ET.fromstring(bfm_rs)
        return tree_in


    def get_alliance(self, itinerary, sep='-' ):
        retorno = set()
        al2all = pd.read_csv('dashboard/static/alliances.csv',index_col='airline').to_dict()
        airlines = [flight[:2] for flight in itinerary.replace(sep*2, sep).split(sep)]
        for airline in airlines:
            if airline in al2all['alliance']: retorno.add( al2all['alliance'][airline] )
            else: retorno.add('na')
        return str(retorno)

    def bfm_rs_decompress(self, bfm_xml, save_to_txt=True, file_name='decompressed.xml'):
        namespaces = {'ota': 'http://www.opentravel.org/OTA/2003/05'}
        bfm_text = bfm_xml.text
        payload = zlib.decompress(base64.b64decode(bfm_text), 16+zlib.MAX_WBITS)
        bfm_rs = ET.fromstring(payload)

        if save_to_txt:
            file_path='decompressed/'+file_name
            out=open(file_path+'-decompressed.xml','w')
            out.write(payload)
            out.close()
        if DEBUG: print ('Decompress DONE')
        return  bfm_rs


    def dechunk_bfm(self, chunk_list, decompress=False, file_name='1.xml',save_to_txt=True):
        namespaces = {'ota': 'http://www.opentravel.org/OTA/2003/05'}
        aux = []
        for chunk in chunk_list:
            if chunk is not '':
                bfm_rs = ET.fromstring(chunk)
                if decompress: bfm_rs = self.bfm_rs_decompress(bfm_xml, True, file_name=file_name)#Decompress
                aux.append(bfm_rs)

        rt_itineraries = aux[0].find('PricedItineraries',namespaces=namespaces)
        for b in aux[1:]:

            OneWayItineraries = aux[0].find('OneWayItineraries',namespaces=namespaces)
            if OneWayItineraries is None:
                aux[0].append(ET.fromstring('<ota:OneWayItineraries xmlns:ota="http://www.opentravel.org/OTA/2003/05"><ota:SimpleOneWayItineraries RPH="1"></ota:SimpleOneWayItineraries><ota:SimpleOneWayItineraries RPH="2" xmlns:ota="http://www.opentravel.org/OTA/2003/05"></ota:SimpleOneWayItineraries></ota:OneWayItineraries>'))
            OneWayItineraries = aux[0].find('OneWayItineraries',namespaces=namespaces) ## ota:
            ow_list = OneWayItineraries.findall('SimpleOneWayItineraries',namespaces=namespaces)
            b_OneWayItineraries = b.find('OneWayItineraries',namespaces=namespaces)
            if b_OneWayItineraries is not None:
                b_ow_list = b.find('OneWayItineraries',namespaces=namespaces)
                for ow in b_ow_list:
                    rph = int ( ow.attrib['RPH']) -1
                    ow_its =  ow.findall('PricedItinerary',namespaces=namespaces)
                    for each in ow_its:
                        ow_list[rph].append(each)

            RoundTripItineraries = b.find('PricedItineraries',namespaces=namespaces)
            if RoundTripItineraries is not None:
                RoundTripItinerary_list = RoundTripItineraries.findall('PricedItinerary',namespaces=namespaces)
                for it in RoundTripItinerary_list:
                    rt_itineraries.append(it)

        if save_to_txt:
            file_path='dechunk/'+file_name
            out=open(file_path+'-dechunked.xml','w')
            out.write(ET.tostring(aux[0], pretty_print = True))
            out.close()

        return aux[0]


    #Parse to DF.csv
    def bfm_rs_to_df(self, save=False, output_path='output_path',sep='|',parameters={'ond':'error'},other={'response_time':0,'payload_size':-1}, RET='', truncate=-1):

        namespaces = {'ota': 'http://www.opentravel.org/OTA/2003/05', 'SOAP-ENV': 'http://schemas.xmlsoap.org/soap/envelope/'}
        compressed = chunked = json = chunked_compressed = False
        file_name = self.file_path.split('/')[-1]
        try:bfm_rs_txt = open(self.file_path,'r').read()
        except: return pd.DataFrame()


        envelope = ET.fromstring(bfm_rs_txt)
        body = envelope.find('SOAP-ENV:Body',namespaces=namespaces)
        bfm_rs = ET.tostring(list(body)[0], pretty_print = True, encoding='unicode')


        if '{' in bfm_rs_txt: json = True
        elif 'StreamingChunkBreak' in bfm_rs_txt and 'CompressedResponse' in bfm_rs: chunked_compressed = True
        elif 'StreamingChunkBreak' in bfm_rs_txt: chunked = True
        elif 'CompressedResponse' in bfm_rs_txt:  compressed = True

        if chunked_compressed:
            chunk_list = self.get_chunks(bfm_rs)
            bfm_rs = self.dechunk_bfm(chunk_list,True,file_name)
        elif chunked:
            chunk_list = self.get_chunks(bfm_rs)
            bfm_rs = self.dechunk_bfm(chunk_list,False,file_name)
        elif compressed:
            bfm_rs = self.bfm_rs_decompress(ET.fromstring(bfm_rs),save_to_txt=False)
        elif json:
            print ('Json') #TODO:
            #bfm_rs = json.loads(bfm_rs)
        else:
            bfm_rs = ET.fromstring(bfm_rs)#ET.fromstring(get_xml_from_path(file_path).text)
        if RET=='bfm_rs':
            return ET.tostring(bfm_rs, pretty_print=True, encoding='unicode')
            return ET.tostring(bfm_rs, pretty_print = True)

        PricedItineraries = bfm_rs.find('ota:PricedItineraries',namespaces=namespaces)

        try:
            OneWayItineraries = bfm_rs.find('ota:OneWayItineraries',namespaces=namespaces)
            SimpleOneWayItineraries_list =  OneWayItineraries.findall('ota:SimpleOneWayItineraries',namespaces=namespaces)
        except Exception as e: SimpleOneWayItineraries_list = []
        options = [PricedItineraries]+SimpleOneWayItineraries_list
        try:
            priced_itineraries = ([ options[x][y] for x in range(len(options)) for y in range(len(options[x]))] )
        except Exception as e:
            priced_itineraries = []
            if DEBUG: print ('Error on reading priced Itineraries on priced_itineraries = ([ options[x][y] for x in range(len(options)) for y in range(len(options[x]))] )')
        di  = []
        priced_itineraries = enumerate(priced_itineraries)
        if truncate != -1:
            priced_itineraries[: truncate ] # parse only a given amount of itineraries
        for idx, option in priced_itineraries:

            # ITINERARY PART
            AirItinerary = option.find('ota:AirItinerary',namespaces=namespaces)
            OriginDestinationOptions = AirItinerary.find('ota:OriginDestinationOptions',namespaces=namespaces)
            legs = OriginDestinationOptions.findall('ota:OriginDestinationOption',namespaces=namespaces)

            flight_list, flight_op_list, ElapsedTimes, booking_classes, MarriageGrps, DepartureAirports, ArrivalAirports,DepartureDateTimes, ArrivalDateTimes =[],[],[],[],[],[],[],[],[]
            for leg in legs:
                leg_flights, leg_flights_op = [],[]
                ElapsedTime = leg.attrib['ElapsedTime']

                flights = leg.findall('ota:FlightSegment',namespaces=namespaces)
                for flight in flights:
                    DepartureDateTimes.append(flight.attrib['DepartureDateTime'])
                    ArrivalDateTimes.append(flight.attrib['ArrivalDateTime'] )
                    FlightNumber, StopQuantity = flight.attrib['FlightNumber'], flight.attrib['StopQuantity']
                    try: ResBookDesigCode = flight.attrib['ResBookDesigCode']
                    except: ResBookDesigCode='n/a'

                    DepartureAirport = flight.find('ota:DepartureAirport',namespaces=namespaces).attrib['LocationCode']
                    ArrivalAirport = flight.find('ota:ArrivalAirport',namespaces=namespaces).attrib['LocationCode']
                    OperatingAirline = flight.find('ota:OperatingAirline',namespaces=namespaces).attrib['Code']
                    OperatingFlightNumber = flight.find('ota:OperatingAirline',namespaces=namespaces).attrib['FlightNumber']
                    MarketingAirline = flight.find('ota:MarketingAirline',namespaces=namespaces).attrib['Code']
                    MarriageGrp = flight.find('ota:MarriageGrp',namespaces=namespaces).text

                    leg_flights.append(MarketingAirline + (4 - len(FlightNumber))*'0'+  FlightNumber)
                    leg_flights_op.append(OperatingAirline + (4 - len(OperatingFlightNumber))*'0'+  OperatingFlightNumber)
                    booking_classes.append(ResBookDesigCode)
                    MarriageGrps.append(MarriageGrp)
                    DepartureAirports.append(DepartureAirport)
                    ArrivalAirports.append(ArrivalAirport)


                flight_list.append(leg_flights)
                flight_op_list.append(leg_flights_op)
                ElapsedTimes.append(ElapsedTime)

            ITINERARY = (sep*2).join([sep.join(x) for x in flight_list])
            ITINERARY_OP = (sep*2).join([sep.join(x) for x in flight_op_list])


            d = {'idx':idx, 'option_number':option.attrib['SequenceNumber'], 'itinerary':ITINERARY}

            try:
                d['origin'] = parameters['origin']
                d['destination'] = parameters['destination']
                d['dep_date'] = parameters['dep_date']
                d['ret_date'] = parameters['ret_date']
            except: d['origin'] = d['destination'] = d['dep_date'] = d['ret_date'] = 'n/A'
            d['response_time']=other['response_time']
            d['payload_size']=other['payload_size']
            d['DepartureAirports']='|'.join(DepartureAirports)
            d['ArrivalAirports']='|'.join(ArrivalAirports)
            d['DepartureDateTime']='|'.join(DepartureDateTimes)
            d['ArrivalDateTime']= '|'.join(ArrivalDateTimes)
            d['booking_classes']='|'.join(booking_classes)
            d['marriage_indicators']='|'.join(MarriageGrps)
            d['travel_time_list']='|'.join(ElapsedTimes)
            d['travel_time']= sum([int(x) for x in ElapsedTimes])
            d['flight_count'] = len( ITINERARY.replace( (sep*2),sep ).split(sep) )
            d['mktg_optg_set'] = str ([x[:2]+'('+y[:2]+')' for (x, y )in zip(ITINERARY.replace((sep*2),sep).split(sep), ITINERARY_OP.replace((sep*2),sep).split(sep) )] ).replace("'","")

            DepartureDateTimes = [datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in DepartureDateTimes ] # Convert to datetime object
            ArrivalDateTimes = [datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in ArrivalDateTimes ]     # Convert to datetime object
            cnx_time = [ str(int((d-a).total_seconds() / 60 )) for (d, a) in zip(DepartureDateTimes[1:], ArrivalDateTimes[:-1])]         # Get the difference in minutes
            d['cnx_time'] = sep.join(cnx_time)

            #FARE PART
            AirItineraryPricingInfo = option.find('ota:AirItineraryPricingInfo',namespaces=namespaces)
            ItinTotalFare = AirItineraryPricingInfo.find('ota:ItinTotalFare',namespaces=namespaces)
            TotalFare = ItinTotalFare.find('ota:TotalFare',namespaces=namespaces)
            TotalFare_Amount = TotalFare.attrib['Amount']
            TotalFare_CurrencyCode = TotalFare.attrib['CurrencyCode']
            d['price'] = TotalFare_Amount #price #fare
            d['currency'] = TotalFare_CurrencyCode
            Tickets = AirItineraryPricingInfo.find('Tickets',namespaces=namespaces)
            d['multi_ticket'] = str (Tickets is not None)
            TPA_Extensions = option.find('ota:TPA_Extensions',namespaces=namespaces)
            AdditionalFares = TPA_Extensions.find('ota:AdditionalFares',namespaces=namespaces) # OPTIONAL
            additional_prices, additional_fares_bags = [], []
            try:
                additional_fares = AdditionalFares.findall('ota:AirItineraryPricingInfo',namespaces=namespaces)
                if additional_fares is not None:
                    if len(additional_fares ) >0:
                        for AdditionalFares in additional_fares:
                            ItinTotalFare = AdditionalFares.find('ota:ItinTotalFare',namespaces=namespaces)
                            TotalFare_i = ItinTotalFare.find('ota:TotalFare',namespaces=namespaces).attrib['Amount']
                            additional_prices.append(TotalFare_i)


                            #AirItineraryPricingInfo = AdditionalFares.find('ota:AirItineraryPricingInfo',namespaces=namespaces)
                            PTC_FareBreakdowns = AdditionalFares.find('ota:PTC_FareBreakdowns',namespaces=namespaces)
                            PTC_FareBreakdown = PTC_FareBreakdowns.find('ota:PTC_FareBreakdown',namespaces=namespaces)
                            PassengerFare = PTC_FareBreakdown.find('ota:PassengerFare',namespaces=namespaces)
                            TPA_Extensions = PassengerFare.find('ota:TPA_Extensions',namespaces=namespaces)
                            BaggageInformationList = TPA_Extensions.find('ota:BaggageInformationList',namespaces=namespaces)
                            if BaggageInformationList is not None:
                                BaggageInformation = BaggageInformationList.findall('ota:BaggageInformation',namespaces=namespaces)
                            additional_fares_bags = []
                            if len(BaggageInformation) > 0:
                                bags = []
                                for bag_info in BaggageInformation:
                                    try:
                                        bags.append(bag_info.find('ota:Allowance',namespaces=namespaces).attrib['Pieces'])
                                    except Exception as e:
                                        if DEBUG: print ('--> WARNING: getting bag pieces from Add Fare. '+str(e))
                                        try:
                                            bag_info.find('ota:Allowance',namespaces=namespaces).attrib['Weight'] == 1
                                            bags.append('1')
                                            if DEBUG: print ('-->Weight found')
                                        except  Exception as e:
                                            if DEBUG: print ('--> ERROR Trying to get the WEIGHT of Bags included on Main Fare. '+str(e))
                                            bags.append('n/a')

                            additional_fares_bags.append(sep.join(bags) )

            except Exception as e:
                if DEBUG: print ('--> ERROR. Getting ADD FAres' +str(e))
            d['additional_prices'] = sep.join(additional_prices)
            d['add_fares_bags'] = additional_fares_bags

            if DEBUG: print (additional_fares_bags)
            for ad_fare_idx in range(len(additional_fares_bags)):
                bags_per_leg_add_fares = [int(x) for x in additional_fares_bags[ad_fare_idx].split(sep) if x != 'n/a']
                if len(bags_per_leg_add_fares) != 0: bag_all_ad_fare = min(bags_per_leg_add_fares)
                else: bag_all_ad_fare = 0
                if type(bag_all_ad_fare) is int:
                    if bag_all_ad_fare > 0:
                        d['bag_incl_add_fare'] = additional_prices[ad_fare_idx]
                        break


            PTC_FareBreakdowns = AirItineraryPricingInfo.find('ota:PTC_FareBreakdowns',namespaces=namespaces)
            PTC_FareBreakdown = PTC_FareBreakdowns[0]
            PassengerFare = PTC_FareBreakdown.find('ota:PassengerFare',namespaces=namespaces)
            TPA_Extensions = PassengerFare.find('ota:TPA_Extensions',namespaces=namespaces)


            main_fare_bags_included = []
            try:
                BaggageInformationList = TPA_Extensions.find('ota:BaggageInformationList',namespaces=namespaces)
                BaggageInformation = BaggageInformationList.findall('ota:BaggageInformation',namespaces=namespaces)
                if BaggageInformation is not None:
                    if len(BaggageInformation) > 0:
                        for bag_info in BaggageInformation:
                            try:
                                main_fare_bags_included.append(bag_info.find('ota:Allowance',namespaces=namespaces).attrib['Pieces'])
                            except  Exception as e:
                                if DEBUG: print ('--> WARNING Trying to get the pieces of Bags included on Main Fare. '+str(e))
                                try:
                                    bag_info.find('ota:Allowance',namespaces=namespaces).attrib['Weight'] == 1
                                    main_fare_bags_included.append('1')
                                    if DEBUG: print ('--> WEIGHT FOUND')
                                except  Exception as e:
                                    if DEBUG: print ('--> ERROR Trying to get the WEIGHT of Bags included on Main Fare. '+str(e))
                                    main_fare_bags_included.append('n/a')
            except Exception as e:
                if DEBUG: print (str(e))
                main_fare_bags_included=[]
            d['main_fare_bags_included']='|'.join(main_fare_bags_included)


            TPA_Extensions = AirItineraryPricingInfo.find('ota:TPA_Extensions',namespaces=namespaces)
            try:
                AncillaryFeeGroups = TPA_Extensions.find('ota:AncillaryFeeGroups',namespaces=namespaces)
                AncillaryFeeGroup_list = AncillaryFeeGroups.findall('ota:AncillaryFeeGroup',namespaces=namespaces)
            except Exception as e: print ('--> ERROR. Trying to get ancillaries')

            ancillaries_bags = []
            bag_anc = []
            for AncillaryFeeGroup in AncillaryFeeGroup_list:
                bag_anc = []
                try:
                    if AncillaryFeeGroup.attrib['Code'] == 'BG':
                        try:
                            for AncillaryFeeItem in AncillaryFeeGroup.findall('ota:AncillaryFeeItem',namespaces=namespaces):
                                bag_anc.append(AncillaryFeeItem.attrib['StartSegment']+'_'+AncillaryFeeItem.attrib['EndSegment']+':'+AncillaryFeeItem.attrib['Amount'])
                        except Exception as e:
                            if DEBUG: print (str(e))
                            bag_anc.append('n/a')
                        ancillaries_bags.append(sep.join(bag_anc))
                except Exception as e:
                    if DEBUG: print (str(e))
                    ancillaries_bags=['n/a']
            d['bag_anc'] = ancillaries_bags


            ####
            try:
                SEGMENTS = [['' for seg in leg] for leg in flight_list] #segments_bags =
                for j in range(len(main_fare_bags_included)):
                    for i in range(len(SEGMENTS[j]) ):
                        SEGMENTS[j][i] = main_fare_bags_included[j]

            except Exception as e:
                if DEBUG: print ('HERE '*50)
                if DEBUG: print (str(e))
            #Get the first occurrence of bag ancillary for each segment into d2
            try:
                t = ([(b.split(':')[0], b.split(':')[1] ) for b in bag_anc ])     #  OJO:
            except Exception as e:
                if DEBUG: print ('ERROR. Working on t bag_anc')
                if DEBUG: print (f'bag_anc={bag_anc}')
            d_AUX = {}
            for x, y in t: d_AUX.setdefault(x, []).append(y)
            d2 = {}
            for k,v in d_AUX.items(): d2[k]=v[0]

            #Flatten the legs
            SEGMENTS_2 = list(itertools.chain(*SEGMENTS))

            bags_price = [0 for x in SEGMENTS_2]
            for k,v in d2.items():
                [a,b] = k.split('_')
                for r in range(int(a)-1 , int(b),1): #get the range of segments the ancillary applies for
                    if SEGMENTS_2[r] == '0':
                        bags_price[r] = int(v)
                        v=0                     # empty the value if within the same ancillary
                    else: bags_price[r] = 0


            d['buy_bags_price'] = bags_price
            d['total_price_for_bags_from_anc'] = sum(bags_price)
            d['bag_incl_main_fare'] = sum(bags_price) + float(d['price'])


                ####

            #### comparing ancillaries bags vs add_fare
            try: d['main_fare_baggage_allowance'] = min ( int(b) for b in d['main_fare_bags_included'].split('|') )
            except: d['main_fare_baggage_allowance'] = 0

            #Main Fare includes bag?
            CAT = 'NAN'
            if d['main_fare_baggage_allowance'] > 0: CAT = 'MAIN_FARE_INCLUDE_BAG'
            else:
                #Add fare includes bag?
                if 'bag_incl_add_fare' not in d: CAT = 'NO_ADD_FARE_WITH_BAG_OFFERED'
                elif float( d['bag_incl_add_fare'] ) != 0.0:
                    # Ancillary does not offer Bags
                    if len(bag_anc) == 0: CAT = 'COVERAGE_INCREASE'
                    # Ancillary does  offer Bags but more Expensive
                    else:
                        d['bag_incl_fare_diff'] = float(d['bag_incl_main_fare'] ) - float( d['bag_incl_add_fare'] )
                        if d['bag_incl_fare_diff'] > 0 :  CAT =  'BUNDLED_CHEAPER'
                        if d['bag_incl_fare_diff'] < 0 : CAT = 'UNBUNDLED_CHEAPER'
                        if d['bag_incl_fare_diff'] == 0 :CAT = 'TIE'
                else: CAT = 'NO_ADD_FARE_WITH_BAG_OFFERED'
            d['ANCvsMFxI'] = CAT

            'NAN,MAIN_FARE_INCLUDE_BAG,COVERAGE_INCREASE,BUNDLED_CHEAPER,UNBUNDLED_CHEAPER,TIE'

            if False:
                try:
                    1+''
                    d['bag_incl_fare_diff'] = float(d['bag_incl_main_fare'] ) - float( d['bag_incl_add_fare'] )
                    if len(bag_anc) == 0 or float( d['bag_incl_add_fare'] ) == 0.0:
                        if len(bag_anc) == 0: d['cheaper_bag_incl'] = 'ANC_NOT_OFFERED'
                        if float( d['bag_incl_add_fare'] ) == 0.0: d['cheaper_bag_incl'] += '_NO_ADD_FARE_WITH_BAG'
                        if float( d['bag_incl_add_fare'] ) != 0.0: d['cheaper_bag_incl'] += 'COVERAGE_INCREASE'
                    elif d['main_fare_baggage_allowance'] == 0:  #.astype('int')
                        if d['bag_incl_fare_diff'] > 0 :  d['cheaper_bag_incl'] = 'add_fare'
                        if d['bag_incl_fare_diff'] < 0 : d['cheaper_bag_incl'] = 'anc_fare'
                        if d['bag_incl_fare_diff'] == 0 : d['cheaper_bag_incl'] = 'TIE'

                    elif len(bag_anc) == 0: d['cheaper_bag_incl'] = 'ANC_NOT_OFFERED'
                    else: d['cheaper_bag_incl'] = 'main_fare_included_bag'
                except Exception as e:
                    if DEBUG: print (str(e))




            di.append(d)

        df = pd.DataFrame(di,index=range(len(di)))
        df['market'] = 'NA' #other['ond']
        try:df['price_rank'] = df['price'].astype(float).rank(ascending=1,method='dense')
        except: df['price_rank'] = 'n/a'
        try:df['time_rank'] = df['travel_time'].astype(int).rank(ascending=1,method='dense')
        except: df['time_rank'] = 'n/a'
        try:df['price_time_rank'] = df['price_rank']*1000 + df['time_rank']
        except: df['price_time_rank'] = 'n/a'
        try:df['alliance'] = df['itinerary'].apply(lambda x: self.get_alliance(x, sep=sep))
        except: df['alliance'] = 'n/a'

        try:
            #df['main_fare_baggage_allowance'] = df['main_fare_bags_included'].apply(lambda x: min ( int(b) for b in x.split('|') ) )
            df['addfares_baggage_allowance'] = df['add_fares_bags'].apply(lambda y: [min ( int(b) for b in x.split('|') ) for x in y] )
        except:
            df['baggage_allowance'] = 'NA'
            df['addfares_baggage_allowance'] = 'NA'


        try:
            df['airlines'] = df['itinerary'].apply(lambda x: sep.join( set ( [flight[:2] for flight in x.replace(sep*2, sep).split(sep)] ) ))
        except:
            df['airlines'] = 'N/a'



        if save:
            self.save_bfm_df(df, output_path)

        return df
