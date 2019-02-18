# Imports
import base64, datetime, functools, io, itertools, json,  os, re,time, urllib,zlib
import lxml.etree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn import preprocessing

from collections import OrderedDict

from matplotlib.ticker import FuncFormatter


def simple_log(to_log, out_path='log.txt'):
    log_file = open(out_path, 'a')
    log_file.write(f'{datetime.datetime.now},{to_log}\n')
    log_file.close()

DEBUG= True

def bfm_from_file(file_path):
    return json.loads(file_path.open().read())




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

#@log
def bfm_rs_to_df(bfm_rs, payload_type='json', sep='|', RET='ROCK'): # bfm_rs -> lxml
    if DEBUG: print (f'Starting Parsing of {payload_type}')
    namespaces = {'ota': 'http://www.opentravel.org/OTA/2003/05'}
    #tree_in = ET.fromstring(payload)

    if payload_type == 'xml':
        try: bfm_rs_txt = ET.fromstring(bfm_rs)
        except Exception as e:
            print ('ERRROR READING bfm_rs'+str(e))
            return pd.DataFrame()
        if 'CompressedResponse' in bfm_rs_txt: compressed, bfm_rs_txt = True, ''
        if compressed: bfm_rs = bfm_rs_decompress(bfm_rs, save_to_txt=False)

        print ('THIS IS NOT READY YET')
        1+''
    #TODO:
    elif payload_type == 'json':
        body = bfm_rs
        if 'OTA_AirLowFareSearchRS' not in body:
            simple_log(to_log='ERROR,There is no OTA_AirLowFareSearchRS element. ')
            return pd.DataFrame()

        OriginDestinationOptions = body['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary']
        OriginDestinationOptions_list = [(OriginDestinationOptions, 'RT')]
        #Multi-Ticket Treatment SOW
        if 'OneWayItineraries' in   body['OTA_AirLowFareSearchRS']:
            if DEBUG: print('OneWayItineraries')
            one_ways = body['OTA_AirLowFareSearchRS']['OneWayItineraries']
            [OneWay_ob, OneWay_ib] = one_ways['SimpleOneWayItineraries']
            if OneWay_ob is not None and OneWay_ib is not None:
                OriginDestinationOptions_list += [ (OneWay_ob['PricedItinerary'], 'OB'),(OneWay_ib['PricedItinerary'], 'IB')]

        if DEBUG: print (f'Number of Bounds: {len(OriginDestinationOptions_list )}')

        ROWS = []
        for leg_OriginDestinationOptions in OriginDestinationOptions_list:

                        #todo: elapsed_time = 0

            (OriginDestinationOptions, bound) = leg_OriginDestinationOptions

            if DEBUG: print (f'Reading Bound: {bound}')
            for option in OriginDestinationOptions:
                row = {}
                row['bound'] = bound
                # row['bound'] = option['AirItinerary']['DirectionInd']

                #ITINERARY PART
                legs = option['AirItinerary']['OriginDestinationOptions']['OriginDestinationOption']
                row['leg_elapsed_times'] = []
                row['MarketingAirline'], row['FlightNumber'] = [],[]
                row['DepartureAirport'],row['ArrivalAirport']  = [],[]
                row['DepartureDateTime'], row['ArrivalDateTime']  = [],[]
                row['OperatingAirline'], row['OperatingFlightNumber']  = [],[]

                leg_id = 0
                for leg in legs:
                    for x in ['MarketingAirline','FlightNumber','DepartureAirport','ArrivalAirport',
                                               'DepartureDateTime','ArrivalDateTime','OperatingAirline','OperatingFlightNumber']:
                        row[x].append([])
                    row['leg_elapsed_times'].append(float(leg['ElapsedTime']))

                    flights = leg['FlightSegment']
                    for flight in flights:
                        row['MarketingAirline'][leg_id].append( flight['MarketingAirline']['Code'] )
                        row['FlightNumber'][leg_id].append( flight['FlightNumber'] )

                        row['DepartureAirport'][leg_id].append( flight['DepartureAirport']['LocationCode'] )
                        row['ArrivalAirport'][leg_id].append( flight['ArrivalAirport']['LocationCode'] )
                        row['DepartureDateTime'][leg_id].append( flight['DepartureDateTime'] )
                        row['ArrivalDateTime'][leg_id].append( flight['ArrivalDateTime'] )
                        row['OperatingAirline'][leg_id].append( flight['OperatingAirline']['Code'] )
                        row['OperatingFlightNumber'][leg_id].append( flight['OperatingAirline']['FlightNumber'] )




                    leg_id += 1

                row['total_elapsed_time'] = sum(row['leg_elapsed_times'])



                parsed_flights = ([[ cxr + '0'*(4-len(nbr)) + nbr for (cxr, nbr) in list(zip(c,n))] for (c,n ) in
                                  list ( zip ( row['MarketingAirline'] ,row['FlightNumber'] )) ] )
                itinerary = (sep*2).join([sep.join(x) for x in parsed_flights])
                row['itinerary'] = itinerary
                '=========================== END OF ROW'

                try: AdditionalFares = option['TPA_Extensions']['AdditionalFares']
                except: AdditionalFares = []

        # 1 FARE
                #if DEBUG: print (AdditionalFares)
                #TODO: LastTicketDate,PricingSource,PricingSubSource,FareReturned,
                row['prices'], row['Taxes'],row['Total_Taxes'] = [],[],[]
                row['ptcs'],  row['AvailabilityBreaks'] = [],[]
                row['BaggageAllowances'] = []
                row['GovCarriers'] = []


                options = option['AirItineraryPricingInfo'] + [fare['AirItineraryPricingInfo'] for fare in AdditionalFares]
                row['BookingCodes'] =  [
                    [ [fbc['BookingCode'] for fbc in each['FareBasisCodes']['FareBasisCode']
                      ] for each in AirItineraryPricingInfo['PTC_FareBreakdowns']['PTC_FareBreakdown']]
                    for AirItineraryPricingInfo in options
                ]
                row['prices'] = [AirItineraryPricingInfo['ItinTotalFare']['BaseFare']['Amount']
                                for AirItineraryPricingInfo in options ]
                row['currency'] = options[0]['ItinTotalFare']['TotalFare']['CurrencyCode']

                row['BaseFares'] = [
                    AirItineraryPricingInfo['ItinTotalFare']['BaseFare']['Amount']
                    for AirItineraryPricingInfo in options
                ]
                row['fare_basis_codes'] = [
                    [[fbc['content'] for fbc in each['FareBasisCodes']['FareBasisCode'] ] for each in AirItineraryPricingInfo['PTC_FareBreakdowns']['PTC_FareBreakdown']]
                    for AirItineraryPricingInfo in options
                ]

                row['ptcs'] = [
                    [[str(each['PassengerTypeQuantity']['Quantity'])+each['PassengerTypeQuantity']['Code'] ] for each in AirItineraryPricingInfo['PTC_FareBreakdowns']['PTC_FareBreakdown']]
                    for AirItineraryPricingInfo in options
                ]

                row['governing_carriers'] = [
                    [ [fbc['GovCarrier'] for fbc in each['FareBasisCodes']['FareBasisCode']] for each in AirItineraryPricingInfo['PTC_FareBreakdowns']['PTC_FareBreakdown']]
                    for AirItineraryPricingInfo in options
                ]


                for AirItineraryPricingInfo in options:
                    AvailabilityBreak, BaggageAllowance = [],[]

                    PTC_FareBreakdown_list = AirItineraryPricingInfo['PTC_FareBreakdowns']['PTC_FareBreakdown']
                    for each in PTC_FareBreakdown_list:
                        avb, bal = [],[]
                        for fbc in each['FareBasisCodes']['FareBasisCode']: # FareBasisCodes
                            try: avb.append( str(fbc['AvailabilityBreak']) == 'true' )
                            except: avb.append( False )
                        AvailabilityBreak.append(avb)

                        #Baggage Allowance
                        BaggageInformation = each['PassengerFare']['TPA_Extensions']['BaggageInformationList']['BaggageInformation'] # list
                        for BI in BaggageInformation:
                            Allowance_list = BI['Allowance']
                            for ALLOWANCE in Allowance_list:
                                if 'Pieces' in ALLOWANCE: calc_allowance = ALLOWANCE['Pieces']
                                elif 'Weight' in ALLOWANCE:
                                    calc_allowance = 1

                                    Weight = ALLOWANCE['Weight']
                                    Unit = ALLOWANCE['Unit']
                                else: calc_allowance = 0
                            bags_dict = OrderedDict()
                            for ALLOW_SEG in BI['Segment']:
                                bags_dict[ALLOW_SEG["Id"]] = calc_allowance
                            bal.append(str(bags_dict))
                        BaggageAllowance.append(bal)

                    row['BaggageAllowances'].append(BaggageAllowance) # [[main_fare-ADT,main_fare-CNN],[add_fare-ADT,add_fare-CNN]] (segment_id, allowance)
                    row['AvailabilityBreaks'].append(AvailabilityBreak)


                row['ValidatingCarrier'] = option['TPA_Extensions']['ValidatingCarrier']['Code']
                try: row['DiversitySwapper'] = option['TPA_Extensions']['DiversitySwapper']['WeighedPriceAmount']
                except Exception as e: row['DiversitySwapper'] = 'na'






                ROWS.append(row)

            if DEBUG: print(f'Total Number of Options Found: {len(ROWS)}')

            df = pd.DataFrame(ROWS)
            #df.to_csv('THIS_IS_IT.csv')
            simple_log('OK.')
            if RET == 'LIST':
                return ROWS





            if RET=='ROCK':
                DF = []
                print ('ROCK')
                row_id = 0
                for row in ROWS:
                    print (row_id)
                    D = {}
                    IT = []
                    D['itinerary']=IT
                    D['elapsed_time'] = row['total_elapsed_time']
                    D['currency'] = row['currency']

                    leg_id = 0
                    for leg in row['MarketingAirline']:
                        L = []
                        IT.append(L)
                        flight_id = 0
                        for flight in leg:
                            L.append({'flight_number': row['FlightNumber'][leg_id][flight_id],
                                    'MarketingAirline': row['MarketingAirline'][leg_id][flight_id],
                                    'DepartureDateTime': row['DepartureDateTime'][leg_id][flight_id],
                                    'ArrivalDateTime': row['ArrivalDateTime'][leg_id][flight_id],
                                    'DepartureAirport': row['DepartureAirport'][leg_id][flight_id],
                                    'ArrivalAirport': row['ArrivalAirport'][leg_id][flight_id],
                                    #'BookingCode': row['BookingCodes'][leg_id][flight_id],
                                    'OperatingAirline': row['OperatingAirline'][leg_id][flight_id],
                                    'OperatingFlightNumber': row['OperatingFlightNumber'][leg_id][flight_id],


                            })
                            flight_id += 1
                        leg_id += 1
                    row_id +=1

                    bag_aux = lambda x: [[v.values() for v in b] for b in x]
                    D['fare_info'] = []
                    price_id = 0
                    for each in row['prices']:
                        D['fare_info'].append(
                                {'total_price': row['prices'][price_id],
                                'base_fare': row['BaseFares'][price_id],
                                'bags':  bag_aux( row['BaggageAllowances'][price_id] ),

                                }
                        )


                        price_id += 1

                    DF.append(D)
                return pandas.DataFrame(DF)
            return df
