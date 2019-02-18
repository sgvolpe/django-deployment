import datetime, urllib, requests, base64, json

from django.template.loader import get_template

PROXY_USER='sg0216333'
PROXY_PASSWORD='Cactus@123'
PROXY_URL='www-ad-proxy.sabre.com:80'

ENV_URL = 'https://sws-crt.cert.sabre.com'
#ENV_URL = 'https://api.test.sabre.com'
ENV_URL = 'https://webservices.havail.sabre.com'

DEBUG=True
data = ''
security_token = ''


class Rest_Handler:
    token = None

    parameters = {"user": b"7971", "group": b"FF9A", "domain": b"AA", "password": b"WSPTS10"}
    URLS = {"PROD": "https://api.sabre.com", "CERT": "https://api.test.sabre.com"}

    def __init__(self, env="CERT", parameters=parameters, debug=False, token="", automaticAuth=True ):
        self.URL = self.URLS.get(env)
        self.parameters = parameters

        try:
            if token != "": self.token = token
            else:
                AuthenticationRS = self.AuthenticationRQ(self.URL, self.parameters)
                self.token = json.loads(AuthenticationRS.text)["access_token"]
        except Exception as e:
            if DEBUG: print ("Error while Authenticating "+ str(e))



    def encodeBase64(self, stringToEncode):
        return base64.b64encode(stringToEncode) #.encode("utf-8")


    def AuthenticationRQ(self, url, parameters, version='v2'):
        print (version, parameters, url)
        if version == 'v2': url = url + "/v2/auth/token"
        elif version == 'v1': url = url + "/v1/auth/token"
        user = parameters["user"]
        group = parameters["group"]
        domain = parameters["domain"]
        password = parameters["password"]

        encodedUserInfo =  self.encodeBase64(b"V1:" + user + b":" + group + b":" + domain)
        encodedPassword =  self.encodeBase64(password)
        encodedSecurityInfo = self.encodeBase64(encodedUserInfo + b":" + encodedPassword)

        proxies = {"https": f"https://{PROXY_USER}:{PROXY_PASSWORD}@{PROXY_URL}",}
        data = {'grant_type':'client_credentials'}
        headers = {'content-type': b'application/x-www-form-urlencoded ','Authorization': b'Basic ' + encodedSecurityInfo, 'Accept-Encoding': 'gzip,deflate'}
        response = requests.post(url, headers=headers,data=data, proxies=proxies)
        print (response)
        if(response.status_code != 200):
            if DEBUG: print ("ERROR: I couldnt authenticate")
            self.token = json.loads(response.text)["access_token"]
        if DEBUG: print ("Authentiaction Success.")
        return response

    def parse_bfm_rs(self, f_name):
        itin = ''
        rs = json.loads(open(f_name).read())
        price =rs['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary'][0]['AirItineraryPricingInfo'][0]['ItinTotalFare']['TotalFare']['Amount']
        OriginDestinationOptions = rs['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary'][0]['AirItinerary']['OriginDestinationOptions']
        itin = itin +'|'+ bfm_itin(OriginDestinationOptions)
        return {'price':float(price), 'itin': itin}


    def bfm(self, ori = 'LON',des= 'SYD', dep_date='2019-06-15', ret_date='2019-06-22'
        , arr_win='00002359',dep_win='00002359'):
        #if DEBUG: print (f'BFM {ori, des, dep_date, ret_date, arr_win,dep_win}')
        arrivalwindow=None
        dates = None
        parameters = {"user": b"7971", "group": b"FF9A", "domain": b"AA", "password": b"WSPTS10"}

        if self.token is None:
            AuthenticationRS = self.AuthenticationRQ("https://api.test.sabre.com", parameters)
            self.token = json.loads(AuthenticationRS.text)["access_token"]
        #token = 'T1RLAQI1pqVlRsCZ4InNy34iLw2yiS7USRAcyZlcpHi9sXgDTnTjwhG+AACw8a8LUHyeQ16fN7eqM0bshFQX13FpLp2PnmLtdPNOLN+e8XLSP8P8pqOdNbrbgG+pAG/yUFEuaX7+yLaPgtajFJ8D/rWU95A+1galNQ5w6Ro5FlZK5O/c8kf4L+1Bmmen++V4fvx6/wskmEcRU/3HkN+1PtWJlUMcvi+ivD+3bd37/YBrTv4dWTI0nsLieKNE4ORLA+z0AiHB5kMPE0E2ZPkODvPtAjxdRayv5Z0yYdE*'

        url = 'https://api.test.sabre.com/v4.3.0/shop/flights?mode=live'
        headers = {'content-type': 'application/json','Authorization': 'Bearer ' + str(self.token)}

        payload = json.loads(open('bfmrq.json').read())
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['OriginLocation']['LocationCode'] = ori
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['DestinationLocation']['LocationCode'] = des
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['DepartureDateTime'] = f'{dep_date}T11:00:00'
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['ArrivalWindow'] = arr_win
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['DepartureWindow'] = dep_win
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['OriginLocation']['LocationCode'] = des
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['DestinationLocation']['LocationCode'] = ori
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['DepartureDateTime'] = f'{ret_date}T11:00:00'
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['ArrivalWindow'] = arr_win
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['DepartureWindow'] = dep_win

        #TODO: add numitins
        rq = open(f"requests/bfm_{ori}_{des}_{dep_date}_{ret_date}_rs.json",'w')
        rq.write(json.dumps(payload))
        try:
            itin = ''
            response = requests.post(url, headers=headers,data=json.dumps(payload))
            rs = json.loads(response.text)
            price =rs['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary'][0]['AirItineraryPricingInfo'][0]['ItinTotalFare']['TotalFare']['Amount']
            rs_file = open(f"responses/bfm_{ori}_{des}_{dep_date}_{ret_date}_rs.json",'w')
            rs_file.write(response.text)


            OriginDestinationOptions = rs['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary'][0]['AirItinerary']['OriginDestinationOptions']
            itin = itin +'|'+ bfm_itin(OriginDestinationOptions)

            return {'price':float(price), 'itin': itin}
        except Exception as e:
            print (rs)
            print ('ERROR', str(e))
            return {'price':999999, 'itin': itin}



    def bfmad(self, ori = 'LON',des= 'SYD', dep_date='2019-06-15', ret_date='2019-06-22'
        , arr_win='00002359',dep_win='00002359', output_path='new_dash/static/virtualinterlining/requests'):
        #if DEBUG: print (f'BFM {ori, des, dep_date, ret_date, arr_win,dep_win}')
        arrivalwindow=None
        dates = None
        parameters = {"user": b"7971", "group": b"FF9A", "domain": b"AA", "password": b"WSPTS10"}

        if self.token is None:
            AuthenticationRS = self.AuthenticationRQ("https://api.test.sabre.com", parameters)
            self.token = json.loads(AuthenticationRS.text)["access_token"]
        #token = 'T1RLAQI1pqVlRsCZ4InNy34iLw2yiS7USRAcyZlcpHi9sXgDTnTjwhG+AACw8a8LUHyeQ16fN7eqM0bshFQX13FpLp2PnmLtdPNOLN+e8XLSP8P8pqOdNbrbgG+pAG/yUFEuaX7+yLaPgtajFJ8D/rWU95A+1galNQ5w6Ro5FlZK5O/c8kf4L+1Bmmen++V4fvx6/wskmEcRU/3HkN+1PtWJlUMcvi+ivD+3bd37/YBrTv4dWTI0nsLieKNE4ORLA+z0AiHB5kMPE0E2ZPkODvPtAjxdRayv5Z0yYdE*'

        url = 'https://api.test.sabre.com/v4.3.0/shop/altdates/flights?mode=live'
        headers = {'content-type': 'application/json','Authorization': 'Bearer ' + str(self.token)}
        payload_path='new_dash/static/virtualinterlining/BFMADRQ.json'
        payload = json.loads(open(payload_path).read())
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['OriginLocation']['LocationCode'] = ori
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['DestinationLocation']['LocationCode'] = des
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['DepartureDateTime'] = f'{dep_date}T11:00:00'
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['ArrivalWindow'] = arr_win
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][0]['DepartureWindow'] = dep_win
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['OriginLocation']['LocationCode'] = des
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['DestinationLocation']['LocationCode'] = ori
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['DepartureDateTime'] = f'{ret_date}T11:00:00'
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['ArrivalWindow'] = arr_win
        payload['OTA_AirLowFareSearchRQ']['OriginDestinationInformation'][1]['DepartureWindow'] = dep_win

        #todo: add numitins
        rq = open(f"{output_path}/bfm_{ori}_{des}_{dep_date}_{ret_date}_adrs.json",'w')
        rq.write(json.dumps(payload))
        try:
            itin = ''
            response = requests.post(url, headers=headers,data=json.dumps(payload))
            rs = json.loads(response.text)
            price =rs['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary'][0]['AirItineraryPricingInfo'][0]['ItinTotalFare']['TotalFare']['Amount']
            rs_file = open(f"{output_path}/bfm_{ori}_{des}_{dep_date}_{ret_date}_adrs.json",'w')
            rs_file.write(response.text)


            OriginDestinationOptions = rs['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary'][0]['AirItinerary']['OriginDestinationOptions']
            itin = itin +'|'+ bfm_itin(OriginDestinationOptions)

            bfm_itineraries = []
            for it in get_bfm_itineraries(bfm_rs_json=rs):
                ITIN = {'ts':it['ts'],'it_ori':ori, 'it_des':des, 'd_date':dep_date, 'r_date':ret_date, 'path': f'{ori}-{des}',
                        'price':it['price'],'d_date':it['ddate'], 'r_date':it['rdate'], 'cheaper':False
                       }
                bfm_itineraries.append(ITIN)

            return {'price':float(price), 'itin': itin, 'bfm_itineraries':bfm_itineraries}
        except Exception as e:
            print (rs)
            print ('ERROR', str(e))
            return {'price':999999, 'itin': itin}



    def BargainFinderMaxRQ(self, payload, debug=False, token='', version='3.3.0', others=''):
        if token != '':
            self.token = token
        url = self.URL + "/v"+version+"/shop/flights?mode=live"
        if(debug): url += "&debug=true"
        if others != '': url += others
        data = payload
        headers = {'content-type': 'application/json','Authorization': 'Bearer ' + str(self.token)}

        response = requests.post(url, headers=headers,data=data)
        return response

    def AlternateDateRQ(self, payload, debug=False, token=None, version='3.3.0'):
        url = self.URL + "/v"+version+"/shop/altdates/flights?mode=live"
        if(debug): url += "&debug=true"
        data = payload
        headers = {'content-type': 'application/json','Authorization': 'Bearer ' + str(self.token)}
        response = requests.post(url, headers=headers,data=data)
        return response


    def AdvancedCalendarSearchRQ(self, payload, debug=False, token=None, version='3.3.0', pos='US'):
        url = self.URL + "/v"+version+"/shop/calendar/flights?"
        url += "pointofsalecountry=" + pos
        if(debug): url += "&debug=true"
        data = payload
        headers = {'content-type': 'application/json','Authorization': 'Bearer ' + str(self.token)}
        response = requests.post(url, headers=headers,data=data)
        return response

    def AlternateAirportShopRQ(self, payload, debug=False, token=None, version='3.3.0'):
        url = self.URL + "/v"+version+"/shop/altairports/flights?mode=live"
        if(debug): url += "&debug=true"
        data = payload
        headers = {'content-type': 'application/json','Authorization': 'Bearer ' + str(self.token)}
        response = requests.post(url, headers=headers,data=data)
        return response



    def InstaFlightSearch(self, url_parameters ):
        url = self.URL + url_parameters
        headers = {'Authorization': 'Bearer ' + str(self.token)}
        response = requests.get(url, headers=headers)
        return response

    def GenericRestCall(self, url, payload, method='POST' ):
        url =  url
        headers = {'Authorization': 'Bearer ' + str(self.token)}
        if method =='GET':
            response = requests.get(url, headers=headers)
        if method == 'POST':
            data = payload
            response = requests.post(url, headers=headers,data=data)
        return response

    def LowFareForecats(self, url_parameters ):
        url = self.URL +'/v2/forecast/flights/fares?'+ url_parameters
        headers = {'Authorization': 'Bearer ' + str(self.token)}
        response = requests.get(url, headers=headers)
        return response

    def LowFareHistory(self, url_parameters ):
        url = self.URL +'/v1/historical/shop/flights/fares?'+ url_parameters
        headers = {'Authorization': 'Bearer ' + str(self.token)}
        response = requests.get(url, headers=headers)
        return response

def printdebug(func):
    def wrapper():
        print (f)
        func()


def log_bfm(text):
    log_path = 'dashboard/bfm_stats.csv'
    log_file = open(log_path, 'a')
    log_file.write(str( datetime.datetime.now() ) + ',' + str(text) + '\n')
    log_file.close()


import functools
import time

def log(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        log_path = 'dashboard/bfm_stats.csv'
        log_file = open(log_path, 'a')
        log_file.write(str( datetime.datetime.now() ) + ',' + str(text) + '\n')
        log_file.close()
        return value
    return wrapper_timer


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

@timer
def send_BFM(bfm_payload,params={},ENV_URL=ENV_URL):
    if globals()['security_token'] == '':
        globals()['security_token'] = Rest_Handler().token

    context = {'fullRequest':bfm_payload,'conversationId':'santiago.gonzalez@Sabre.com'+'-'.join(list(params.values())),'from':'Me', 'to':'you','BargainAction':'BargainFinderMaxRQ',
        'messageId':'-'.join(list(params.values())),'timestamp':'timestamp','ttl':'ttl','securityToken':globals()['security_token']
    }
    proxies = {"https": f"https://{PROXY_USER}:{PROXY_PASSWORD}@{PROXY_URL}",}
    headers = {"Content-type": "text/xml"}
    bfm_rq_template = get_template('bfmrq.xml').render(context)

    t0 = datetime.datetime.now()
    rs = requests.post(url = ENV_URL, data = bfm_rq_template, headers=headers, proxies=proxies)
    t1 = datetime.datetime.now()
    response_time=(t1-t0).total_seconds()*1000
    payload_size=len(rs.content)

    response = {'rs_text':rs.text,'response_time':response_time,'payload_size':payload_size}
    log_bfm('-'.join(list(params.values())) + ',' + ENV_URL + ','+ str(response_time) +','+ str(payload_size) )
    return response

def get_bfm_itineraries(f_name=None, bfm_rs_json=None):
    itineraries = []
    retorno = []
    if bfm_rs_json is None:
        rs = json.loads(open(f_name).read())
    else: rs = bfm_rs_json
    OriginDestinationOptions_list = rs['OTA_AirLowFareSearchRS']['PricedItineraries']['PricedItinerary']
    for OriginDestinationOptions in OriginDestinationOptions_list:
        option = {'ts': str(datetime.datetime.now() )}
        legs = OriginDestinationOptions['AirItinerary']['OriginDestinationOptions']['OriginDestinationOption']
        price = OriginDestinationOptions['AirItineraryPricingInfo'][0]['ItinTotalFare']['TotalFare']['Amount']

        for leg in legs:
            flights = leg['FlightSegment']
            for flight in flights:
                ArrivalAirport= flight['ArrivalAirport']['LocationCode']
                DepartureAirport= flight['DepartureAirport']['LocationCode']

        option['price']=price
        option['ddate']=legs[0]['FlightSegment'][0]['DepartureDateTime'][:10]
        option['rdate']=legs[1]['FlightSegment'][0]['DepartureDateTime'][:10]
        retorno.append(option)
    return retorno

 ####Include Cache from DB


def bfm_itin(OriginDestinationOptions):
    IT = []
    legs = OriginDestinationOptions['OriginDestinationOption']
    for leg in legs:
        LEG=[]
        IT.append(LEG)
        flights = leg['FlightSegment']
        for flight in flights:
            MarketingAirline= flight['MarketingAirline']['Code']
            ArrivalAirport= flight['ArrivalAirport']['LocationCode']
            DepartureAirport= flight['DepartureAirport']['LocationCode']
            DepartureDateTime = flight['DepartureDateTime']
            ArrivalDateTime = flight['ArrivalDateTime']
            FlightNumber = flight['FlightNumber']
            F= f'{DepartureDateTime}-{DepartureAirport}-{MarketingAirline}{FlightNumber}-{ArrivalAirport}-{ArrivalDateTime}'

            LEG.append(F)

    return '||'.join( ['|'.join([f for f in leg]) for leg in IT] )
