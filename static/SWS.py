import urllib
import requests
data = ''
from django.template.loader import get_template

def send_BFM(bfm_payload):

    security_token = 'T1RLAQLToGvksYXTyVsMjJmVzDxW5VTFLRA213nsr6Zov/3aEMbIb3nuAACw8nNn6L6NhXBE+U0AasMhkq0ogVx4YOGFY3iPvkwKJiWsTmzCEVfTnKjfNjMsYJn3dhxPPwHKZ+EIW6uK21UzyKkMa43Cv4mDyU5a3T91e1lisav/yll3Y6AyzgamPClcDSObxg4VR14cCuZxJ93J+EgFs1XHX76gGwNPiBWAeT33sxohwolgzMHLiUXK0Qjjxh2xwpxAawtvnchNtvpAhi/NXviCBkYsR+Bitki4n7Y*'
    context = {'fullRequest':bfm_payload,'conversationId':'santiago.gonzalez@Sabre.com','from':'Me', 'to':'you','BargainAction':'BargainFinderMaxRQ',
        'messageId':'TESTING','timestamp':'timestamp','ttl':'ttl',
        'securityToken':security_token
    }
    proxies = {
      "https": "https://sg0216333:Cactus@123@www-ad-proxy.sabre.com:80",
    }


    headers = {"Content-type": "text/xml"}
    bfm_rq_template = get_template('bfmrq.xml').render(context)
    r = requests.post(url = 'https://sws-crt.cert.sabre.com', data = bfm_rq_template, headers=headers, proxies=proxies)
    print ('$'*50)

    return (r.text)
