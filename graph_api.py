from configparser import ConfigParser
import facebook
import json
import requests
from pip._vendor.distlib.compat import raw_input


def config_parse():

    config = ConfigParser()
    config.read('configurations.ini')

    token = config.get('tokens', 'per_token')

    return token


def load_data():

    #token = config_parse()
    token = '314822702826969|tpfcb2mMyslct_L1P3UqTkZqF9Q'
    token = 'EAAEeVF9ehdkBANETcSUEIGg93Emj0Cu0UlcDzw8Kd5lLsdl9W6VpfUiS9qS72y5qzJDgjUI5ilUSsU2lmAKmy8ZCZC4ZAlZBCpL9VWbu3Y5idExzJh0BkO5Tk5M0AFZB7SJLZCqcMt2vNXjYJxYIMMMJopqGX056gZCctSrZAoD5n8cZBLJj2ugm7bG6IOmmhzxZBZAMt4iyeXVcdY5Idqp88ZB0'
    graph = facebook.GraphAPI(access_token=token, version=3.1)
    events = graph.request('/EAADu10VNscABAIzzLX9OfSG9ddlJIo5IlgsGEF5d1XwkirzRutvblEkEAWS9esaqhH2ZAiFAqw3B9tq4eQFKLRYETHbuvrDFl2FO0w8qxQqnyqkT4mm35JabEZB655T89pS9srMMd3GuHtB82pNDLL8ATYptNE4wdVLxS7HJqxl3GRfQkjuwsAeyTswekyQ5llCKEKzJpwzHMnFdtV/feed')
    print(events)
    '''
    graph = facebook.GraphAPI(token)
    page_name = raw_input("Enter a page name: ")
    #name = CMOMaharashtra
    # list of required fields
    #fields = ['id', 'name', 'about', 'likes', 'link', 'band_members']

    fields = ['name']

    fields = ','.join(fields)

    page = graph.get_object(page_name, fields=fields)

    print(json.dumps(page, indent=4))

    '''

def extend_token_expiry():

    app_id = '262608348099008'
    app_secret = 'b6542031bf3d05427fbca5d22d3affde'
    user_short_token = 'EAADu10VNscABAN0wVlV6vleuNWBvoliDgah5JVConnCcV4pgBW7g6mX3FSOliRtMMN74eAqkt4ISNT4ZAJPH0mDHLZBQ4BBhZAomWNCdz2rvcJZBHg3BIhSHiyNZBDLOw8KZBow8lPCDthg39ZAlorCoXs0ZBtRJrZAR4ZBFGnx3AL8LTwoMdWcsSE'
    access_token_url = "https://graph.facebook.com/oauth/access_token?grant_type=fb_exchange_token&client_id={}&client_secret={}&fb_exchange_token={}".format(
        app_id, app_secret, user_short_token)

    r = requests.get(access_token_url)

    access_token_info = r.json()
    user_long_token = access_token_info['access_token']

    print(access_token_info)
    #print(user_long_token)

    #permanent access token
    graph = facebook.GraphAPI(access_token=user_long_token,version="3.1")

    pages_data = graph.get_object("/me/accounts")

    print(pages_data)




def main():
    load_data()
    #extend_token_expiry()



if __name__ == '__main__':
    main()