import requests
import pandas as pd


def get_lng_lat(address):
    url = 'https://restapi.amap.com/v3/geocode/geo'
    params = {
        'address': address,
        'key': '4ac30f2d0e3b3f9f5b3d8b2fee881d87'
    }

    response = requests.get(url, params)
    data = response.json()

    if data['status'] == '1' and int(data['count']) > 0:
        location = data['geocodes'][0]['location']
        lng, lat = location.split(',')
        return lng, lat
    else:
        return None, None


def batch_get_lng_lat(addresses):
    result = []

    for address in addresses:
        lng, lat = get_lng_lat(address)
        result.append({'address': address, 'lng': lng, 'lat': lat})

    return pd.DataFrame(result)


df = pd.read_excel('../data/attractions/attraction.xlsx')

addresses = df["Attraction"]
df_address = batch_get_lng_lat(addresses)

df['lng'] = df_address['lng']
df['lat'] = df_address['lat']

# If you are sure to save the converted addresses, remove the comment
# df.to_excel('../data/attractions/attraction.xlsx', index=False)
