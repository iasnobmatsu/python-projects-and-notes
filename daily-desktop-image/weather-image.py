import requests
import json

id_file=open("citylist.json","r")
id_data=json.load(id_file)
city_list={}
for city in id_data:
    city_name=city["name"].lower()
    city_id=city["id"]
    city_list[city_name]=city_id

print(city_list['chapel hill'])


# r=requests.get()