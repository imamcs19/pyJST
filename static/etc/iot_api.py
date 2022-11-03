import requests
# import re
# from bs4 import BeautifulSoup
import sqlite3
# import time
from datetime import datetime
# from time import strftime
import pytz
Date = str(datetime.today().astimezone(pytz.timezone('Asia/Jakarta')).strftime('%d-%m-%Y %H:%M:%S'))

# def scrapMarketwatch(address):
#     #creating formatting data from scrapdata
#     r = requests.get(address)
#     c = r.content
#     sup = bs(c,"html.parser")
#     # print(sup)
#     return sup
def F2C(f_in):
    return (f_in - 32)* 5/9

def Kelvin2C(k_in):
  return (k_in-273.15)

def connect_db():
    import os.path

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # db_path = os.path.join(BASE_DIR, "data.db")
    # jika ingin membuat DB baru untuk penyimpanannya
    # db_path = os.path.join(BASE_DIR, "/home/bigdatafga/mysite/data2.db")
    
    db_path = os.path.join(BASE_DIR, "/home/bigdatafga/mysite/data.db")
    
    # with sqlite3.connect(db_path) as db:

    return sqlite3.connect(db_path)


# simpan ke db
conn = connect_db()
db = conn.cursor()

db.execute("""CREATE TABLE IF NOT EXISTS data_suhu_dll (date DATETIME, kota TEXT, suhu_dlm_celcius TEXT, precipitation_curah_hujan_dlm_persen TEXT, humidity_kelembaban_dlm_persen TEXT, wind_angin_dlm_km_per_jam TEXT) """)

# siang
# jam 6 pagi - 18.00
#
# malam
# jam 18.00 - 6 pagi

# Ref. daerah di negara lain yang selisihnya -/+[6/1/2], dll jam dgn Indonesia
# [0] https://time.is/id/Jakarta#time_zone
# [1] https://www.kompas.com/skola/read/2021/03/04/120155469/tabel-perbedaan-waktu-di-indonesia-dengan-negara-lainnya
#

# Perbedaan waktu dari Jakarta:
# Los Angeles		−14 jam
# Chicago			−12 jam
# New York City	    −11 jam
# Toronto			−11 jam
# São Paulo	    	−10 jam
# UTC				−7 jam
# Lagos		    	−6 jam
# London			−6 jam
# Johannesburg  	−5 jam
# Kairo		    	−5 jam
# Paris		    	−5 jam
# Zurich			−5 jam
# Istanbul	    	−4 jam
# Moskwa			−4 jam
# Dubai		    	−3 jam
# Mumbai			−1,5 jam
# Hong Kong	    	+1 jam
# Shanghai	    	+1 jam
# Singapura	    	+1 jam
# Tokyo		    	+2 jam
# Sydney			+4 jam

# list_kota = ['Jakarta, Indonesia As of 8:52 am WIB',\
# 'Los Angeles, CA As of 6:40 pm PDT',\
# 'Chicago, IL As of 8:54 pm CDT', \
# 'New York City, NY As of 9:54 pm EDT',\
# 'Toronto, Ontario, Canada As of 9:43 pm EDT',\
# 'São Paulo, São Paulo, Brazil As of 10:58 pm BRT', \
# 'Lagos, Lagos, Nigeria As of 2:51 am WAT', \
# 'London, England, United Kingdom As of 2:53 am BST', \
# 'Johannesburg, Gauteng, South Africa As of 4:01 am SAST',\
# 'Cairo, Cairo, Egypt As of 3:50 am EET', \
# 'Paris, France As of 3:56 am CEST', \
# 'Zurich, Zürich, Switzerland As of 3:49 am CEST', \
# 'Istanbul, Turkey As of 4:58 am EET', \
# 'Moscow, Moscow, Russia As of 4:55 am MSK', \
# 'Dubai, Dubai, United Arab Emirates As of 6:02 am GST', \
# 'Mumbai, Maharashtra, India As of 7:24 am IST',\
# 'Hong Kong, People's Republic of China As of 10:01 am HKT',\
# 'Shanghai, People's Republic of China As of 10:01 am CST',\
# 'Singapore, Central, Singapore As of 9:59 am SGT',\
# 'Tokyo, Tokyo Prefecture, Japan As of 10:52 am JST',\
# 'Sydney, New South Wales, Australia As of 1:02 pm AEDT']

list_kota = ['Jakarta','Los Angeles','Chicago','New York City','Toronto','São Paulo', \
             'Lagos', 'London', 'Johannesburg', 'Kairo', 'Paris', 'Zurich', 'Istanbul', 'Moskwa', 'Dubai', \
            'Mumbai','Hong Kong','Shanghai','Singapura','Tokyo','Sydney']


for nama_kota in list_kota:

  each_list_link='http://api.weatherapi.com/v1/current.json?key=to2181c95fd6d746e9a1331323220104099&q='+nama_kota
  resp=requests.get(each_list_link)

  #http_respone 200 means OK status
  if resp.status_code==200:
      resp=resp.json()
      suhu = resp['current']['temp_c']
      curah_hujan = resp['current']['precip_mm']
      lembab = resp['current']['humidity']
      angin = resp['current']['wind_mph']
  else:
      # print("Error")
      suhu = '-'
      curah_hujan = '-'
      lembab = '-'
      angin = '-'

  db.execute("""INSERT INTO data_suhu_dll (date, kota, suhu_dlm_celcius, precipitation_curah_hujan_dlm_persen, humidity_kelembaban_dlm_persen, wind_angin_dlm_km_per_jam) VALUES (?,?,?,?,?,?) """,(Date,nama_kota,suhu,curah_hujan,lembab,angin))


conn.commit()
db.close()
conn.close()
