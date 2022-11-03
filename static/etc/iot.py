import requests
# import re
from bs4 import BeautifulSoup
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

def connect_db():
    import os.path

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # db_path = os.path.join(BASE_DIR, "data.db")
    # jika ingin membuat DB baru untuk penyimpanannya
    # db_path = os.path.join(BASE_DIR, "/home/bigdatafga/mysite/data2.db")
    
    db_path = os.path.join(BASE_DIR, "/home/bigdatafga/mysite/data.db")

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

list_link_kota = ['https://weather.com/weather/today/l/363e9580fc9dcc8e3ab95158a9435cd81609c821e02330d47b121ea1d65771dc',\
             'https://weather.com/weather/today/l/a4bf563aa6c1d3b3daffff43f51e3d7f765f43968cddc0475b9f340601b8cc26',\
             'https://weather.com/weather/today/l/e0abde3003a88dedecad92fedc96375000c16843287a51dbf2cd92f062217180',\
             'https://weather.com/weather/today/l/96f2f84af9a5f5d452eb0574d4e4d8a840c71b05e22264ebdc0056433a642c84',\
             'https://weather.com/weather/today/l/d8ccf908e3c4c748e232720575df7cdbca6e0f1b412bca8595d8a28d0c28e8bc',\
             'https://weather.com/weather/today/l/63e18eea74a484c42c3921cf52a8fec98113dbb13f6deb7c477b2f453c95b837', \
             'https://weather.com/weather/today/l/7d391501221842b79c58c5260dbafbe2305deffed37a075972092251243a4ad8', \
             'https://weather.com/weather/today/l/ae8230efd4bc57fdf721a02c7eb2b88c56aa6e71d73666328e33af3ea2039032132e24ae91b6a07862c5091a9d95a4b8',\
             'https://weather.com/weather/today/l/bcc7d09eb5f3638e22dbb8465eeb4065b24ccc12e1b02891643c18d30410ba41', \
             'https://weather.com/weather/today/l/2baa93f2531b18395e9b0062c11ffee82838615b3ac6141394235eb734bac64d', \
             'https://weather.com/weather/today/l/1a8af5b9d8971c46dd5a52547f9221e22cd895d8d8639267a87df614d0912830', \
             'https://weather.com/weather/today/l/151d93a8aa1a5fa8c93142a2499b472960ea57c494977eecdd6810dabed490df', \
             'https://weather.com/weather/today/l/33d1e415eb66f3e1ab35c3add45fccf4512715d329edbd91c806a6957e123b49', \
             'https://weather.com/weather/today/l/34f2aafc84cff75ae0b014754856ea5e7f8ddf618cf9735549dfb5e016c28e10', \
             'https://weather.com/weather/today/l/af60f113ba123ce93774fed531be2e1e51a1666be5d6012f129cfa27bae1ee6c', \
             'https://weather.com/weather/today/l/e1bbaf5ba44a74170e3bb9f892416301c36b3b17f37e1a666c6e1213de0f5668',\
             'https://weather.com/weather/today/l/8f0658124f5f5b725ca5ed254decc028fd2099a8ac1843faa2ceb206c9b464d1',\
             'https://weather.com/weather/today/l/7f14186934f484d567841e8646abc61b81cce4d88470d519beeb5e115c9b425a',\
             'https://weather.com/weather/today/l/bfbafb71cea3672231349f36b198478ecc3d5fd524d0918b8051ee838f743675',\
             'https://weather.com/weather/today/l/4ba28384e2da53b2861f5b5c70b7332e4ba1dc83e75b948e6fbd2aaceeeceae3',\
             'https://weather.com/weather/today/l/98ef17e6662508c0af6d8bd04adacecde842fb533434fcd2c046730675fba371']

for each_list_link in list_link_kota:
  #open with GET method
  resp=requests.get(each_list_link)

  nama_kota = list_kota[list_link_kota.index(each_list_link)]

  #http_respone 200 means OK status
  if resp.status_code==200:
      soup=BeautifulSoup(resp.text,'html.parser')
      divs = soup.find("span", {"class": "CurrentConditions--tempValue--3a50n","data-testid":"TemperatureValue"}, partial=False)
      int_hasil = int(''.join([n for n in divs.text if n.isdigit()]))
      suhu = str(F2C(int_hasil))
      curah_hujan = '-'
      lembab = '-'
      angin = '-'
      # print(F2C(int_hasil))
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
