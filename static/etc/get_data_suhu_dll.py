import requests
from bs4 import BeautifulSoup as bs
import sqlite3
import time
from datetime import datetime
# from time import strftime
import pytz
Date = str(datetime.today().astimezone(pytz.timezone('Asia/Jakarta')).strftime('%d-%m-%Y %H:%M:%S'))

def scrapMarketwatch(address):
    #creating formatting data from scrapdata
    r = requests.get(address)
    c = r.content
    sup = bs(c,"html.parser")
    # print(sup)
    return sup
def F2C(f_in):
    return (f_in - 32)* 5/9

def connect_db():
    import os.path

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "fga_big_data_rev2.db")
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

# #
list_kota = ['Malang','Lamongan','Jakarta','Los Angeles','Chicago','New York City','Toronto','São Paulo', \
             'Lagos', 'London', 'Johannesburg', 'Kairo', 'Paris', 'Zurich', 'Istanbul', 'Moskwa', 'Dubai', \
            'Mumbai','Hong Kong','Shanghai','Singapura','Tokyo','Sydney']

# list_kota = ['Jakarta','Los Angeles','Chicago','New York City','Toronto','São Paulo', \
#              'Lagos', 'London', 'Johannesburg', 'Kairo', 'Paris', 'Zurich', 'Istanbul', 'Moskwa', 'Dubai', \
#             'Mumbai','Hong Kong','Shanghai','Singapura','Tokyo','Sydney']

# n_byk_ambil_data = 10
# for iter_get in range(n_byk_ambil_data):
# flag = True
# while flag:
for nama_kota in list_kota:
  # print(nama_kota)
  url = 'https://www.google.com/search?q=suhu+malang&ei=5z9TYd-QN9_Yz7sPuvid2AM&oq=suhu+malang&gs_lcp=Cgdnd3Mtd2l6EAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsANKBAhBGABQAFgAYPDMAWgBcAJ4AIABAIgBAJIBAJgBAMgBCMABAQ&sclient=gws-wiz&ved=0ahUKEwifoIHPiKLzAhVf7HMBHTp8BzsQ4dUDCA4&uact=5'
  each_list_link = url.replace('malang', nama_kota)
  soup_level2 = scrapMarketwatch(each_list_link)

  #Get suhu suatu kota
  str_hasil = soup_level2.find_all("div", class_="BNeawe iBp4i AP7Wnd")[1].get_text()
  int_hasil = int(''.join([n for n in str_hasil if n.isdigit()]))

  #print('Suhu',nama_kota.title(),'= ', F2C(int_hasil))

  suhu = str(F2C(int_hasil))
  curah_hujan = '-'
  lembab = '-'
  angin = '-'

  db.execute("""INSERT INTO data_suhu_dll (date, kota, suhu_dlm_celcius, precipitation_curah_hujan_dlm_persen, humidity_kelembaban_dlm_persen, wind_angin_dlm_km_per_jam) VALUES (?,?,?,?,?,?) """,(Date,nama_kota,suhu,curah_hujan,lembab,angin))


        # set sleep 3 detik
    # time.sleep(3600) # per 1 jam
    # time.sleep(10800) # per 3 jam
    # time.sleep(3)


# CREATE TABLE data_suhu_dll (
#     date                                 DATETIME,
#     kota                                 TEXT,
#     suhu_dlm_celcius                     TEXT,
#     precipitation_curah_hujan_dlm_persen TEXT,
#     humidity_kelembaban_dlm_persen       TEXT,
#     wind_angin_dlm_km_per_jam            TEXT
# );





conn.commit()
db.close()
conn.close()

# Rujukan waktu Internasional bukan lagi menggunakan GMT (Greenwich Mean Time), tetapi menggunakan UTC (Universal Time Coordinated)
# UTC +7 memiliki arti bahwa wilayah Indonesia bagian barat memiliki selisih waktu tujuh jam lebih cepat dari UTC.
# Misalnya, jam 00.00 UTC setara dengan jam 07.00 WIB.
#
# London, Inggris UTC +00 (artinya jam di London, Inggris sama dengan jam UTC.
# Apabila jam UTC menunjukkan angka 10.00, maka di London, Inggris juga menunjukkan angka 10.00).
#
# Base suhu++ di Ibu Kota beberapa negara:
# kandidat, misal di jakarta Indonesia jam 7 pagi:
# ==============================================================================
# + 0 jam
# jakarta, Indonesia jam 7 pagi
#
# ========
# + 1 jam
# ========
# Kuala Lumpur, Malaysia jam 8 pagi | +1 jam | Hari yang sama
# Manila, Filipina jam 8 pagi | +1 jam | Hari yang sama
# Bandar Seri Begawan, Brunei Darussalam jam 8 pagi | +1 jam | Hari yang sama
#
# + 2 jam
# ========
# Seoul, korea selatan jam 9 pagi |  +2 jam Hari yang sama
# Dili, Timor Leste jam 9 pagi |  +2 jam Hari yang sama
#
# + 3 jam
# ======== --> jam 10 pagi
#
# + 4 jam
# ========
# Australia (Canberra) jam 11 siang | +4 jam Hari yang sama
#
# + 5 jam
# ======== --> jam 12 siang
#
# + 6 jam
# ======== --> jam 13 siang
#
#
# + 7 jam
# ======== --> jam 14 siang

# Moskow, Rusia
#
#
# amsterdam, Belanda jam 1 siang | -6 jam | Hari yang sama
# Paris, Perancis jam 1 siang | -5 jam | Hari yang sama
# Roma, Italia jam 1 siang | -6 jam | Hari yang sama
#
# London, Inggris, jam 01.00 pagi | -6 jam | Hari yang sama
# ------------------------------------------------------------------------------
# Washington, D.C. USA jam 7 malam | -12 jam | Hari sebelumnya
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

#

#
# ------------------------------------------------------------------------------
# Islamabad, Pakistan jam 5 pagi  |  -2 jam Hari yang sama
# Abu Dhabi/Dubai, Uni Emirat Arab jam 4 pagi | -3 jam Hari yang sama
# Ankara, Turki jam 3 pagi | -4 jam Hari yang sama
# Kairo, Mesir jam 2 pagi | -5 jam Hari yang sama
# Brasil (Rio de Janeiro) jam 9 malam | -10 jam Hari sebelumnya

