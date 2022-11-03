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

# cari daerah di negara lain yang selisihnya 6 jam dgn Indonesia
# https://www.kompas.com/skola/read/2021/03/04/120155469/tabel-perbedaan-waktu-di-indonesia-dengan-negara-lainnya
#
# kandidat, misal di Indonesia jam 7 pagi:
# amsterdam Belanda jam 1 siang | -6 jam | Hari yang sama
# Paris Perancis jam 1 siang | -6 jam | Hari yang sama
# Roma Italia jam 1 siang | -6 jam | Hari yang sama
# Washington, D.C. USA jam 7 malam | -12 jam | Hari sebelumnya
#
list_kota = ['malang','lamongan','jakarta']

n_byk_ambil_data = 10
for iter_get in range(n_byk_ambil_data):
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
    time.sleep(3)


# CREATE TABLE data_suhu_dll (
#     date                                 DATETIME,
#     kota                                 TEXT,
#     suhu_dlm_celcius                     TEXT,
#     precipitation_curah_hujan_dlm_persen TEXT,
#     humidity_kelembaban_dlm_persen       TEXT,
#     wind_angin_dlm_km_per_jam            TEXT
# );



# page 6 -> 59

conn.commit()
db.close()
conn.close()

