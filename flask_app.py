# Koding Contoh MK Jaringan Saraf Tiruan (JST) Semester Ganjil 2022/2023 Filkom UB
# Rencana Pembelajaran MK JST Semester Ganjil 2022/2023 Kelas EF
# Fakultas Ilmu Komputer (Filkom), Universitas Brawijaya (UB) 2022.

# Dosen Pengampu:
# 1. Imam Cholissodin, S.Si., M.Kom. | email: imamcs@ub.ac.id | Filkom UB
# 2. Muhammad Tanzil Furqon, S.Kom., M.Cs. | email: m.tanzil.furqon@ub.ac.id | Filkom UB

from flask import Flask,render_template, Response, redirect,url_for,session,request,jsonify
from flask import render_template_string
import sqlite3
from flask_cors import CORS

from flask import send_file
from flask_qrcode import QRcode

from io import BytesIO
import os

import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
# from bokeh.util.string import encode_utf8

app = Flask(__name__, static_folder='static')
qrcode = QRcode(app)

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

project_dir = os.path.dirname(os.path.abspath(__file__))
database_file = "sqlite:///{}".format(os.path.join(project_dir, "static/qr_app/db/qrdata.db"))

app.config["SQLALCHEMY_DATABASE_URI"] = database_file
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True

# DB untuk qr_app
db_qr = SQLAlchemy(app)
migrate = Migrate(app, db_qr)

# Operasi untuk migrate
# flask db_qr init
# flask db_qr migrate
# flask db_qr upgrade

from static.qr_app.model.StudentModel import Student
from static.qr_app.model.AttendanceModel import Attendance
from static.qr_app.module.Camera import Scanner
# import pyqrcode
import uuid

CORS(app, resources=r'/api/*')

app.secret_key = 'filkomub1919^&&*(&^(filkom#BJH#G#VB#JST99nDataPyICS_ap938255bnUB'

# keterangan:
# "#" adalah untuk comment
# <br> adalah new line
# &nbsp; adalah spasi
# <!-- --> atau <!--- ---> adalah untuk comment

# FrameWeb_atas & FrameWeb_bawah untuk dekorasi web
# agar menjadi Web yang Responsif

FrameWeb_atas = """
{% extends "extends/base.html" %}
{% block title %}
    <title>Web App JST Dgn Python</title>
{% endblock title %}
{{ self.title() }}
    Home
{{ self.title() }}
<button onclick="window.location.href='/'" class="btn btn-outline btn-rounded btn-info">
    <i class="ti-arrow-left m-l-5"></i>
    <span>Back Home</span>
</button> Project 1

{{ self.title() }}
    Project 1

{% block content %}
"""
A_a = FrameWeb_atas

FrameWeb_bawah = """
{% endblock content %}
"""
Z_z = FrameWeb_bawah

# @app.route('/')
# def hello_jst():
#    return 'Hello Students | Koding Jaringan Saraf Tiruan (JST) pada Teknologi Cloud :D'

@app.route('/som', methods=['GET'])
def som():
    # ref:
    # [0] https://towardsdatascience.com/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
    # [1] ..

    import os.path

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 1. import lib.
    import numpy as np
    from numpy.ma.core import ceil
    from scipy.spatial import distance #distance calculation
    from sklearn.preprocessing import MinMaxScaler #normalisation
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score #scoring
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from matplotlib import animation, colors

    # 2. Import dataset
    # banknote authentication Data Set
    # https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    # Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
    # Irvine, CA: University of California, School of Information and Computer Science.

    # cara download, ketikkan pada bash
    # $ wget https://raw.githubusercontent.com/matbesancon/BankNotes/master/data_banknote_authentication.txt
    # $ mv data_banknote_authentication.txt mysite/static/data_contoh

    # data_file = "data_banknote_authentication.txt"
    data_file = os.path.join(BASE_DIR, "static/data_contoh/data_banknote_authentication.txt")
    data_x = np.loadtxt(data_file, delimiter=",", skiprows=0, usecols=range(0,4) ,dtype=np.float64)
    data_y = np.loadtxt(data_file, delimiter=",", skiprows=0, usecols=(4,),dtype=np.int64)

    # 3. Training and testing data split
    # The data is split for training and testing at 0.8:0.2.
    # atau dengan jumlah 1097 : 275
    # train and test split
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape) # check the shapes

    # 4. Helper functions
    # Data Normalisation
    def minmax_scaler(data):
      scaler = MinMaxScaler()
      scaled = scaler.fit_transform(data)
      return scaled

    # Euclidean distance
    def e_distance(x,y):
      return distance.euclidean(x,y)

    # Manhattan distance
    def m_distance(x,y):
      return distance.cityblock(x,y)

    # Best Matching Unit search
    def winning_neuron(data, t, som, num_rows, num_cols):
      winner = [0,0]
      shortest_distance = np.sqrt(data.shape[1]) # initialise with max distance
      input_data = data[t]
      for row in range(num_rows):
        for col in range(num_cols):
          distance = e_distance(som[row][col], data[t])
          if distance < shortest_distance:
            shortest_distance = distance
            winner = [row,col]
      return winner

    # Learning rate and neighbourhood range calculation
    def decay(step, max_steps,max_learning_rate,max_m_dsitance):
      coefficient = 1.0 - (np.float64(step)/max_steps)
      learning_rate = coefficient*max_learning_rate
      neighbourhood_range = ceil(coefficient * max_m_dsitance)
      return learning_rate, neighbourhood_range


    # 5. Hyperparameters
    # grid (10*10) dari num_rows x num_cols
    num_rows = 10
    num_cols = 10
    max_m_dsitance = 4
    max_learning_rate = 0.5
    # max_steps = int(7.5*10e3)
    max_steps = 100

    # num_nurons = 5*np.sqrt(train_x.shape[0])
    # grid_size = ceil(np.sqrt(num_nurons))
    # print(grid_size)

    # 6. Training
    #mian function
    train_x_norm = minmax_scaler(train_x) # normalisation

    # initialising self-organising map
    num_dims = train_x_norm.shape[1] # numnber of dimensions in the input data
    np.random.seed(40)
    som = np.random.random_sample(size=(num_rows, num_cols, num_dims)) # map construction

    # start training iterations
    for step in range(max_steps):
      if (step+1) % 1000 == 0:
        print("Iteration: ", step+1) # print out the current iteration for every 1k
      learning_rate, neighbourhood_range = decay(step, max_steps,max_learning_rate,max_m_dsitance)

      t = np.random.randint(0,high=train_x_norm.shape[0]) # random index of traing data
      winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
      for row in range(num_rows):
        for col in range(num_cols):
          if m_distance([row,col],winner) <= neighbourhood_range:
            som[row][col] += learning_rate*(train_x_norm[t]-som[row][col]) #update neighbour's weight

    print("SOM training completed")

    # 7. Show labels to the trained SOM
    # collecting labels
    label_data = train_y
    map = np.empty(shape=(num_rows, num_cols), dtype=object)

    for row in range(num_rows):
      for col in range(num_cols):
        map[row][col] = [] # empty list to store the label

    for t in range(train_x_norm.shape[0]):
      if (t+1) % 1000 == 0:
        print("sample data: ", t+1)
      winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
      map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron


    # construct label map
    label_map = np.zeros(shape=(num_rows, num_cols),dtype=np.int64)
    for row in range(num_rows):
      for col in range(num_cols):
        label_list = map[row][col]
        if len(label_list)==0:
          label = 2
        else:
          label = max(label_list, key=label_list.count)
        label_map[row][col] = label

    title = ('Iteration ' + str(max_steps))
    cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange'])
    plt.imshow(label_map, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.show()

    # 8. Predicting the test set labels
    # test data
    # using the trained som, search the winning node of corresponding to the test data
    # get the label of the winning node

    data = minmax_scaler(test_x) # normalisation

    winner_labels = []

    for t in range(data.shape[0]):
     winner = winning_neuron(data, t, som, num_rows, num_cols)
     row = winner[0]
     col = winner[1]
     predicted = label_map[row][col]
     winner_labels.append(predicted)

    # print("Accuracy: ",accuracy_score(test_y, np.array(winner_labels)))



    template_view = '''
            <!--- <html> --->
            <!--- <head> --->
            <!--- </head> --->
            <!--- <body> --->
            <h2>
                <p style="text-decoration: underline;">
                  Implementasi Algoritma SOM:
                </p>
            </h2>
                  <form method="post">
                    Data Y test Aktual: <br>
                    {{ test_y_aktual }}
                    <br>
                    Data Y hasil Predik: <br>
                    {{ test_y_predik }}
                    <br><br>

                  </form>
                  <h2>Hasil Nilai Evaluasi:  </h2>
                    {{ nilai_eval }}
                    <br>

                  <h2>Plot Hasil SOM:  </h2>
                  <img src={{url_image}} alt="Chart" height="480" width="640">

            <!--- </body> --->
            <!--- </html> --->
        '''

    # Cara ke-2
    # simpan dalam path + nama file /static/img/new_timer.png
    url_simpan = "static/img/new_gambar.png"

    fig = plt.figure()
    # plt.plot(n, [x*math.log(x,2) for x in n], 'r.-', label='TimSort by T(n)') # NlogN
    # plt.plot(n,list(array_waktu_dalam_detik[:,1][:len(n)]), 'g.-', label='TimSort by Timer') # TimSort

    title = ('Iteration ' + str(max_steps))
    cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange'])
    plt.imshow(label_map, cmap=cmap)
    plt.colorbar()
    plt.title(title)

    # plt.xlabel('Banyak Data (N)')
    # plt.ylabel('Waktu Komputasi')
    # plt.legend(loc="upper left")
    plt.show()

    url_file_image_simpan = os.path.join(BASE_DIR, url_simpan)
    plt.savefig(url_file_image_simpan)

    # Cara ke-3
    # /bokeh

    nilai_eval = accuracy_score(test_y, np.array(winner_labels))


    # return hasil
    # return render_template_string(A_a+template_view+Z_z, y_aktual_n_y_predik = zip(test_y, np.array(winner_labels)), nilai_eval = nilai_eval, url_image = url_simpan)
    return render_template_string(A_a+template_view+Z_z, test_y_aktual = test_y, test_y_predik = np.array(winner_labels), nilai_eval = nilai_eval, url_image = url_simpan)

@app.route('/bokeh')
def bokeh():
    # init a basic bar chart:
    # http://bokeh.pydata.org/en/latest/docs/user_guide/plotting.html#bars
    fig = figure(plot_width=300, plot_height=300)
    fig.vbar(
        x=[1, 2, 3, 4],
        width=0.5,
        bottom=0,
        top=[1.7, 2.2, 4.6, 3.9],
        color='navy'
    )

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    template_view ='''
    <!-- <!doctype html> -->
    <!-- <html lang="en"> -->
    <!--  <head> -->
        <meta charset="utf-8">
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <title>Embed Demo</title>
        {{ js_resources|indent(4)|safe }}
        {{ css_resources|indent(4)|safe }}
        {{ plot_script|indent(4)|safe }}
    <!--  </head> -->
    <!--  <body> -->
        {{ plot_div|indent(4)|safe }}
    <!--  </body> -->
    <!-- </html> -->
    '''

    # render template
    script, div = components(fig)
    html = render_template_string(
        A_a+template_view+Z_z,
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )

    return html

@app.route('/db/<aksi>')
def manipulate_tabel(aksi):
    conn = connect_db()
    db = conn.cursor()

    # Aksi => Buat, Hapus

    if aksi == 'c':
        str_info = 'tabel berhasil dibuat :D'
        # create tabel
        db.execute("""
        CREATE TABLE IF NOT EXISTS data_cronjob
        (tipe_run TEXT, date_pembuatan DATETIME,
        teks_call_sintaks TEXT,
        keterangan TEXT,
        date_masa_berlaku DATETIME)
        """)
    elif aksi== 'd':
        str_info = 'tabel berhasil dihapus :D'
        # hapus tabel
        db.execute("""
        DROP TABLE IF EXISTS data_cronjob
        """)

    conn.commit()
    db.close()
    conn.close()

    return str_info

@app.route('/db/CloudAI_Air/<aksi>')
def manipulate_tabel_CloundAI_Air(aksi):
    conn = connect_db()
    db = conn.cursor()

    if aksi == 'c':
        # create tabel
        db.execute("""
        CREATE TABLE IF NOT EXISTS CloudAI_Air (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suhu_dlm_celcius TEXT,
                    humidity_kelembaban_dlm_persen TEXT,
                    precipitation_curah_hujan_dlm_persen TEXT,
                    wind_angin_dlm_km_per_jam TEXT,
                    durasi_air_dlm_menit TEXT
                )
        """)
        str_info = 'tabel berhasil dibuat :D'
    elif aksi== 'd':
        # hapus tabel
        db.execute("""
        DROP TABLE IF EXISTS CloudAI_Air
        """)

        str_info = 'tabel berhasil dihapus :D'

    conn.commit()
    db.close()
    conn.close()

    return str_info

@app.route('/db/CloudAI_Air_Rev/<aksi>')
def manipulate_tabel_CloundAI_Air_Rev(aksi):
    conn = connect_db()
    db = conn.cursor()

    if aksi == 'c':
        # create tabel
        db.execute("""
        CREATE TABLE IF NOT EXISTS CloudAI_Air_Rev (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suhu_dlm_celcius TEXT,
                    humidity_kelembaban_dlm_persen TEXT,
                    precipitation_curah_hujan_dlm_persen TEXT,
                    wind_angin_dlm_km_per_jam TEXT,
                    durasi_air_dlm_menit TEXT
                )
        """)
        str_info = 'tabel berhasil dibuat :D'
    elif aksi== 'd':
        # hapus tabel
        db.execute("""
        DROP TABLE IF EXISTS CloudAI_Air_Rev
        """)

        str_info = 'tabel berhasil dihapus :D'

    conn.commit()
    db.close()
    conn.close()

    return str_info

@app.route('/user')
def data_user():
    try:
        conn = connect_db()
        db = conn.cursor()

        rs = db.execute("SELECT * FROM user order by id")
        userslist = rs.fetchall()
        return render_template('data_user.html',userslist=userslist)

    except Exception as e:
        print(e)
    finally:
        db.close()
        conn.close()

@app.route("/update_user",methods=["POST","GET"])
def update_user():
    try:
        conn = connect_db()
        db = conn.cursor()
        if request.method == 'POST':
            field = request.form['field']
            value = request.form['value']
            editid = request.form['id']

            if field == 'mail':
                db.execute("""UPDATE user SET Mail=? WHERE id=?""",(value,editid))
            if field == 'name':
                db.execute("""UPDATE user SET Name=? WHERE id=?""",(value,editid))
            if field == 'pwd':
                db.execute("""UPDATE user SET Password=? WHERE id=?""",(value,editid))
            if field == 'level':
                db.execute("""UPDATE user SET Level=? WHERE id=?""",(value,editid))

            conn.commit()
            success = 1
        return jsonify(success)
    except Exception as e:
        print(e)
    finally:
        db.close()
        conn.close()

# ================ awal - dasar ke-2 ===============
#

# buat input dari url, untuk penjumlahan misal 2 bilangan
@app.route('/add/<a>/<b>')
def add_ab(a,b):
    c = int(a) + float(b)
    return 'a + b = ' + str(c)
    # return 'a + b = %s' % c
# https://userAnda.pythonanywhere.com/add/1/2.5
# hasil => a + b = 3.5

#
# buatlah halaman post sekaligus get
# nilai a dan b, lalu ditambahkan
# dengan return kode html dalam flask python Web App
@app.route('/post_add2', methods=["POST", "GET"])
def inputkan_ab():
    # membuat penjumlahan 2 bilangan

    if request.method == 'POST': # dioperasikan dihalaman sendiri tanpa send ke route, misal /post_add2

        a_in = float(request.form['a'])
        b_in = float(request.form['b'])
        c = a_in + b_in

        return '''
        <html>
            <head>
            </head>
            <body>
              <form method="post">
                <input type="text" name="a" value="%s" />
                <input type="text" name="b" value="%s" />
                <input type="submit" value="Hitung a + b"/>

              </form>
              <h2>Hasil a + b = %s + %s = %s </h2>
            </body>
        </html>
        ''' % (a_in, b_in, a_in, b_in, c)

    else: # untuk yang 'GET' data awal untuk di send ke /post_add2
        return '''
            <html>
                <head>
                </head>
                <body>
                  <form action="/post_add2" method="post">
                    Masukkan nilai a = <input type="text" name="a" value="" />
                    <br>
                    Masukkan nilai b = <input type="text" name="b" value="" />
                    <input type="submit" value="Hitung a + b"/>
                  </form>
                </body>
            </html>
        '''

#
# buatlah halaman post sekaligus get
# nilai a dan b, lalu ditambahkan
# dengan return file "form_add3.html" dalam folder "mysite/templates", flask python Web App
@app.route('/post_add3', methods=["POST", "GET"])
def inputkan_ab3():
    # membuat penjumlahan 2 bilangan
    if request.method == 'POST': # dioperasikan dihalaman sendiri tanpa send ke route, misal /post_add2

        a_in = float(request.form['a'])
        b_in = float(request.form['b'])
        c = a_in + b_in

        return render_template('form_add3.html', a_save = a_in, b_save = b_in, c_save = c)

    else: # untuk yang 'GET' data awal untuk di send ke /post_add3
        return render_template('form_add3.html')


# ================================================================================
# Contoh koding dasar operasi CRUD pada tabel CloudAI_Air,
# mulai dari "def dasar2_create_database():" sampai sebelum "# ================ akhir - dasar ke-2 ==============="
#
# membuat render_template_string sebagai pengganti render_template
# agar semua kodenya hanya dalam 1 file, sehingga lebih mudah untuk membuat dan run kodingnya
#
def dasar2_create_database():
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
                CREATE TABLE IF NOT EXISTS CloudAI_Air (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suhu_dlm_celcius TEXT,
                    humidity_kelembaban_dlm_persen TEXT,
                    precipitation_curah_hujan_dlm_persen TEXT,
                    wind_angin_dlm_km_per_jam TEXT,
                    durasi_air_dlm_menit TEXT
                )
                """)

    conn.commit()
    conn.close()

def dasar2_generate_data():
    """Generate sintesis atau dummy data untuk percontohan."""
    conn = connect_db()
    cur = conn.cursor()

    cur.execute('SELECT * FROM CloudAI_Air')
    entry = cur.fetchone()

    if entry is None:
        import numpy as np
        import pandas as pd
        import os.path

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))


        # Misal skema dataset-nya seperti berikut: => Silahkan dimodifikasi sesuai case Anda
        kolomFitur_X_plus_Target_Y = ['Suhu (X1)','Kelembaban (X2)', 'Curah Hujan (X3)','Angin (X4)','Durasi Air Dlm Menit (Y)']

        # set bykData = 3*np.power(10,7)
        bykData = 10
        bykFitur = len(kolomFitur_X_plus_Target_Y)-1

        # Interval atau Variasi nilai fitur
        nilaiFitur_Suhu = [17,35]
        nilaiFitur_Kelembaban = [70,90]
        nilaiFitur_Curah_Hujan = [2,95]
        nilaiFitur_Angin = [0,15]
        labelTargetY = [0.0,90.0]

        # generate isi dataset
        content_dataGenerate = np.array([np.arange(bykData)]*(bykFitur+1)).T
        df_gen = pd.DataFrame(content_dataGenerate, columns=kolomFitur_X_plus_Target_Y)

        df_gen ['Suhu (X1)'] = np.random.randint(nilaiFitur_Suhu[0], nilaiFitur_Suhu[1], df_gen.shape[0])
        df_gen ['Kelembaban (X2)'] = np.random.randint(nilaiFitur_Kelembaban[0], nilaiFitur_Kelembaban[1], df_gen.shape[0])
        df_gen ['Curah Hujan (X3)'] = np.random.randint(nilaiFitur_Curah_Hujan[0], nilaiFitur_Curah_Hujan[1], df_gen.shape[0])
        df_gen ['Angin (X4)'] = np.random.randint(nilaiFitur_Angin[0], nilaiFitur_Angin[1], df_gen.shape[0])
        df_gen ['Durasi Air Dlm Menit (Y)'] = np.round(np.random.uniform(labelTargetY[0], labelTargetY[1], df_gen.shape[0]),2)

        # save dataframe generate ke *.csv
        import os
        userhome = os.path.expanduser("~").split("/")[-1]

        path = "/home/"+userhome+"/mysite/static/data_contoh"
        if not os.path.exists(path):
            os.makedirs(path)
        # file_name_data_generate = 'static/data_contoh/Data_CloudAI_Air.csv'
        # df_gen.to_csv(file_name_data_generate, encoding='utf-8', index=False)
        url_file_name_data_generate = os.path.join(BASE_DIR, "static/data_contoh/Data_CloudAI_Air.csv")
        df_gen.to_csv(url_file_name_data_generate, encoding='utf-8', index=False)

        # read file *.csv dan tampilkan
        # data_generate = pd.read_csv(file_name_data_generate)

        url = os.path.join(BASE_DIR, "static/data_contoh/Data_CloudAI_Air.csv")

        # Importing the dataset => ganti sesuai dengan case yg anda usulkan
        dataset = pd.read_csv(url)
        # X = dataset.iloc[:, :-1].values
        # y = dataset.iloc[:, 1].values

        def pushCSVdatasetToDB(x1,x2,x3,x4,y):
            #inserting values inside the created table

            cmd = "INSERT INTO CloudAI_Air(suhu_dlm_celcius, humidity_kelembaban_dlm_persen, precipitation_curah_hujan_dlm_persen, wind_angin_dlm_km_per_jam, durasi_air_dlm_menit) VALUES('{}','{}','{}','{}','{}')".format(x1,x2,x3,x4,y)
            cur.execute(cmd)
            conn.commit()

        # CSV_to_SQLite3 dari file dataset
        for i in range(0,len(dataset)):
            pushCSVdatasetToDB(dataset.iloc[i][0],dataset.iloc[i][1],dataset.iloc[i][2],dataset.iloc[i][3],dataset.iloc[i][4])
    else:
        ket_hasil = 'Tidak dilakukan Insert, karena Tabel tidak kosong'
        print(ket_hasil)

    conn.commit()
    cur.close()
    conn.close()

@app.route('/dasar2_crud')
def dasar2_index():
    return '<a href="/dasar2_list">Demo Menampilkan List dari Tabel + Support => Create, Read, Update, Delete (CRUD)</a>'

@app.route('/dasar2_list')
def dasar2_list():

    # buat tabel dan generate data dummy
    dasar2_create_database()
    dasar2_generate_data()

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM CloudAI_Air")
    rows = cur.fetchall()

    conn.close()

    #return render_template("list.html", rows=rows)
    return render_template_string(template_list, rows=rows)


@app.route('/dasar2_edit/<int:number>', methods=['GET', 'POST'])
def dasar2_edit(number):
    conn = connect_db()
    cur = conn.cursor()

    if request.method == 'POST':
        item_id      = number
        item_suhu    = request.form['suhu']
        item_kelembaban = request.form['kelembaban']
        item_hujan  = request.form['hujan']
        item_angin = request.form['angin']
        item_durasi = request.form['durasi']

        # suhu_dlm_celcius, humidity_kelembaban_dlm_persen, precipitation_curah_hujan_dlm_persen, wind_angin_dlm_km_per_jam, durasi_air_dlm_menit

        cur.execute("UPDATE CloudAI_Air SET suhu_dlm_celcius = ?, humidity_kelembaban_dlm_persen = ?, precipitation_curah_hujan_dlm_persen = ?, wind_angin_dlm_km_per_jam = ?, durasi_air_dlm_menit = ? WHERE id = ?",
                    (item_suhu, item_kelembaban, item_hujan, item_angin, item_durasi, item_id))
        conn.commit()

        return redirect('/dasar2_list')

    cur.execute("SELECT * FROM CloudAI_Air WHERE id = ?", (number,))
    item = cur.fetchone()

    conn.close()

    #return render_template("edit.html", item=item)
    return render_template_string(template_edit, item=item)

@app.route('/dasar2_delete/<int:number>')
def dasar2_delete(number):
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("DELETE FROM CloudAI_Air WHERE id = ?", (number,))

    conn.commit()
    conn.close()

    return redirect('/dasar2_list')

@app.route('/dasar2_add', methods=['GET', 'POST'])
def dasar2_add():
    conn = connect_db()
    cur = conn.cursor()

    if request.method == 'POST':
        # item_id      = number
        item_suhu    = request.form['suhu']
        item_kelembaban = request.form['kelembaban']
        item_hujan  = request.form['hujan']
        item_angin = request.form['angin']
        item_durasi = request.form['durasi']

        cur.execute("""INSERT INTO CloudAI_Air (suhu_dlm_celcius, humidity_kelembaban_dlm_persen, precipitation_curah_hujan_dlm_persen, wind_angin_dlm_km_per_jam, durasi_air_dlm_menit) VALUES (?, ?, ?, ?, ?)""",
                    (item_suhu, item_kelembaban, item_hujan, item_angin, item_durasi))
        conn.commit()

        return redirect('/dasar2_list')

    #return render_template("add.html", item=item)
    return render_template_string(template_add)

@app.route('/dasar2_add2')
def dasar2_add2():
    conn = connect_db()
    cur = conn.cursor()

    # get data dari iot API
    import requests
    # from datetime import datetime
    # import pytz
    # Date = str(datetime.today().astimezone(pytz.timezone('Asia/Jakarta')).strftime('%d-%m-%Y %H:%M:%S'))

    def F2C(f_in):
        return (f_in - 32)* 5/9

    def Kelvin2C(k_in):
      return (k_in-273.15)

    # list_kota = ['Jakarta','Los Angeles','Chicago','New York City','Toronto','São Paulo', \
    #              'Lagos', 'London', 'Johannesburg', 'Kairo', 'Paris', 'Zurich', 'Istanbul', 'Moskwa', 'Dubai', \
    #             'Mumbai','Hong Kong','Shanghai','Singapura','Tokyo','Sydney']
    list_kota = ['Malang']


    for nama_kota in list_kota:
        #   each_list_link='http://api.weatherapi.com/v1/current.json?key=re2181c95fd6d746e9a1331323220104&q='+nama_kota
        each_list_link='http://api.weatherapi.com/v1/current.json?key=2181c95fd6d746e9a1331323220104&q='+nama_kota
        resp=requests.get(each_list_link)

        # print(nama_kota)

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

        # print(nama_kota, 'dengan suhu = ', round(float(suhu),2),'°C', end='\n')

        cur.execute("""INSERT INTO CloudAI_Air (suhu_dlm_celcius, humidity_kelembaban_dlm_persen, precipitation_curah_hujan_dlm_persen, wind_angin_dlm_km_per_jam) VALUES (?, ?, ?, ?)""",
                (suhu, lembab, curah_hujan, angin))

        conn.commit()
        cur.close()
        conn.close()

    return redirect('/dasar2_list')

template_list = """
<h2>Menampilkan Data CloudAI Air + Support Create, Read, Update, delete (CRUD)</h2>
<a href="{{ url_for( "dasar2_add" ) }}">Tambah Data</a> |
<a href="{{ url_for( "dasar2_add2" ) }}">Tambah Data dari iot_api (tanpa nilai Durasi Waktu)</a>
{% if rows %}
<table border="1">
    <thead>
        <td>No</td>
        <td>Suhu (°C)</td>
        <td>Kelembaban (%)</td>
        <td>Curah Hujan (%)</td>
        <td>Kecepatan Angin (Km/Jam)</td>
        <td>Durasi Waktu Pengairan / Penyiraman (Menit)</td>
    </thead>

    {% for row in rows %}
    <tr>
        <td>{{ loop.index }}</td>
        <td>{{row[1]}}</td>
        <td>{{row[2]}}</td>
        <td>{{row[3]}}</td>
        <td>{{row[4]}}</td>
        <td>{{row[5]}}</td>
        <td>
            <a href="{{ url_for( "dasar2_edit", number=row[0] ) }}">Edit</a> |
            <a href="{{ url_for( "dasar2_delete", number=row[0] ) }}">Hapus</a>
        </td>
    </tr>
    {% endfor %}
</table>
{% else %}
Empty</br>
{% endif %}
"""

template_add = """
<h1>Tambah Data CloudAI Air</h1>
<form method="POST" action="{{ url_for( "dasar2_add" ) }}">
    Suhu: <input name="suhu" value=""/></br>
    Kelembaban: <input name="kelembaban" value=""/></br>
    Curah Hujan: <input name="hujan" value=""/></br>
    Kecepatan Angin: <input name="angin" value=""/></br>
    Durasi Waktu Pengairan / Penyiraman: <input name="durasi" value=""/></br>
    <button>Simpan Data</button></br>
</form>
"""

template_edit = """
<h1>Edit Data CloudAI Air</h1>
<form method="POST" action="{{ url_for( "dasar2_edit", number=item[0] ) }}">
    Suhu: <input name="suhu" value="{{item[1]}}"/></br>
    Kelembaban: <input name="kelembaban" value="{{item[2]}}"/></br>
    Curah Hujan: <input name="hujan" value="{{item[3]}}"/></br>
    Kecepatan Angin: <input name="angin" value="{{item[4]}}"/></br>
    Durasi Waktu Pengairan / Penyiraman: <input name="durasi" value="{{item[5]}}"/></br>
    <button>Simpan Update Data</button></br>
</form>
"""

# ================ akhir - dasar ke-2 ===============

# ================ awal - dasar ke-1 ===============
# #

# @app.route('/add')
# def add():
#     # membuat penjumlahan 2 bilangan
#     a = 10
#     b = 90
#     c = a + b

#     return str(c)

# # buatlah halaman perkalian
# # antara a*b
# @app.route('/kali')
# def kali():
#     # membuat perkalian 2 bilangan
#     a = 10
#     b = 90
#     c = a * b

#     return str(c)

# # buatlah tampilan indeks looping 1..10
# @app.route('/loop')
# def loop():
#     c = ''
#     for i in range(10): # i = 0,1,..,9
#         c +=str(i+1) + '  '

#     return str(c)

# # buatlah tampilan indeks looping 1..10 dengan new line (<br> dari tag html)
# @app.route('/loop_new_line')
# def loop_new_line():
#     c = ''
#     for i in range(10): # i = 0,1,..,9
#         c +=str(i+1) + '<br>'

#     return str(c)

# # buatlah tampilan indeks looping 1 sampai 10
# # yang ganjil
# @app.route('/ganjil')
# def ganjil():
#     c = ''
#     for i in range(10): # i = 0,1,..,9
#         if((i+1)%2!=0):
#             c +=str(i+1) + '  '

#     return str(c)
# # ================ akhir - dasar ke-1 ===============

# ========= untuk Tugas Ke-1 & 2 | Project =================

@app.route("/")
def index():
    # return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/login",methods=["GET", "POST"])
def login():
  conn = connect_db()
  db = conn.cursor()
  msg = ""
  if request.method == "POST":
      mail = request.form["mail"]
      passw = request.form["passw"]

      rs = db.execute("SELECT * FROM user WHERE Mail=\'"+ mail +"\'"+" AND Password=\'"+ passw+"\'" + " LIMIT 1")

      conn.commit()

      hasil = []
      for v_login in rs:
          hasil.append(v_login)

      if hasil:
          session['name'] = v_login[3]
          return redirect(url_for("launchpad_menu"))
      else:
          msg = "Masukkan Username (Email) dan Password dgn Benar!"

  return render_template("login.html", msg = msg)

@app.route("/register", methods=["GET", "POST"])
def register():
  conn = connect_db()
  # db = conn.cursor()
  if request.method == "POST":
      mail = request.form['mail']
      uname = request.form['uname']
      passw = request.form['passw']

      cmd = "insert into user(Mail, Password,Name,Level) values('{}','{}','{}','{}')".format(mail,passw,uname,'1')
      conn.execute(cmd)
      conn.commit()

      # conn = db

      return redirect(url_for("login"))
  return render_template("register.html")

def connect_db():
    import os.path

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "data.db")

    return sqlite3.connect(db_path)

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html")

@app.errorhandler(500)
def internal_server_error(error):
    return render_template("500.html")

@app.route('/iot', methods=["GET", "POST"])
def iot():

    if 'name' in session:
        name = session['name']
    else:
        name = 'Guest'

    # start kode untuk download atau export semua data dari tabel data_suhu_dll menjadi file *.csv
    if request.method == "POST":

        from io import StringIO
        import csv

        # date_var = request.args.get('date_var')
        # kota_var = request.args.get('kota_var')
        conn = connect_db()
        db = conn.cursor()

        output = StringIO()
        writer = csv.writer(output)
        c = db.execute("SELECT * FROM data_suhu_dll")

        result = c.fetchall()
        writer.writerow([i[0] for i in c.description])

        for row in result:
            line = [str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5])]
            writer.writerow(line)

        output.seek(0)

        conn.commit()
        db.close()
        conn.close()

        return Response(output, mimetype="text/csv",
                        headers={"Content-Disposition": "attachment;filename=data_suhu_iot_all.csv"})
    # ending kode untuk download atau export semua data dari tabel data_suhu_dll menjadi file *.csv


    # menampilkan data dari tabel data_suhu_dll
    conn = connect_db()
    db = conn.cursor()

    c = db.execute(""" SELECT * FROM  data_suhu_dll """)

    mydata = c.fetchall()
    for x in c.fetchall():
        name_v=x[0]
        data_v=x[1]
        break

    hasil = []
    for v_login in c:
        hasil.append(v_login)

    conn.commit()
    db.close()
    conn.close()


    return render_template("getsuhu_dll.html", header = mydata)

@app.route('/del_iot/', methods=["GET"])
def del_iot():
    date_var = request.args.get('date_var')
    kota_var = request.args.get('kota_var')
    conn = connect_db()
    db = conn.cursor()

    db.execute("DELETE FROM data_suhu_dll WHERE date =\'"+ date_var +"\' AND  kota =\'"+ kota_var +"\'")

    conn.commit()
    db.close()
    conn.close()

    return redirect(url_for("iot"))

@app.route('/dw_iot/', methods=["GET"])
def dw_iot():

    from io import StringIO
    import csv

    date_var = request.args.get('date_var')
    # kota_var = request.args.get('kota_var')
    conn = connect_db()
    db = conn.cursor()

    output = StringIO()
    writer = csv.writer(output)
    c = db.execute("SELECT * FROM data_suhu_dll WHERE date =\'"+ date_var +"\'")

    result = c.fetchall()
    writer.writerow([i[0] for i in c.description])

    for row in result:
        line = [str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5])]
        writer.writerow(line)

    output.seek(0)

    conn.commit()
    db.close()
    conn.close()

    return Response(output, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=data_suhu_iot.csv"})

@app.route('/logout')
def logout():
   # remove the name from the session if it is there
   session.pop('name', None)
   return redirect(url_for('index'))


# ================
# Ergo Project

@app.route("/in")
def index_qrcode():
    return render_template("qrcode.html")


@app.route("/qrcode", methods=["GET"])
def get_qrcode():
    # please get /qrcode?data=<qrcode_data>
    data = request.args.get("data", "")
    return send_file(qrcode(data, mode="raw"), mimetype="image/png")

@app.route('/qr_index')
def qr_index():
    attendance = Attendance.getAll()
    return render_template("qr_scan2.html", data=enumerate(attendance, 1))


@app.route("/qr_scan", methods=["GET"])
def qr_scan():
    return Response(scanner(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/qr_student", methods=["GET", "POST"])
def qr_student():
    if request.method == "POST":
        name = request.form['name']
        nim = request.form['nim']
        UUID = str(uuid.uuid4())
        qr_code_mark = "static/img/tmp_qr/{}.png".format(UUID)
        student = Student(nim=nim, name=name, qr_code=qr_code_mark)
        student.save()

        import qrcode

        # # /qrcode
        # qrcode_img = qrcode.make(student.id)
        # # buf = io.BytesIO()
        # buf_qrcode = BytesIO()
        # qrcode_img.save(buf_qrcode)
        # buf_qrcode.seek(0)
        # # return send_file(buf_qrcode, mimetype='image/jpeg')

        qrcode_img = qrcode.make(student.id)
        # qrcode_img = qrcode(student.id)
        # canvas = Image.new('RGB', (290,290), 'white')
        # draw = ImageDraw.Draw(canvas)
        # canvas.paste(qrcode_img)
        # fname = f'qr_code_{self.name}.png'
        fname = f'static/img/tmp_qr/qr_code_{student.id}.png'.format(UUID)
        buffer = BytesIO()
        # canvas.save(buffer,'PNG')
        # qrcode_img.save(fname, File(buffer), save=False)
        # qrcode_img.save(fname, buffer, save=False)
        # qrcode_img.save(buffer)

        import os.path

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        url_file_name_qrcode = os.path.join(BASE_DIR, fname)

        qrcode_img.save(url_file_name_qrcode, format="PNG")
        # canvas.close()
        # super().save(*args, **kwargs)

        # img = pyqrcode.create(student.id, error="L", mode="binary", version=5)
        # img.png(qr_code, scale=10)
    students = Student.getAll()
    return render_template("qr_student.html", data=enumerate(students, 1))


def scanner():
    camera = Scanner()
    while True:
        frame = camera.get_video_frame()

        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break


@app.route("/my_async2", methods=['GET','POST'])
def my_async2():
    # ref.:
    # [0] https://github.com/adrianton3/pso.js/tree/master/examples/async

    template_view = '''
            <!--- <!DOCTYPE html> --->
            <!--- <html> --->
            <!--- <head lang="en"> --->
            <!---	<meta charset="UTF-8"> --->
            <!---	<link rel="stylesheet" href="async.css"> --->
            <!---	<title>Async - pso.js demo</title> --->
            <!---</head> --->


            	<script type="text/javascript" src="{{ url_for('static', filename = 'js/mylib_async/pso.js') }}"></script>
            	<script>
            	function init() {
                	   'use strict';

                	//function async(){

                    	var outTextarea = document.getElementById('out');

                    	function log(text) {
                    		outTextarea.value += text;
                    	}


                    	var optimizer = new pso.Optimizer();

                    	optimizer.setObjectiveFunction(function (x, done) {
                    		setTimeout(function () {
                    			log('x');
                    			done(-Math.pow(x[0], 2));
                    		}, Math.random() * 800 + 20);
                    	}, {
                    		async: true
                    	});

                    	var initialPopulationSize = 20;
                    	var domain = [new pso.Interval(-5, 5)];

                    	optimizer.init(initialPopulationSize, domain);

                    	var iterations = 0, maxIterations = 10;

                    	function loop() {
                    		if (iterations >= maxIterations) {
                    			log([
                    				'\n--- ---\nOptimasi selesai',
                    				'Nilai terbaik yang didapatkan f(x): ' + optimizer.getBestFitness(),
                    				'dengan x = ' + optimizer.getBestPosition()[0]
                    			].join('\n'));
                    		} else {
                    			iterations++;
                    			log('\nIterasi ke - ' + iterations + ' dari ' + maxIterations + ' ');
                    			optimizer.step(loop);
                    		}
                    	}

                    	log('Mulai melakukan proses optimasi');
                    	loop();
                   }
             </script>

             <body onload="init()">

            	<div class="app">
            		<p class="desc">
            			Mencari nilai x yang dapat memaksimalkan fungsi tujuan atau objective function <code>f(x) = -x^2, di mana x dalam interval [-5, 5]</code>.
            			Fungsi objektif tersebut dilakukan returns secara asynchron melalui penggunakan
            			<code>setTimeout</code> dengan nilai random kecil sebagai timeout.
            			Di mana setiap nilai yang ditelusuri <code>x</code> merepresentasikan pemanggilan objective function.
            		</p>
            		<div class="small"><b>Detail hasil proses iterasi (Real-time):</b></div>
            		<textarea id="out" rows="16" cols="80"></textarea>
            	</div>

            </body>
            <!--- </html> --->

        '''

    return render_template_string(A_a+template_view+Z_z)

@app.route("/my_async", methods=['GET','POST'])
def my_async():
    # ref.:
    # [0] https://github.com/adrianton3/pso.js/tree/master/examples/async

    template_view = '''
            <!--- <!DOCTYPE html> --->
            <!--- <html> --->
            <!--- <head lang="en"> --->
            <!---	<meta charset="UTF-8"> --->
            <!---	<link rel="stylesheet" href="async.css"> --->
            <!---	<title>Async - pso.js demo</title> --->
            <!---</head> --->
            <!--- <body> --->

            	<div class="app">
            		<p class="desc">
            			Mencari nilai x yang dapat memaksimalkan fungsi tujuan atau objective function <code>f(x) = -x^2, di mana x dalam interval [-5, 5]</code>.
            			Fungsi objektif tersebut dilakukan returns secara asynchron melalui penggunakan
            			<code>setTimeout</code> dengan nilai random kecil sebagai timeout.
            			Di mana setiap nilai yang ditelusuri <code>x</code> merepresentasikan pemanggilan objective function.
            		</p>
            		<div class="small"><b>Detail hasil proses iterasi (Real-time):</b></div>
            		<textarea id="out" rows="16" cols="80"></textarea>
            	</div>

            	<script type="text/javascript" src="{{ url_for('static', filename = 'js/mylib_async/pso.js') }}"></script>
            	<script type="text/javascript" src="{{ url_for('static', filename = 'js/mylib_async/async.js') }}"></script>
            <!--- </body>
            <!--- </html>

        '''

    return render_template_string(A_a+template_view+Z_z)

@app.route("/homepage", methods=['GET', 'POST'])
def homepage():
    # if request.method == 'POST':
    #     # holder = Submission(request.form['submitText'])
    #     holder = request.form['submitText']
    #     it = holder.res.splitlines()

    #     for line in range(len(it)):
    #         global tst
    #         if "What kind of submission is this?" in it[line]:
    #             if "Sold Property" in it[line+1]:
    #                 # tst = Sale(request.form['submitText'])
    #                 tst = request.form['submitText']
    #                 return jsonify(tst=tst)
    #             elif "Financed" in it[line+1]:
    #                 # tst = Loan(request.form['submitText'])
    #                 tst = request.form['submitText']
    #                 return jsonify(tst=tst)
    #             else:
    #                 # tst = Lease(request.form['submitText'])
    #                 tst = request.form['submitText']
    #                 return jsonify(tst=tst)

    def myfunction(x):
        for i in range(10):
         return 'Iteration ' + str(i) + '/10'


    if request.method == 'POST':
        story = request.form.get('story')
        if story:
            result = myfunction(story)
            return jsonify(result=result)
        else:
            return jsonify(result='Input needed')

    # return render_template('index.html')

    template_view = '''
            <!--- <html> --->
            <!--- <head> --->
            <!--- </head> --->
            <!--- <body> --->

            <script src="//ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>
            <script type=text/javascript>
                $(function() {
                  $('a#process_input').bind('click', function() {
                    $.post('/', {
                      story: $('textarea[name="story"]').val(),
                    }, function(data) {
                      $('#result').text(data.result);
                    });
                    return false;
                  });
                });
            </script>

            <div class='container'>
            <form>
                <textarea id="text_input" rows="4" cols="80" name=story></textarea>
                <br>
                <a href=# id=process_input><button class='btn btn-default'>Submit</button></a>
            </form>
            <br>
            <p><h2 align='center'>Result:</h2><h2 id=result align='center'></h2></p>

        </div>

            <!--- <div class="textAreaCont"> --->
            <!---    <textarea rows="4" cols="50" class="textArea" id="getText" readonly></textarea> --->
            <!---    <button class="smallButton" id="saveAsButton">SAVE AS</button> --->
            <!---    <button class="smallButton" id="copyButton">COPY</button> --->
            <!--- </div> --->

        '''

    return render_template_string(A_a+template_view+Z_z)

@app.route("/testView", methods=['GET', 'POST'])
def testView():

    if request.method == 'POST': # dioperasikan dihalaman sendiri tanpa send ke route, misal /testView

        var1_in = float(request.form['var1'])
        var2_in = request.form['var2']
        var3_in = request.form['var3']
        c = 2*var1_in

        template_view = '''

            <!DOCTYPE html>
            <html lang="en"><head>
            <meta http-equiv="content-type" content="text/html; charset=UTF-8">
                    <meta charset="UTF-8">
                    <meta http-equiv="X-UA-Compatible" content="IE=edge">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <link href="{{ url_for('static', filename = 'css/css_testView/tailwind.min.css') }}" rel="stylesheet">
                    <title>Web App Prediksi Hasil Pengujian</title>
                </head>
                <body class="
                        flex flex-col
                        justify-between
                        item-center
                        p-8
                        min-h-screen
                        bg-gradient-to-r
                        from-green-400
                        to-blue-500
                        text-gray-800
                        md:flex-row
                    ">
                    <main class="bg-white p-16 rounded-lg">
                        <h1 class="text-3xl font-bold mb-8">
                            Prediksi Hasil Pengujian:
                        </h1>

                        <form action="/testView" method="post" class="flex flex-col">
                            <label for="Usia" class="mb-2">Nama Var. 1</label>
                            <input type="text" name="var1" {% if var1 is defined and var1 %} value="{{var1}}" {% else %} value="" {% endif %} placeholder="Variabel_1" required="required" class="p-4 bg-gray-100 rounded-md">

                            <label for="Nama Var. 2" class="mt-4 mb-2">Jenis Kelamin:</label>
                            <select name="var2" id="Var2" class="p-4 bg-gray-100 rounded-md">
                                {% if var2 is defined and var2 %}
                                <option value="pria" {% if(var2=='pria') %} selected="selected" {% endif %}>Pria</option>
                                <option value="wanita" {% if(var2=='wanita') %} selected="selected" {% endif %}>Wanita</option>
                                {% else %}
                                <option value="pria" selected="selected">Pria</option>
                                <option value="wanita">Wanita</option>
                                {% endif %}
                            </select>

                            <label for="Nama Var. 3" class="mt-4 mb-2">Nama Var. 3:</label>
                            <select name="var3" id="Var3" class="p-4 bg-gray-100 rounded-md">
                                {% if var3 is defined and var3 %}
                                <option value="Ya" {% if(var3=='Ya') %} selected="selected" {% endif %}>Ya</option>
                                <option value="Tidak" {% if(var3=='Tidak') %} selected="selected" {% endif %}>Tidak</option>
                                {% else %}
                                <option value="Ya" selected="selected">Ya</option>
                                <option value="Tidak">Tidak</option>
                                {% endif %}
                            </select>

                            <button type="submit" class="
                                    flex
                                    justify-center
                                    align-center
                                    mt-8
                                    p-4
                                    bg-gradient-to-r
                                    from-green-400
                                    to-blue-500
                                    text-white
                                    rounded-md
                                ">
                                Hitung Hasil Prediksi
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-right ml-4">
                                    <line x1="5" y1="12" x2="19" y2="12"></line>
                                    <polyline points="12 5 19 12 12 19"></polyline>
                                </svg>
                            </button>
                        </form>
                    </main>
                    <section class="mt-8 h-full flex-auto rounded-lg md:mt-0 md:ml-8">
                        <div class="bg-white p-16 rounded-lg">
                            <h1 class="text-2xl mb-8">
                                Estimasi hasil prediksinya adalah
                            </h1>
                            {% if c_save is defined and c_save %}
                            <h2 class="text-5xl font-bold">USD {{c_save}}</h2>
                            {% endif %}

                            <div class="mt-8">
                                {% if var1 is defined and var1 %}
                                <p>Usia: {{var1}} tahun</p>
                                <p>Jenis kelamin: {{var2}}</p>
                                <p>Status perokok: {{var3}}</p>
                                {% endif %}
                            </div>

                        </div>
                        <div class="flex justify-around bg-white p-16 rounded-lg mt-8">
                            <img class="h-24 md:h-28" src="{{ url_for('static', filename = 'img/logo filkom.png') }}" alt="logo-filkom">
                            <img class="h-24 md:h-28" src="{{ url_for('static', filename = 'img/conan.jpg') }}" alt="kartun-conan">
                        </div>
                    </section>

            </body></html>

        '''

        return render_template_string(template_view, var1 = var1_in, var2 = var2_in, var3 = var3_in, c_save = c)

    else: # untuk yang 'GET' data awal untuk di send ke /testView

        template_view = '''

            <!DOCTYPE html>
            <html lang="en"><head>
            <meta http-equiv="content-type" content="text/html; charset=UTF-8">
                    <meta charset="UTF-8">
                    <meta http-equiv="X-UA-Compatible" content="IE=edge">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <link href="{{ url_for('static', filename = 'css/css_testView/tailwind.min.css') }}" rel="stylesheet">
                    <title>Web App Prediksi Hasil Pengujian</title>
                </head>
                <body class="
                        flex flex-col
                        justify-between
                        item-center
                        p-8
                        min-h-screen
                        bg-gradient-to-r
                        from-green-400
                        to-blue-500
                        text-gray-800
                        md:flex-row
                    ">
                    <main class="bg-white p-16 rounded-lg">
                        <h1 class="text-3xl font-bold mb-8">
                            Prediksi Hasil Pengujian:
                        </h1>

                        <form action="/testView" method="post" class="flex flex-col">
                            <label for="Usia" class="mb-2">Nama Var. 1</label>
                            <input type="text" name="var1" {% if var1 is defined and var1 %} value="{{var1}}" {% else %} value="" {% endif %} placeholder="Variabel_1" required="required" class="p-4 bg-gray-100 rounded-md">

                            <label for="Nama Var. 2" class="mt-4 mb-2">Jenis Kelamin:</label>
                            <select name="var2" id="Var2" class="p-4 bg-gray-100 rounded-md">
                                {% if var2 is defined and var2 %}
                                <option value="pria" {% if(var2=='pria') %} selected="selected" {% endif %}>Pria</option>
                                <option value="wanita" {% if(var2=='wanita') %} selected="selected" {% endif %}>Wanita</option>
                                {% else %}
                                <option value="pria" selected="selected">Pria</option>
                                <option value="wanita">Wanita</option>
                                {% endif %}
                            </select>

                            <label for="Nama Var. 3" class="mt-4 mb-2">Nama Var. 3:</label>
                            <select name="var3" id="Var3" class="p-4 bg-gray-100 rounded-md">
                                {% if var3 is defined and var3 %}
                                <option value="Ya" {% if(var3=='Ya') %} selected="selected" {% endif %}>Ya</option>
                                <option value="Tidak" {% if(var3=='Tidak') %} selected="selected" {% endif %}>Tidak</option>
                                {% else %}
                                <option value="Ya" selected="selected">Ya</option>
                                <option value="Tidak">Tidak</option>
                                {% endif %}
                            </select>

                            <button type="submit" class="
                                    flex
                                    justify-center
                                    align-center
                                    mt-8
                                    p-4
                                    bg-gradient-to-r
                                    from-green-400
                                    to-blue-500
                                    text-white
                                    rounded-md
                                ">
                                Hitung Hasil Prediksi
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-right ml-4">
                                    <line x1="5" y1="12" x2="19" y2="12"></line>
                                    <polyline points="12 5 19 12 12 19"></polyline>
                                </svg>
                            </button>
                        </form>
                    </main>
                    <section class="mt-8 h-full flex-auto rounded-lg md:mt-0 md:ml-8">
                        <div class="bg-white p-16 rounded-lg">
                            <h1 class="text-2xl mb-8">
                                Estimasi hasil prediksinya adalah
                            </h1>
                            {% if c_save is defined and c_save %}
                            <h2 class="text-5xl font-bold">USD {{c_save}}</h2>
                            {% endif %}

                            <div class="mt-8">
                                {% if var1 is defined and var1 %}
                                <p>Usia: {{var1}} tahun</p>
                                <p>Jenis kelamin: {{var2}}</p>
                                <p>Status perokok: {{var3}}</p>
                                {% endif %}
                            </div>

                        </div>
                        <div class="flex justify-around bg-white p-16 rounded-lg mt-8">
                            <img class="h-24 md:h-28" src="{{ url_for('static', filename = 'img/logo filkom.png') }}" alt="logo-filkom">
                            <img class="h-24 md:h-28" src="{{ url_for('static', filename = 'img/conan.jpg') }}" alt="kartun-conan">
                        </div>
                    </section>

            </body></html>

        '''

        return render_template_string(template_view)

@app.route("/testView2", methods=['GET', 'POST'])
def testView2():

    if request.method == 'POST': # dioperasikan dihalaman sendiri tanpa send ke route, misal /testView2

        var1_in = float(request.form['var1'])
        var2_in = request.form['var2']
        var3_in = request.form['var3']
        c = 2*var1_in

        template_view = '''
            <script type="text/javascript" src="{{ url_for('static', filename = 'js/inputmask/jquery.inputmask.bundle.min.js') }}"></script>
            <script type="text/javascript" src="{{ url_for('static', filename = 'js/jquery.min.js') }}"></script>

            <div class="row">
                    <div class="col-md-6">
                        <div class="white-box">
                            <h3 class="box-title m-b-0">Prediksi Hasil Pengujian: </h3>
                            <p class="text-muted m-b-30 font-13"> masukkan nilai parameter Anda </p>
                            <form action="/testView2" method="post" class="form-horizontal">
                                <div class="form-group">
                                    <label for="exampleInputuname" class="col-sm-3 control-label">Nilai Skor*</label>
                                    <div class="col-sm-9">
                                        <div class="input-group">
                                            <input type="text" name="var1" {% if var1 is defined and var1 %} value="{{var1}}" {% else %} value="" {% endif %} class="form-control" id="exampleInputuname" placeholder="Skor" required="required">
                                            <div class="input-group-addon"><i class="ti-user"></i></div>
                                        </div>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="exampleInputEmail1" class="col-sm-3  control-label">Jenis Gender*</label>
                                    <div class="col-sm-9">
                                    <select name="var2" class="form-control">
                                        {% if var2 is defined and var2 %}
                                        <option value="pria" {% if(var2=='pria') %} selected="selected" {% endif %}>Pria</option>
                                        <option value="wanita" {% if(var2=='wanita') %} selected="selected" {% endif %}>Wanita</option>
                                        {% else %}
                                        <option value="pria" selected="selected">Pria</option>
                                        <option value="wanita">Wanita</option>
                                        {% endif %}
                                    </select>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="inputEmail3" class="col-sm-3 control-label">Percentage</label>
                                     <div class="col-sm-9">
                                    <input type="text" name="var3" {% if var3 is defined and var3 %} value="{{var3}}" {% else %} value="" {% endif %} class="form-control percentage-inputmask" id="percentage-mask" placeholder="Enter Value in %">
                                    </div>
                                </div>

                                <div class="form-group m-b-0">
                                    <div class="col-sm-offset-3 col-sm-9 text-right">
                                        <button type="submit" class="btn btn-info waves-effect waves-light m-t-10">Hitung Hasil Prediksi</button>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="white-box row">
                            <h3 class="box-title m-b-0">Estimasi hasil prediksinya adalah </h3>
                            {% if c_save is defined and c_save %}
                            <p class="text-muted m-b-30 font-13"> Nilai Skor = {{c_save}} </p>
                            {% endif %}

                            <div class="mt-8">
                                {% if var1 is defined and var1 %}
                                <p>Nilai Skor: {{var1}} tahun</p>
                                <p>Jenis kelamin: {{var2}}</p>
                                <p>Persentase (%): {{var3}}</p>
                                {% endif %}
                            </div>

                        </div>
                    </div>

                    <div class="col-md-6">
                    <div class="white-box mt-8 row">
                    <div class="justify-around bg-white rounded-lg">
                            <img class="col-md-3 col-xs-12" src="{{ url_for('static', filename = 'img/logo filkom.png') }}" alt="logo-filkom">
                            <img class="col-md-3 col-xs-12" src="{{ url_for('static', filename = 'img/conan.jpg') }}" alt="kartun-conan">
                    </div>

                     </div>
                    </div>

                </div>

        '''

        return render_template_string(A_a+template_view+Z_z, var1 = var1_in, var2 = var2_in, var3 = var3_in, c_save = c)

    else: # untuk yang 'GET' data awal untuk di send ke /testView2

        template_view = '''
            <script type="text/javascript" src="{{ url_for('static', filename = 'js/inputmask/jquery.inputmask.bundle.min.js') }}"></script>
            <script type="text/javascript" src="{{ url_for('static', filename = 'js/jquery.min.js') }}"></script>

            <div class="row">
                    <div class="col-md-6">
                        <div class="white-box">
                            <h3 class="box-title m-b-0">Prediksi Hasil Pengujian: </h3>
                            <p class="text-muted m-b-30 font-13"> masukkan nilai parameter Anda </p>
                            <form action="/testView2" method="post" class="form-horizontal">
                                <div class="form-group">
                                    <label for="exampleInputuname" class="col-sm-3 control-label">Nilai Skor*</label>
                                    <div class="col-sm-9">
                                        <div class="input-group">
                                            <input type="text" name="var1" {% if var1 is defined and var1 %} value="{{var1}}" {% else %} value="" {% endif %} class="form-control" id="exampleInputuname" placeholder="Skor" required="required">
                                            <div class="input-group-addon"><i class="ti-user"></i></div>
                                        </div>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="exampleInputEmail1" class="col-sm-3  control-label">Jenis Gender*</label>
                                    <div class="col-sm-9">
                                    <select name="var2" class="form-control">
                                        {% if var2 is defined and var2 %}
                                        <option value="pria" {% if(var2=='pria') %} selected="selected" {% endif %}>Pria</option>
                                        <option value="wanita" {% if(var2=='wanita') %} selected="selected" {% endif %}>Wanita</option>
                                        {% else %}
                                        <option value="pria" selected="selected">Pria</option>
                                        <option value="wanita">Wanita</option>
                                        {% endif %}
                                    </select>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="inputEmail3" class="col-sm-3 control-label">Percentage</label>
                                     <div class="col-sm-9">
                                    <input type="text" name="var3" {% if var3 is defined and var3 %} value="{{var3}}" {% else %} value="" {% endif %} class="form-control percentage-inputmask" id="percentage-mask" placeholder="Enter Value in %">
                                    </div>
                                </div>

                                <div class="form-group m-b-0">
                                    <div class="col-sm-offset-3 col-sm-9 text-right">
                                        <button type="submit" class="btn btn-info waves-effect waves-light m-t-10">Hitung Hasil Prediksi</button>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="white-box row">
                            <h3 class="box-title m-b-0">Estimasi hasil prediksinya adalah </h3>
                            {% if c_save is defined and c_save %}
                            <p class="text-muted m-b-30 font-13"> Nilai Skor = {{c_save}} </p>
                            {% endif %}

                            <div class="mt-8">
                                {% if var1 is defined and var1 %}
                                <p>Nilai Skor: {{var1}} tahun</p>
                                <p>Jenis kelamin: {{var2}}</p>
                                <p>Persentase (%): {{var3}}</p>
                                {% endif %}
                            </div>

                        </div>
                    </div>

                    <div class="col-md-6">
                    <div class="white-box mt-8 row">
                    <div class="justify-around bg-white rounded-lg">
                            <img class="col-md-3 col-xs-12" src="{{ url_for('static', filename = 'img/logo filkom.png') }}" alt="logo-filkom">
                            <img class="col-md-3 col-xs-12" src="{{ url_for('static', filename = 'img/conan.jpg') }}" alt="kartun-conan">
                    </div>

                     </div>
                    </div>
                </div>

        '''

        return render_template_string(A_a+template_view+Z_z)


@app.route('/launchpad_menu')
def launchpad_menu():
   return render_template("launchpad_menu.html")