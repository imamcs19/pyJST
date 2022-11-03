# from app import app
# from flask import request, jsonify, render_template, Response, redirect
# from app.model.StudentModel import Student
# from app.model.AttendanceModel import Attendance
# from app.module.Camera import Scanner
# import pyqrcode
# import uuid


# @app.route('/')
# def index():
#     attendance = Attendance.getAll()
#     return render_template("scan.html", data=enumerate(attendance, 1))


# @app.route("/scan", methods=["GET"])
# def scan():
#     return Response(scanner(),
#                 mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route("/student", methods=["GET", "POST"])
# def student():
#     if request.method == "POST":
#         name = request.form['name']
#         nim = request.form['nim']
#         UUID = str(uuid.uuid4())
#         qr_code = "tmp/{}.png".format(UUID)
#         student = Student(nim=nim, name=name, qr_code=qr_code)
#         student.save()
#         img = pyqrcode.create(student.id, error="L", mode="binary", version=5)
#         img.png(qr_code, scale=10)
#     students = Student.getAll()
#     return render_template("student.html", data=enumerate(students, 1))


# def scanner():
#     camera = Scanner()
#     while True:
#         frame = camera.get_video_frame()

#         if frame is not None:
#             yield (b'--frame\r\n'
#                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         else:
#             break
