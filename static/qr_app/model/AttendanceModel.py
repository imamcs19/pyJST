import sys
# append current python modules' folder path
# example: need to import module.py present in '/path/to/python/module/not/in/syspath'
# sys.path.append('/home/bigdatafga/mysite')

import os
userhome = os.path.expanduser("~").split("/")[-1]
mypath = "/home/"+userhome+"/mysite"

sys.path.append(mypath)

# from static.qr_app import db
# from static.qr_app.model.StudentModel import Student
from static.qr_app.model.StudentModel import Student
import datetime

from flask_app import db_qr


class Attendance(db_qr.Model):
    __tablename__ = "attendance"

    id = db_qr.Column(db_qr.Integer, unique=True, primary_key=True, nullable=False)
    student_id = db_qr.Column(db_qr.Integer, db_qr.ForeignKey("student.id"), nullable=False)
    time = db_qr.Column(db_qr.DateTime, nullable=False, default=datetime.datetime.utcnow())

    def __init__(self, student_id):
        self.student_id = Student.findById(student_id).id

    def __repr__(self):
        return "<Student id: {}>".format(self.student_id)

    def save(self):
        db_qr.session.add(self)
        db_qr.session.commit()

    def delete(self):
        db_qr.session.delete(self)
        db_qr.session.commit()

    @staticmethod
    def getAll():
        data = Attendance.query.all()
        result = list()
        for value in data:
            obj = {
                "id": value.id,
                "student": Student.getById(value.student_id),
                "time": value.time
            }
            result.append(obj)
        return result
