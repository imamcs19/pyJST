import sys
# append current python modules' folder path
# example: need to import module.py present in '/path/to/python/module/not/in/syspath'
# sys.path.append('/home/bigdatafga/mysite')

import os
userhome = os.path.expanduser("~").split("/")[-1]
mypath = "/home/"+userhome+"/mysite"

sys.path.append(mypath)

# from static.qr_app import db
from flask_app import db_qr


class Student(db_qr.Model):
    __tablename__ = "student"  # Must be defined the table name

    id = db_qr.Column(db_qr.Integer, unique=True, primary_key=True, nullable=False)
    nim = db_qr.Column(db_qr.String, nullable=False)
    name = db_qr.Column(db_qr.String, nullable=False)
    qr_code = db_qr.Column(db_qr.String, nullable=False, unique=True)
    attendance = db_qr.relationship("Attendance", backref="student", lazy="dynamic")

    def __init__(self, nim, name, qr_code):
        self.nim = nim
        self.name = name
        self.qr_code = qr_code

    def __repr__(self):
        return "<Name: {}, Nim: {}>".format(self.name, self.nim)

    def save(self):
        db_qr.session.add(self)
        db_qr.session.commit()

    def delete(self):
        db_qr.session.delete(self)
        db_qr.session.commit()

    @staticmethod
    def getAll():
        students = Student.query.all()
        result = []
        for student in students:
            obj = {
                "id": student.id,
                "nim": student.nim,
                "name": student.name,
                "qr_code": student.qr_code
            }
            result.append(obj)
        return result

    @staticmethod
    def getById(id):
        student = Student.findById(id)
        result = {
            "id": student.id,
            "nim": student.nim,
            "name": student.name,
            "qr_code": student.qr_code
        }
        return result

    @staticmethod
    def findById(id):
        return Student.query.filter_by(id=id).first()
