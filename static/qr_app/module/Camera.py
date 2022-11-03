import sys
# append current python modules' folder path
# example: need to import module.py present in '/path/to/python/module/not/in/syspath'
# sys.path.append('/home/bigdatafga/mysite')

import os
userhome = os.path.expanduser("~").split("/")[-1]
mypath = "/home/"+userhome+"/mysite"

sys.path.append(mypath)

from static.qr_app.model.AttendanceModel import Attendance
# from static.qr_app.model.AttendanceModel import Attendance
import cv2
import pyzbar.pyzbar as pyzbar
# import os

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


class Scanner(object):
    def __init__(self):
        # Change your RSTP IP for your another device, make sure same network
        # self.cap = cv2.VideoCapture("rtsp://YOUR_IP_ADDRESS:YOUR_PORT", cv2.CAP_FFMPEG)
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
        # self.cap = cv2.VideoCapture(1)
        self.transformed_frame = None

    def __del__(self):
        self.cap.release()

    def get_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            decode = pyzbar.decode(frame)

            if decode is not None:
                for obj in decode:
                    if obj.data is not None:
                        attendance = Attendance(student_id=int(obj.data))
                        attendance.save()
                        self.__del__()

            ret, jpeg = cv2.imencode('.jpg', frame)
            self.transformed_frame = jpeg.tobytes()
            return jpeg.tobytes()
        else:
            return None

    def get_image_frame(self):
        return self.transformed_frame