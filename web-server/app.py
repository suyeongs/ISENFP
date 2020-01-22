#!/usr/bin/env python
from flask import Flask, render_template, Response, send_file
from camera import Camera
import os
import io
import cv2
import requests as req
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
camera = None

@app.route('/')
@app.route('/index')
def index():
   return render_template('index.html')

def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera

def gen(camera):
    while True:
        frame = camera.get_feed()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/')
def video_feed():
    camera = get_camera()
    return Response(gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture/')
def capture():
    camera = get_camera()
    path = camera.capture()

    files = {'myfile':open(path, 'rb')}
    url="http://3.229.202.238:8080/upload"
    response=req.post(url, files=files)

    imgurl="http://3.229.202.238:8080/static/image.jpg"
    img_data=req.get(imgurl).content
    with open(os.getcwd() + '/static/cvimage.jpg', 'wb') as handler:
        handler.write(img_data)
    print(response.text)
    result = response.text
    return render_template('capture.html', result=result)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True, threaded=True)
