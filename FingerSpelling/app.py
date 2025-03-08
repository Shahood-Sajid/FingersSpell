import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np

import torch
from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)


# Load Model
model = torch.hub.load("", "custom", path ="best.pt", force_reload=True,source= 'local')

model.eval()
model.conf = 0.25 
model.iou = 0.45 

from io import BytesIO

def gen():
    cap=cv2.VideoCapture(0)

    while(cap.isOpened()):
        
        
        success, frame = cap.read()
        if success == True:

            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 50)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 50)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            

            img = Image.open(BytesIO(frame))
            results = model(img, size=640)
            results.print()  # print results to screen
            
            
            
            img = np.squeeze(results.render())
            
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            break

        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/video')
def video():

    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)

