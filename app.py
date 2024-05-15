import os
import cv2
import torch
import pathlib

from flask import Flask, request, jsonify, render_template, redirect, url_for
from pathlib import Path

import work_db
import work_yolo as yolo
from db_setup import db
from flask_session import Session

# 윈도우 운영체제에서 실행할 경우에는 아래 코드 한 줄 주석처리 해야 함.
# pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)

# 세션 설정
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)
Session(app)

# DB 초기화
basedir = os.path.abspath(os.path.dirname(__file__))
database_path = os.path.join(basedir, 'drones.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + database_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
work_db.init_db(app)


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        if 'video' in request.files:
            return redirect(url_for('upload_video'))
        elif 'image' in request.files:
            return redirect(url_for('upload_image'))
    return render_template("Main_page.html")


@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        video_file = request.files['file']
        location = {
            'latitude': float(request.form['latitude']),
            'longitude': float(request.form['longitude']),
            'altitude': float(request.form['altitude'])
        }

        video_path = os.path.join(yolo.SAVE_DIR, 'uploaded_video.mp4')
        video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)

        frames = yolo.process_video(cap, location)
        cap.release()
    except Exception as e:
        return str(e), 500
    finally:
        os.remove(video_path)

    return "File uploaded and processed successfully", 200


@app.route('/register_building', methods=['POST'])
def register_building():
    building_name = request.form['building_name']
    sentence, code = work_db.register_building(db.session, building_name)
    return sentence, code


@app.route('/register_wall', methods=['POST'])
# building_id, direction
def register_wall():
    direction = request.form['direction']
    sentence, code = work_db.register_wall(db.session, direction)
    return sentence, code


# 13.209.231.12 : EC2_Public_IP
# 로컬에서 실행 시 0.0.0.0 으로 바꿔주세요
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
