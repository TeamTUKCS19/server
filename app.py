import os
import cv2
import torch
import pathlib

from flask import Flask, request, jsonify, render_template, redirect, url_for
from pathlib import Path

import work_db
import work_yolo as y
import db_setup
from flask_session import Session

# 윈도우 운영체제에서 실행할 경우에는 아래 코드 한 줄 주석처리 해야 함.
# pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)

# 세션 설정
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)
Session(app)


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
    video_file = request.files['video']

    video_path = os.path.join(y.SAVE_DIR, 'uploaded_video.mp4')
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    frames = y.process_video(cap)
    cap.release()
    os.remove(video_path)

    return jsonify(frames)

@app.route('/register_wall', methods=['POST'])
# building_id, direction
direction = request.form['direction']
word = work_db.register_wall(session, db, Wall, direction)

# 13.209.231.12 : EC2_Public_IP
# 로컬에서 실행 시 0.0.0.0 으로 바꿔주세요
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)