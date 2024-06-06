import os
import cv2

from flask import Flask, request, render_template, redirect, url_for, session

import work_yolo as yolo
from db_setup import db, DroneData, Wall, Building, init_db
from flask_session import Session
from s3 import s3

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
init_db(app)


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        if 'video' in request.files:
            return redirect(url_for('upload_video'))
        elif 'image' in request.files:
            return redirect(url_for('upload_image'))
    return render_template("Main_page.html")


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        location = {
            'latitude': 32.0,
            'longitude': 64.0,
            'altitude': 12.0
        }
        video_path = os.path.join(yolo.SAVE_DIR, 'uploaded_video.mp4')
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)

        yolo.process_video(cap, location)
        cap.release()
        os.remove(video_path)
        s3.download_all_files(s3.S3_BUCKET, local_dir)
        return 'File uploaded successfully', 200


@app.route('/upload_video', methods=['POST'])
def upload_video():
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
    os.remove(video_path)
    s3.download_all_files(s3.S3_BUCKET, local_dir)
    return "File uploaded and processed successfully", 200


##건물 등록 엔드포인트
@app.route('/register_building', methods=['POST'])
def register_building():
    building_name = request.form['building_name']

    building = Building.query.filter_by(name=building_name).first()

    if not building:
        building = Building(name=building_name)
        db.session.add(building)
        db.session.commit()

    session['building_id'] = building.id
    return "building registered successfully", 201


# 벽면 등록 엔드포인트
@app.route('/register_wall', methods=['POST'])
def register_wall():
    building_id = session.get('building_id')

    direction = request.form['direction']

    wall = Wall.query.filter_by(direction=direction, building_id=building_id).first()
    if not wall:
        wall = Wall(direction=direction, building_id=building_id)
        db.session.add(wall)
        db.session.commit()

    session['wall_id'] = wall.id
    temp_wall_id = session.get('wall_id')
    print(temp_wall_id)
    if session.get('wall_id') is None:
        return "wall bad request", 400
    else:
        return "Wall registered successfully", 201


def save_to_db(latitude, longitude, altitude, width, risk, s3_url_cropped, s3_url_bbox):
    # wall_id = session.get('wall_id')
    # wall_id = 1
    wall = Wall.query.order_by(Wall.id.desc()).first()

    print(wall.id)
    new_data = DroneData(
        wall_id=wall.id,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        width=width,
        risk=risk,
        s3_url_cropped=s3_url_cropped,
        s3_url_bbox=s3_url_bbox,
    )
    db.session.add(new_data)
    db.session.commit()


local_dir = '../saved_Detection'

# 13.209.231.12 : EC2_Public_IP
# 로컬에서 실행 시 0.0.0.0 으로 바꿔주세요
# EC2에서 실행시 인자에 port = 9900 추가해주세요
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9900, debug=True)
