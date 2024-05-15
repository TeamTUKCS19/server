import os
import cv2
import torch
import pathlib
import s3
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from pathlib import Path


from db_setup import db, DroneData, Wall, Building, init_db
from flask_session import Session
import tempfile
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


@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files['file']
    location = {
        'latitude': float(request.form['latitude']),
        'longitude': float(request.form['longitude']),
        'altitude': float(request.form['altitude'])
    }

    video_path = os.path.join(SAVE_DIR, 'uploaded_video.mp4')
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    frames = process_video(cap, location)
    cap.release()
    os.remove(video_path)

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


##벽면 등록 엔드포인트
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
    return "Wall registered successfully", 201


def save_to_db(latitude, longitude, altitude, s3_url):
    wall_id = session.get('wall_id')

    new_data = DroneData(
        wall_id=wall_id,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        s3_url=s3_url,
    )
    db.session.add(new_data)
    db.session.commit()

# model_path = "/app/TUKproject/flask_api/best.pt"
model_path = "./best.pt"

model = torch.hub.load('../custom_yolov5', 'custom', path=model_path, source='local')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# SAVE_DIR = '../saved_Detection'
SAVE_DIR = tempfile.gettempdir()
# SAVE_DIR_video = '../saved_Detection_video'

COLOR_MAPPING = {
    'Bullet_impact': (255, 0, 0),  # 파란색
    'Explosion_impact': (0, 255, 0),  # 초록색
    'normal_crack': (0, 0, 255),  # 빨간색
    'severe_crack': (255, 255, 0)  # 노란색
}

CLASS_MAPPING = {
    0: 'Bullet_impact',
    1: 'Explosion_impact',
    2: 'normal_crack',
    3: 'severe_crack'
}

# 무시할 객체 클래스 리스트
IGNORE_CLASSES = {'Bullet_impact', 'Explosion_impact'}


# 비디오에서 특정 프레임에 맞춰서 이미지를 뽑고 그 이미지를 YOLO모델을 통해 균열 검출하는 함수
def process_video(cap, location):
    frames = []
    s3_urls = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도

    latitude = location['latitude']
    longitude = location['longitude']
    altitude = location['altitude']

    count = 0
    # frame_rate = 1 이면, x초의 영상을 1초씩 자른다.  만약 값이 0.1일경우 영상을 0.1초씩 자른다.
    frame_rate = 0.5
    interval_frames = int(fps * frame_rate)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % interval_frames == 0:
            results = model(frame)

            filename = f'frame_{count}.jpg'
            filepath = os.path.join(SAVE_DIR, filename)

            processed_frame = draw_boxes(frame, results)

            cv2.imwrite(filepath, processed_frame)
            frames.append(filepath)
            # upload to AWS_S3
            # s3 불필요할 때 아랫줄 주석처리.
            s3_url = s3.upload_to_s3(processed_frame, filename)
            # s3_urls.append(s3_url)
            save_to_db(latitude, longitude, altitude, s3_url)

    # output_video_path = os.path.join(SAVE_DIR_video, 'processed_video.mp4')
    # create_video_from_frames(frames, output_video_path)

    cap.release()
    return frames


# 바운딩박스를 그려주는 함수
def draw_boxes(image, results):
    for result in results.xyxy[0]:
        box = result[:4]  # 바운딩 박스의 좌표값 (x1, y1, x2, y2)
        label_index = int(result[5])
        label = CLASS_MAPPING.get(label_index, 'Unknown')  # 객체 클래스
        conf = result[4]  # 신뢰도

        if label in IGNORE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = COLOR_MAPPING.get(label, (0, 0, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label} ({conf:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# 13.209.231.12 : EC2_Public_IP
# 로컬에서 실행 시 0.0.0.0 으로 바꿔주세요
# EC2에서 실행시 인자에 port = 9900 추가해주세요
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9900, debug=True)
