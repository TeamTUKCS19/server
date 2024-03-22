import os
import cv2
import torch
import time


import sys

from flask import Flask, request, jsonify, render_template, redirect, url_for

SAVE_DIR = './saved_Detection'
SAVE_DIR_video = './saved_Detection_video'

## sys.path.insert(0, '../custom_yolov5')

app = Flask(__name__)

# model_path = "/app/TUKproject/flask_api/best.pt"
model_path = "./best.pt"
## model = torch.hub.load('../custom_yolov5', 'custom', path=model_path, source='local')

model = torch.hub.load('../custom_yolov5', 'custom', path=model_path, source='local')


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

    video_path = os.path.join(SAVE_DIR, 'uploaded_video.mp4')
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    frames = process_video(cap)

    return jsonify(frames)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    image_file = request.files['image']

    image_path = os.path.join(SAVE_DIR, 'uploaded_image.jpg')
    image_file.save(image_path)

    image = cv2.imread(image_path)

    results = model(image)

    processed_image = draw_boxes(image, results)

    output_image_path = os.path.join(SAVE_DIR, 'processed_image.jpg')
    cv2.imwrite(output_image_path, processed_image)

    return jsonify(output_image_path)


def process_video(cap):
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도

    count = 0
    frame_rate = 0.1
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

    output_video_path = os.path.join(SAVE_DIR_video, 'processed_video.mp4')
    create_video_from_frames(frames, output_video_path)

    cap.release()
    return frames


CLASS_MAPPING = {
    0: 'Bullet_impact',
    1: 'Explosion_impact',
    2: 'normal_crack',
    3: 'severe_crack'
}


def create_video_from_frames(frames, output_video_path):
    if not frames:
        return None

    frame = cv2.imread(frames[0])
    height, width, _ = frame.shape

    # 비디오 저장 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))  # 여기서 30은 임의의 프레임 속도입니다.

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

    return output_video_path


COLOR_MAPPING = {
    'Bullet_impact': (255, 0, 0),  # 파란색
    'Explosion_impact': (0, 255, 0),  # 초록색
    'normal_crack': (0, 0, 255),  # 빨간색
    'severe_crack': (255, 255, 0)  # 노란색
}


def draw_boxes(image, results):
    for result in results.xyxy[0]:
        box = result[:4]  # 바운딩 박스의 좌표값 (x1, y1, x2, y2)
        label_index = int(result[5])
        label = CLASS_MAPPING.get(label_index, 'Unknown')  # 객체 클래스
        conf = result[4]  # 신뢰도
        x1, y1, x2, y2 = map(int, box)

        color = COLOR_MAPPING.get(label, (0, 0, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'({label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    app.run(debug=True)
