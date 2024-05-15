import tempfile

import torch
import cv2
import s3
import os
from app import save_to_db

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
def process_video(cap, db, location):
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
            save_to_db(latitude , longitude, altitude, s3_url)

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



"""
# 이미지 테스트용 함수. (비중요)
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
"""

"""
# Bounding box 처리한 이미지를 합쳐서 영상으로 생성해주는 함수. ( 중요하지 않은 기능 )
def create_video_from_frames(frames, output_video_path):
    if not frames:
        return None

    frame = cv2.imread(frames[0])
    height, width, _ = frame.shape

    # 비디오 저장 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))  # 여기서 30은 임의의 프레임 속도.

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

    return output_video_path
"""