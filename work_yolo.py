import tempfile
import torch
import cv2
import s3
from app import save_to_db
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage import data
from skimage.color import rgb2gray
from skimage.data import page
from skimage.filters import (threshold_sauvola)
from PIL import Image

# model_path = "/app/TUKproject/flask_api/best.pt"
model_path = "./best.pt"

# 로컬에 custom_yolo5 다운 안받아놓았으면 아랫줄 주석 처리 후 그 아랫줄의 ultralytics/yolov5 줄의 주석 제거
#model = torch.hub.load('../custom_yolov5', 'custom', path=model_path, source='local')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# SAVE_DIR = '../saved_Detection'
#임시 저장소 사용. (임시저장소에 영상을 올려놓음)
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
    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도

    latitude = location['latitude']
    longitude = location['longitude']
    altitude = location['altitude']

    count = 0
    # frame_rate = 1 이면, x초의 영상을 1초씩 자른다.  만약 값이 0.1일경우 영상을 0.1초씩 자른다.
    frame_rate = 0.3
    interval_frames = int(fps * frame_rate)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % interval_frames == 0:
            results = model(frame)

            filename = f'frame_{count}.jpg'

            processed_frame = draw_boxes(frame, results)
            #cropped_image = crop_crack_region(frame, results)
            if (processed_frame is None):
                continue
            else:
                # upload to AWS_S3
                # s3 불필요할 때 아랫줄 주석처리.
                s3_url = s3.upload_to_s3(processed_frame, filename)
                save_to_db(latitude, longitude, altitude, s3_url)
    cap.release()

def process_video2(cap, location): #균열을 검출하고 crop_crack_region 함수를 호출하여 균열이 존재하는 영역만 잘라냄

    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도

    latitude = location['latitude']
    longitude = location['longitude']
    altitude = location['altitude']

    count = 0
    frame_rate = 0.3  # 프레임 간격 설정 (0.3초마다 처리)
    interval_frames = int(fps * frame_rate)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % interval_frames == 0:
            results = model(frame)
            cropped_images = crop_crack_region(frame, results)

            if not cropped_images:
                continue

            for idx, (cropped_frame, label, conf) in enumerate(cropped_images):
                filename = f'frame_{count}_{idx}.jpg'
                #cropped_frame에 대해 Iamge_Binariation 진행하여 사이즈 측정
                if cropped_frame is None:
                    continue
                else:
                    # upload to AWS_S3
                    # s3 불필요할 때 아랫줄 주석처리.
                    s3_url = s3.upload_to_s3(cropped_frame, filename)
                    # s3_urls.append(s3_url)
                    save_to_db(latitude, longitude, altitude, s3_url)

    cap.release()

def crop_crack_region(frame, results): #균열이 검출된 영역을 잘라내는 함수
    cropped_images = []
    if results is None or len(results.xyxy[0]) == 0: #예외 처리
        return None

    for result in results.xyxy[0]:
        box = result[:4]
        label_index = int(result[5])
        label = CLASS_MAPPING.get(label_index, 'Unknown')
        reliability = result[4]

        if label in IGNORE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        cropped_images.append((cropped_image, label, reliability))
    return cropped_images

# 바운딩박스를 그려주는 함수
def draw_boxes(image, results):
    if results is None or len(results.xyxy[0]) == 0:
        return None
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
