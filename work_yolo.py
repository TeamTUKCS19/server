import tempfile
from s3 import s3
from app import save_to_db
from ultralytics import YOLO
from cal_crack import calculate_crack as cal
import cv2

# model_path = "/app/TUKproject/flask_api/best.pt"
model_path = "weight/yolov8s_best2.pt"
model = YOLO(model_path)

# 로컬에 custom_yolo5 다운 안받아놓았으면 아랫줄 주석 처리 후 그 아랫줄의 ultralytics/yolov5 줄의 주석 제거
# model = torch.hub.load('../custom_yolov5', 'custom', path=model_path, source='local')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# SAVE_DIR = '../saved_Detection'
# 임시 저장소 사용. (임시저장소에 영상을 올려놓음)
SAVE_DIR = tempfile.gettempdir()


# SAVE_DIR_video = '../saved_Detection_video'

# 비디오에서 특정 프레임에 맞춰서 이미지를 뽑고 그 이미지를 YOLO모델을 통해 균열 검출하는 함수
def process_video(cap, location):
    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 프레임 속도

    latitude = location['latitude']
    longitude = location['longitude']
    altitude = location['altitude']

    count = 0
    spelling = 'A'
    # frame_rate = 1 이면, x초의 영상을 1초씩 자른다.  만약 값이 0.1일경우 영상을 0.1초씩 자른다.
    frame_rate = 0.3
    interval_frames = int(fps * frame_rate)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if spelling > 'z':
            spelling = 'A'
        else:
            spelling += 1

        if count % interval_frames == 0:
            # frame 은 하나의 image file
            results = model(frame)
            # results : 한 frame에서 발견된 균열들의 정보 (1개 이상이 될 수 있다.)
            for result in results:  # result : 균열 하나씩 꺼냄
                bbox = result.boxes
                processed_frame = draw_boxes(frame, bbox)
                if processed_frame is None:
                    continue
                else:
                    filename_bbox = f'{spelling}{count}.jpg'
                    count += 1
                    cropped_images = cal.crop_crack_region(frame, bbox)
                    if not cropped_images:
                        continue
                    for idx, (cropped_frame, label, conf) in enumerate(cropped_images):
                        filename_cropped = f'{spelling}{count}_{idx}.jpg'
                        # cropped_frame에 대해 Iamge_Binariation 진행
                        binary_sauvola_Pw_bw, binary_sauvola_Pw = cal.image_binaryzation(cropped_frame)
                        skeleton_Pw = cal.img_skeletonize(binary_sauvola_Pw)
                        edges_Pw = cal.detect_edge(binary_sauvola_Pw)
                        real_width = cal.calculate_width(skeleton_Pw, edges_Pw)
                        risk = cal.categorize_risk(real_width)

                        if cropped_frame is None:
                            continue
                        else:
                            # upload to AWS_S3
                            # s3 불필요할 때 아랫줄 주석처리.
                            s3_url_cropped = s3.upload_to_s3(cropped_frame, filename_cropped)
                            s3_url_bbox = s3.upload_to_s3(processed_frame, filename_bbox)
                            # s3_urls.append(s3_url)
                            save_to_db(latitude, longitude, altitude, real_width, risk, s3_url_cropped, s3_url_bbox)
                    # cropped_image = crop_crack_region(frame, results)
    cap.release()


# 바운딩박스를 그려주는 함수
# results 값으로는 균열 하나에 대한 boxes 값이 들어온다.
def draw_boxes(image, result):
    if result is None:
        return None

    box = result.xyxy  # 바운딩 박스의 좌표값 (x1, y1, x2, y2)
    label = 'Crack'  # 객체 클래스
    conf = result.conf  # 신뢰도
    x1, y1, x2, y2 = map(int, box[0])
    color = (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f'{label} ({conf:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image
