from skimage.color import rgb2gray
from skimage.filters import (threshold_sauvola)
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage import feature
import queue
import math
import numpy as np


def image_binaryzation(cropped_frame):
    # sauvola_frames_Pw_bw = []
    # sauvola_frames_Pw = []

    img_gray = rgb2gray(cropped_frame)

    window_size_Pw = 71
    thresh_sauvola_Pw = threshold_sauvola(img_gray, window_size=window_size_Pw, k=0.42)

    binary_sauvola_Pw = img_gray > thresh_sauvola_Pw
    binary_sauvola_Pw_bw = img_gray > thresh_sauvola_Pw

    binary_sauvola_Pw_bw.dtype = 'uint8'
    binary_sauvola_Pw_bw *= 255

    # sauvola_frames_Pw_bw.append(binary_sauvola_Pw_bw)
    # sauvola_frames_Pw.append(binary_sauvola_Pw)

    return binary_sauvola_Pw_bw, binary_sauvola_Pw


def img_skeletonize(sauvola_frames_Pw):
    # skeleton_frames_Pw = []

    img_Pw = invert(sauvola_frames_Pw)
    skeleton_Pw = skeletonize(img_Pw)
    skeleton_Pw.dtype = 'uint8'
    skeleton_Pw *= 255

    # skeleton_frames_Pw.append(skeleton_Pw)

    return skeleton_Pw


def detect_edge(sauvola_frames_Pw):
    # edges_frames_Pw = []
    # edges_frames_Pl = []

    edges_Pw = feature.canny(sauvola_frames_Pw, 0.09)
    edges_Pw.dtype = 'uint8'
    edges_Pw *= 255

    # edges_frames_Pw.append(edges_Pw)

    return edges_Pw


def calculate_width(skeleton_frames_Pw, edges_frames_Pw):
    dx_dir_right = [-5, -5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5]
    dy_dir_right = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1]

    dx_dir_left = [5, 5, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -5]
    dy_dir_left = [0, -1, -2, -3, -4, -5, -5, -5, -5, -5, -4, -3, -2, -1]

    dx_bfs = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy_bfs = [0, 1, 1, 1, 0, -1, -1, -1]

    start = [0, 0]
    next = []
    q = queue.Queue()
    q.put(start)

    len_x = skeleton_frames_Pw.shape[0]
    len_y = skeleton_frames_Pw.shape[1]

    visit = np.zeros((len_x, len_y))
    crack_width_list = []

    # Skeleton pixel 로부터 균열의 진행 방향을 찾아냄
    while q.empty() == 0:
        next = q.get()
        x = next[0]
        y = next[1]
        right_x = right_y = left_x = left_y = -1

        if skeleton_frames_Pw[x][y] == 255:
            # skeleton을 바탕으로 균열의 진행 방향을 구함
            for i in range(0, len(dx_dir_right)):
                right_x = x + dx_dir_right[i]
                right_y = y + dy_dir_right[i]
                if right_x < 0 or right_y < 0 or right_x >= len_x or right_y >= len_y:
                    right_x = right_y = -1
                    continue

                if skeleton_frames_Pw[right_x][right_y] == 255: break;
                if i == 13: right_x = right_y = -1

            if right_x == -1:
                right_x = x
                right_y = y

            for i in range(0, len(dx_dir_left)):
                left_x = x + dx_dir_left[i]
                left_y = y + dy_dir_left[i]
                if left_x < 0 or left_y < 0 or left_x >= len_x or left_y >= len_y:
                    left_x = left_y = -1
                    continue
                if skeleton_frames_Pw[left_x][left_y] == 255: break
                if i == 13: left_x = left_y = -1

            if left_x == -1:
                left_x = x
                left_y = y

            # acos 공식을 바탕으로 균열의 진행 방향을 각도(theta)로 나타냄
            base = right_y - left_y
            height = right_x - left_x
            hypotenuse = math.sqrt(base * base + height * height)

            if base == 0 and height != 0:
                theta = 90.0
            elif base == 0 and height == 0:
                continue
            else:
                theta = math.degrees(
                    math.acos((base * base + hypotenuse * hypotenuse - height * height) / (2.0 * base * hypotenuse)))

            theta += 90
            dist = 0

            # 균열 진행 방향의 수직선과 Edge가 만나면, 그 거리를 구함
            for i in range(0, 2):
                pix_x = x
                pix_y = y
                if theta > 360:
                    theta -= 360
                elif theta < 0:
                    theta += 360

                if theta == 0.0 or theta == 360.0:
                    while True:
                        pix_y += 1
                        if pix_y >= len_y:
                            pix_x = x
                            pix_y = y
                            break
                        if edges_frames_Pw[pix_x][pix_y] == 255: break

                elif theta == 90.0:
                    while True:
                        pix_x -= 1
                        if pix_x < 0:
                            pix_x = x
                            pix_y = y
                            break
                        if edges_frames_Pw[pix_x][pix_y] == 255: break

                elif theta == 180.0:
                    while True:
                        pix_y -= 1
                        if pix_y < 0:
                            pix_x = x
                            pix_y = y
                            break
                        if edges_frames_Pw[pix_x][pix_y] == 255: break

                elif theta == 270.0:
                    while True:
                        pix_x += 1
                        if pix_x >= len_x:
                            pix_x = x
                            pix_y = y
                            break
                        if edges_frames_Pw[pix_x][pix_y] == 255: break
                else:
                    a = 1
                    radian = math.radians(theta)
                    while True:
                        pix_x = x - round(a * math.sin(radian))
                        pix_y = y + round(a * math.cos(radian))
                        if pix_x < 0 or pix_y < 0 or pix_x >= len_x or pix_y >= len_y:
                            pix_x = x
                            pix_y = y
                            break
                        if edges_frames_Pw[pix_x][pix_y] == 255: break

                        if 0 < theta < 90:
                            if pix_y + 1 < len_y and edges_frames_Pw[pix_x][pix_y + 1] == 255:
                                pix_y += 1
                                break
                            if pix_x - 1 >= 0 and edges_frames_Pw[pix_x - 1][pix_y] == 255:
                                pix_x -= 1
                                break

                        elif 90 < theta < 180:
                            if pix_y - 1 >= 0 and edges_frames_Pw[pix_x][pix_y - 1] == 255:
                                pix_y -= 1
                                break
                            if pix_x - 1 >= 0 and edges_frames_Pw[pix_x - 1][pix_y] == 255:
                                pix_x -= 1
                                break

                        elif 180 < theta < 270:
                            if pix_y - 1 >= 0 and edges_frames_Pw[pix_x][pix_y - 1] == 255:
                                pix_y -= 1
                                break
                            if pix_x + 1 < len_x and edges_frames_Pw[pix_x + 1][pix_y] == 255:
                                pix_x += 1
                                break

                        elif 270 < theta < 360:
                            if pix_y + 1 < len_y and edges_frames_Pw[pix_x][pix_y + 1] == 255:
                                pix_y += 1
                                break
                            if pix_x + 1 < len_x and edges_frames_Pw[pix_x + 1][pix_y] == 255:
                                pix_x += 1
                                break
                        a += 1

                dist += math.sqrt((y - pix_y) ** 2 + (x - pix_x) ** 2)
                theta += 180

            # 균열의 폭을 저장
            crack_width_list.append(dist)

        for i in range(0, 8):
            next_x = x + dx_bfs[i]
            next_y = y + dy_bfs[i]

            if next_x < 0 or next_y < 0 or next_x >= len_x or next_y >= len_y: continue
            if visit[next_x][next_y] == 0:
                q.put([next_x, next_y])
                visit[next_x][next_y] = 1

    crack_width_list.sort(reverse=True)

    # 실제의 길이로 변환
    if len(crack_width_list) == 0:
        real_width = 0
    elif len(crack_width_list) < 10:
        real_width = round(crack_width_list[len(crack_width_list) - 1] * 0.92, 2)
    else:
        real_width = round(crack_width_list[9] * 0.92, 2)

    return real_width


def categorize_risk(real_width):
    if real_width >= 0.3:
        return 2
    elif 0.3 > real_width >= 0.2:
        return 1
    else:
        return 0


# result 값으로는 균열 하나에 대한 boxes 값이 들어온다.
def crop_crack_region(frame, result):  # 균열이 검출된 영역을 잘라내는 함수
    cropped_images = []
    if result is None:  # 예외 처리
        return None
    box = result.xyxy
    label = "Crack"
    conf = result.conf
    x1, y1, x2, y2 = map(int, box[0])
    cropped_image = frame[y1:y2, x1:x2]
    cropped_images.append((cropped_image, label, conf))
    return cropped_images





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
