import cv2
from ultralytics import YOLO
from collections import defaultdict
import math
import numpy as np
from sklearn.cluster import KMeans

# track_id : 팀 저장
player_team = {}
# YOLO 모델 로드
model = YOLO("yolo11s.pt")

# 동영상 파일 경로
video_path = ".\\Match_Video\\World Cup - 2022. Spain - Costa-Rica. Tactical cam.mp4"

#동영상 캡처 객체 생성
video_capture = cv2.VideoCapture(video_path)

window_name = "SmartSportVision"

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 추적 ID별로 위치 기록을 저장할 딕셔너리
track_history = defaultdict(list)
player_colors = {}
# 추적 ID별로 총 이동 거리를 저장할 딕셔너리
total_distance = defaultdict(float)

def color_to_team(color):
    if color is None:
        return "UNKNOWN"

    # BGR
    b, g, r = color
    # 빨간색 유니폼
    if (
        r > g * 1.3
        and r > b * 1.3
    ):
        return "A"

    # 흰색 유니폼
    if (
        r > 150
        and g > 150
        and b > 150
        and abs(r-g) < 40
        and abs(g-b) < 40
    ):
        return "B"

    return "UNKNOWN"



def get_team(track_id, frame, bbox):
    # 이미 분석한 선수
    if track_id in player_team:
        return player_team[track_id]

    # 처음 보는 선수
    color = get_uniform_color(
        frame,
        bbox
    )
    team = color_to_team(color)

    # 저장
    player_team[track_id] = team

    print(
        f"ID {track_id} -> Team {team}"
    )

    return team

def get_uniform_color(frame, bbox):
    x1, y1, x2, y2 = bbox
    person = frame[y1:y2, x1:x2]

    if person.size == 0:
        return (255,255,255)

    height = person.shape[0]
    # 상체 영역
    jersey = person[0:int(height*0.55), :]

    # HSV 변환
    hsv = cv2.cvtColor(
        jersey,
        cv2.COLOR_BGR2HSV
    )

    # 채도 높은 영역만 사용
    mask = (
        (hsv[:,:,1] > 50)
        |
        (hsv[:,:,2] > 180)
    )
    pixels = jersey[mask]
    if len(pixels) < 50:
        return (255,255,255)

    pixels = pixels.reshape(-1,3)
    kmeans = KMeans(
        n_clusters=3,
        random_state=0,
        n_init=10
    )

    kmeans.fit(pixels)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    dominant_color = (
        kmeans.cluster_centers_[
            np.argmax(counts)
        ]
    )

    return tuple(
        map(int, dominant_color)
    )
    
while True:
    # 다음 프레임 읽기
    ret, frame = video_capture.read()
    # 다음 프레임 읽기
    if not ret:
        print("End of video or cannot read the frame.")
        break
    
    # YOLO 추론
    # classes=[0, 32]는 사람과 공만 감지하도록 설정
    # persist=True는 추적 결과를 유지하도록 설정
    # tracker="bytetrack.yaml"는 ByteTrack 추적기를 사용하도록 설정
    results = model.track(frame, persist=True, tracker = "bytetrack.yaml", classes=[0], verbose=False)
            
    # 결과 그리기
    annotated_frame = frame.copy()
    
    for result in results:
        if result.boxes is None:
            continue
        
        if result.boxes.id is None:
            continue
        
        boxes = result.boxes.xyxy
        ids = result.boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, ids):
            
            x1, y1, x2, y2 = map(int, box)
            
            if track_id not in player_colors:
                color = get_uniform_color(
                    frame,
                    (x1,y1,x2,y2)
                )

                player_colors[track_id] = color
            
            cx = (x1 + x2) // 2
            cy = y2
            
            track_history[track_id].append((cx, cy))

            if len(track_history[track_id]) > 100:
                track_history[track_id].pop(0)
            
            team = get_team(
                track_id,
                frame,
                (x1, y1, x2, y2)
            )

            # 테두리 그리기
            # annotated_frame = 프레임
            # cx,cy = 중심 좌표
            # 10 = 원의 반지름
            # (0, 255, 0) = 초록색
            # 2 = 선의 두께
            cv2.circle(annotated_frame, (cx, cy), 10, player_colors[track_id], 2)
            
            cv2.putText(annotated_frame, f"ID: {track_id} TEAM:{team}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, player_colors[track_id], 2)

            # 이동 거리 계산을 위한 points 가져오기
            points = track_history[track_id]

            # 이동 거리 계산
            if len(points) > 1:
                prev_x, prev_y = points[-2]
                curr_x, curr_y = points[-1]
                distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                
                # 2픽셀 미만은 노이즈로 간주
                if distance > 2:
                    total_distance[track_id] += distance
                
    # 화면 출력
    cv2.imshow(window_name, annotated_frame)
    
    # ESC키를 누르면 종료
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC key
        break
    # 창의 X 버튼을 누르면 종료
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    
video_capture.release()
cv2.destroyAllWindows()