import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import numpy as np
from sklearn.cluster import KMeans
from color_distinguish import ColorDistinguish
import supervision as sv

# track_id : 팀 저장
player_team = {}
# YOLO 모델 로드
model = YOLO("yolo11n.pt")

# 동영상 파일 경로
video_path = ".\\Match_Video\\World Cup - 2022. Spain - Costa-Rica. Tactical cam.mp4"


def parse_user_color(value):
    return ColorDistinguish.parse_color(value)


def prompt_for_team_colors():
    print("\n[Team color input]")
    print("Enter Team A color (examples: red, blue, white, #FF0000, 255,0,0):")
    team_a = input().strip()
    print("Enter Team B color (examples: red, blue, white, #FF0000, 255,0,0):")
    team_b = input().strip()

    colors = {}
    parsed_a = parse_user_color(team_a)
    parsed_b = parse_user_color(team_b)
    if parsed_a is not None:
        colors["A"] = parsed_a
    if parsed_b is not None:
        colors["B"] = parsed_b
    return colors


initial_team_colors = prompt_for_team_colors()

# 동영상 캡처 객체 생성
video_capture = cv2.VideoCapture(video_path)

window_name = "SmartSportVision"

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 추적 ID별로 위치 기록을 저장할 딕셔너리
track_history = defaultdict(list)
ball_last_position = None
ball_track_id = None
total_distance = defaultdict(float)


class TeamColorTracker:
    """ByteTrack로 추적되는 선수의 상체 색상을 누적해 두 팀으로 분류한다."""

    color_helper = ColorDistinguish()

    def __init__(self, sample_window=12, refresh_every=10, min_pixels=70, initial_team_colors=None, debug=False):
        self.player_states = {}
        self.team_profiles = {}
        self.sample_window = sample_window
        self.refresh_every = refresh_every
        self.min_pixels = min_pixels
        self.frame_idx = 0
        self.kmeans_refresh_every = 20
        self.team_split_threshold = 0.10
        # Raised threshold to allow more tolerant matching for team colors (e.g., faded/variant reds)
        self.unknown_threshold = 0.36
        # If distances to A/B are too close (< ambiguous_margin), treat as UNKNOWN
        self.ambiguous_margin = 0.06
        self.initial_team_colors = initial_team_colors or {}
        self.debug = debug

    def _normalize_lab(self, lab_color):
        return np.asarray(lab_color, dtype=np.float32) / 255.0

    def _lab_distance(self, a, b):
        return self.color_helper.lab_distance(a, b)

    def _lab_to_bgr(self, lab_color):
        return self.color_helper.lab_to_bgr(lab_color)

    def _extract_uniform_lab(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        person = frame[y1:y2, x1:x2]
        if person.size == 0:
            return None

        height = person.shape[0]
        upper_body = person[0:max(8, int(height * 0.55)), :]
        if upper_body.size == 0:
            return None

        smooth = cv2.GaussianBlur(upper_body, (5, 5), 0)
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        neutral_mask = (v > 120) & (s < 70)
        colorful_mask = (s > 18) & (v > 18) & (v < 240) & ~((h >= 35) & (h <= 85))
        mask = neutral_mask | colorful_mask

        if mask.sum() < self.min_pixels:
            mask = (v > 12) & (v < 245)

        pixels = lab[mask]
        if len(pixels) < self.min_pixels:
            pixels = lab.reshape(-1, 3)

        if len(pixels) < 24:
            return None

        if len(pixels) > 4000:
            idx = np.random.choice(len(pixels), 4000, replace=False)
            pixels = pixels[idx]

        pixels = pixels / 255.0
        chroma = np.sqrt(pixels[:, 1] ** 2 + pixels[:, 2] ** 2)
        luma = pixels[:, 0]

        if np.median(chroma) < 0.06 and np.median(luma) > 0.70:
            return np.array([np.median(luma), 0.0, 0.0], dtype=np.float32)

        if len(pixels) >= 3:
            kmeans = KMeans(n_clusters=min(3, max(2, len(pixels) // 700)), random_state=0, n_init=10)
            kmeans.fit(pixels)
            counts = np.bincount(kmeans.labels_)
            best_idx = int(np.argmax(counts))
            color_lab = kmeans.cluster_centers_[best_idx]
        else:
            color_lab = np.median(pixels, axis=0)

        return np.asarray(color_lab, dtype=np.float32)

    def update_player(self, frame, bbox, track_id):
        state = self.player_states.setdefault(
            track_id,
            {
                "samples": deque(maxlen=self.sample_window),
                "team": None,
                "rep_lab": None,
                "rep_bgr": (255, 255, 255),
                "last_refresh": -1,
            },
        )

        color_lab = self._extract_uniform_lab(frame, bbox)
        if color_lab is not None:
            state["samples"].append(color_lab)

        if len(state["samples"]) >= 3:
            rep_lab = np.median(np.asarray(list(state["samples"]), dtype=np.float32), axis=0)
        elif color_lab is not None:
            rep_lab = color_lab
        else:
            rep_lab = np.asarray([0.5, 0.0, 0.0], dtype=np.float32)

        state["rep_lab"] = rep_lab
        state["rep_bgr"] = self._lab_to_bgr(rep_lab)

        team = self._classify_player(track_id)
        state["team"] = team
        state["last_refresh"] = self.frame_idx

        if team in ("A", "B"):
            self._update_team_centroid(team, state["rep_lab"])
            self._calibrate_team_profile(team, state["rep_lab"])

        if self.frame_idx % self.refresh_every == 0:
            self._refresh_team_profiles()

        return state["rep_bgr"], state["team"]

    def _seed_team_profiles(self):
        if self.team_profiles:
            return
        if "A" in self.initial_team_colors:
            self.team_profiles["A"] = {
                "centroid_lab": self.initial_team_colors["A"].copy(),
                "member_count": 1,
            }
        if "B" in self.initial_team_colors:
            self.team_profiles["B"] = {
                "centroid_lab": self.initial_team_colors["B"].copy(),
                "member_count": 1,
            }

    def _calibrate_team_profile(self, team_label, observed_color):
        if team_label not in self.team_profiles:
            return

        profile = self.team_profiles[team_label]
        if team_label in self.initial_team_colors and self.initial_team_colors[team_label] is not None:
            user_color = np.asarray(self.initial_team_colors[team_label], dtype=np.float32)
            dist_to_user = self._lab_distance(observed_color, user_color)
            dist_to_profile = self._lab_distance(observed_color, profile["centroid_lab"])

            if dist_to_user < 0.08 and dist_to_profile < 0.08:
                profile["centroid_lab"] = 0.7 * profile["centroid_lab"] + 0.3 * observed_color
            elif dist_to_user < 0.12:
                profile["centroid_lab"] = 0.6 * profile["centroid_lab"] + 0.4 * observed_color
            elif dist_to_profile < 0.08:
                profile["centroid_lab"] = 0.5 * profile["centroid_lab"] + 0.5 * observed_color
            else:
                profile["centroid_lab"] = 0.7 * profile["centroid_lab"] + 0.3 * observed_color

    def _classify_player(self, track_id):
        state = self.player_states[track_id]
        if state["rep_lab"] is None:
            return "UNKNOWN"

        self._seed_team_profiles()

        if not self.team_profiles:
            self.team_profiles["A"] = {
                "centroid_lab": state["rep_lab"].copy(),
                "member_count": 1,
            }
            return "A"

        if "A" in self.team_profiles and "B" not in self.team_profiles:
            dist_a = self._lab_distance(state["rep_lab"], self.team_profiles["A"]["centroid_lab"])
            if dist_a > self.team_split_threshold:
                self.team_profiles["B"] = {
                    "centroid_lab": state["rep_lab"].copy(),
                    "member_count": 1,
                }
                return "B"
            return "A"

        if "A" in self.team_profiles and "B" in self.team_profiles:
            dist_a = self._lab_distance(state["rep_lab"], self.team_profiles["A"]["centroid_lab"])
            dist_b = self._lab_distance(state["rep_lab"], self.team_profiles["B"]["centroid_lab"])

            # debug print
            if self.debug:
                try:
                    print(f"[DEBUG] id={track_id} rep_lab={state['rep_lab']} distA={dist_a:.3f} distB={dist_b:.3f}")
                except Exception:
                    pass

            # If too far from both profiles -> UNKNOWN
            if min(dist_a, dist_b) > self.unknown_threshold:
                return "UNKNOWN"

            # If distances are too close (ambiguous), mark UNKNOWN
            if abs(dist_a - dist_b) < self.ambiguous_margin:
                # If team initial color exists, enforce family check to avoid mislabeling
                if self.initial_team_colors:
                    family = self.color_helper.detect_family(state['rep_lab'])
                    # If family is neutral or doesn't match either initial team family, treat UNKNOWN
                    fam_a = self.color_helper.detect_family(self.team_profiles['A']['centroid_lab']) if 'A' in self.team_profiles else None
                    fam_b = self.color_helper.detect_family(self.team_profiles['B']['centroid_lab']) if 'B' in self.team_profiles else None
                    if family is None or (family != fam_a and family != fam_b):
                        return "UNKNOWN"
                else:
                    return "UNKNOWN"

            return "A" if dist_a < dist_b else "B"

        return "UNKNOWN"

    def _update_team_centroid(self, team_label, color_lab):
        if team_label not in self.team_profiles:
            self.team_profiles[team_label] = {"centroid_lab": color_lab.copy(), "member_count": 1}
            return

        # Prevent centroid drift: if user provided an initial color, skip updates for observations
        # that are far from the user color (likely non-uniform artifacts).
        if team_label in self.initial_team_colors and self.initial_team_colors[team_label] is not None:
            user_color = np.asarray(self.initial_team_colors[team_label], dtype=np.float32)
            dist_to_user = self._lab_distance(color_lab, user_color)
            if dist_to_user > 0.30:
                if self.debug:
                    try:
                        print(f"[DEBUG] skip centroid update for team {team_label}: dist_to_user={dist_to_user:.3f} > 0.30")
                    except Exception:
                        pass
                return

        profile = self.team_profiles[team_label]
        alpha = 0.18 if profile["member_count"] < 4 else 0.10
        profile["centroid_lab"] = (1.0 - alpha) * profile["centroid_lab"] + alpha * color_lab
        profile["member_count"] += 1

    def _refresh_team_profiles(self):
        if self.frame_idx % self.kmeans_refresh_every != 0:
            return

        team_colors = {team: [] for team in ("A", "B")}
        for state in self.player_states.values():
            if state["team"] in team_colors and state["rep_lab"] is not None:
                team_colors[state["team"]].append(state["rep_lab"])

        if len(team_colors["A"]) >= 2 and len(team_colors["B"]) >= 2:
            all_colors = np.asarray(team_colors["A"] + team_colors["B"], dtype=np.float32)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            kmeans.fit(all_colors)
            centers = kmeans.cluster_centers_

            if "A" in self.team_profiles:
                a_center = centers[np.argmin([self._lab_distance(center, self.team_profiles["A"]["centroid_lab"]) for center in centers])]
                self.team_profiles["A"]["centroid_lab"] = a_center
            if "B" in self.team_profiles:
                b_center = centers[np.argmin([self._lab_distance(center, self.team_profiles["B"]["centroid_lab"]) for center in centers])]
                self.team_profiles["B"]["centroid_lab"] = b_center


# track_id : 팀 저장
player_team = {}
# 추적 ID별로 위치 기록을 저장할 딕셔너리
track_history = defaultdict(list)
player_colors = {}
# 추적 ID별로 총 이동 거리를 저장할 딕셔너리
total_distance = defaultdict(float)

team_tracker = TeamColorTracker(sample_window=12, refresh_every=8, min_pixels=80, initial_team_colors=initial_team_colors, debug=True)
DEBUG_MAX_FRAMES = 220


while True:
    # 다음 프레임 읽기
    ret, frame = video_capture.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    team_tracker.frame_idx += 1

    # YOLO 추론
    # classes=[0, 32]는 사람과 공만 감지하도록 설정
    # persist=True는 추적 결과를 유지하도록 설정
    # tracker="bytetrack.yaml"는 ByteTrack 추적기를 사용하도록 설정
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0, 32], verbose=False)

    # 결과 그리기
    annotated_frame = frame.copy()
    player_boxes = []
    player_track_ids = []
    player_teams = []
    player_labels = []
    ball_boxes = []
    ball_track_ids = []
    ball_labels = []

    for result in results:
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if class_id == 32:
                ball_boxes.append([x1, y1, x2, y2])
                ball_track_ids.append(track_id)
                ball_labels.append(f"BALL {track_id}")
                ball_last_position = (cx, cy)
                ball_track_id = track_id
                continue

            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 100:
                track_history[track_id].pop(0)

            rep_bgr, team = team_tracker.update_player(frame, (x1, y1, x2, y2), track_id)
            player_team[track_id] = team
            player_colors[track_id] = rep_bgr

            player_boxes.append([x1, y1, x2, y2])
            player_track_ids.append(track_id)
            player_teams.append(team or "UNKNOWN")
            player_labels.append(f"ID:{track_id} {team or 'UNKNOWN'}")

            points = track_history[track_id]
            if len(points) > 1:
                prev_x, prev_y = points[-2]
                curr_x, curr_y = points[-1]
                distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                if distance > 2:
                    total_distance[track_id] += distance

    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT, thickness=2, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.DEFAULT, text_color=sv.Color.from_hex("#000000"))

    if player_boxes:
        player_detections = sv.Detections(
            xyxy=np.array(player_boxes, dtype=np.float32),
            class_id=np.zeros(len(player_boxes), dtype=np.int32),
            tracker_id=np.array(player_track_ids, dtype=np.int32),
            data={"team": np.array(player_teams, dtype=object)},
        )
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=player_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=player_detections, labels=player_labels)

    if ball_boxes:
        ball_detections = sv.Detections(
            xyxy=np.array(ball_boxes, dtype=np.float32),
            class_id=np.ones(len(ball_boxes), dtype=np.int32),
            tracker_id=np.array(ball_track_ids, dtype=np.int32),
            data={"kind": np.array(["ball"] * len(ball_boxes), dtype=object)},
        )
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=ball_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=ball_detections, labels=ball_labels)

    for track_id, points in list(track_history.items()):
        if len(points) > 1:
            trace_xy = np.array(points, dtype=np.float32)
            if len(trace_xy) >= 2:
                for j in range(1, len(trace_xy)):
                    x1, y1 = map(int, trace_xy[j - 1])
                    x2, y2 = map(int, trace_xy[j])
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 221, 87), 2)

    if ball_last_position is not None:
        cv2.circle(annotated_frame, ball_last_position, 6, (0, 0, 255), -1)
        cv2.putText(
            annotated_frame,
            "BALL",
            (ball_last_position[0] + 8, ball_last_position[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 화면 출력
    cv2.imshow(window_name, annotated_frame)

    # debug mode: optionally stop after a limited number of frames for log capture
    if team_tracker.debug and team_tracker.frame_idx >= DEBUG_MAX_FRAMES:
        print(f"[DEBUG] reached DEBUG_MAX_FRAMES={DEBUG_MAX_FRAMES}, stopping loop")
        break
    # ESC키를 누르면 종료
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC key
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()