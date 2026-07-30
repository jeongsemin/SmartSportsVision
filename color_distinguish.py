import cv2
import numpy as np


class ColorDistinguish:
    """LAB 기반 색상 분류 유틸리티.

    이 모듈은 팀 유니폼 색상을 세분화해 분류하는 데 사용한다.
    - 팀 색상(A/B)와 Unknown을 구분
    - 중성색, 밝은색, 어두운색, 강한색, 피드백 색상 등 세부 분류
    - 사용자 입력 색상과 비교할 때 재사용 가능
    """

    def __init__(self):
        self.color_names = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "orange": (0, 165, 255),
            "purple": (128, 0, 128),
            "pink": (203, 192, 255),
        }

    @staticmethod
    def bgr_to_lab(bgr):
        bgr = np.asarray(bgr, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
        return lab.astype(np.float32) / 255.0

    @staticmethod
    def lab_to_bgr(lab):
        lab = np.asarray(lab, dtype=np.float32)
        lab = np.clip(lab, 0.0, 1.0)
        lab_img = np.uint8(lab * 255.0)
        bgr = cv2.cvtColor(lab_img.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
        return tuple(int(v) for v in bgr)

    @staticmethod
    def lab_distance(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        delta = a - b
        return float(np.sqrt((delta[0] * 1.2) ** 2 + (delta[1] * 1.0) ** 2 + (delta[2] * 1.0) ** 2))

    @staticmethod
    def classify_lab_color(lab_color, team_a_lab=None, team_b_lab=None, unknown_threshold=0.14):
        color = np.asarray(lab_color, dtype=np.float32)
        if team_a_lab is None and team_b_lab is None:
            return "UNKNOWN"

        a = float(color[1] - 0.5)
        b = float(color[2] - 0.5)
        chroma = float(np.sqrt(a * a + b * b))
        luma = float(color[0])

        is_neutral = chroma < 0.06 and luma > 0.70
        is_extreme = chroma > 0.23 and luma < 0.30
        is_goalkeeper_like = (chroma < 0.08 and luma < 0.55) or (chroma < 0.04 and luma > 0.70)

        if team_a_lab is not None and team_b_lab is not None:
            dist_a = ColorDistinguish.lab_distance(color, team_a_lab)
            dist_b = ColorDistinguish.lab_distance(color, team_b_lab)
            if min(dist_a, dist_b) > unknown_threshold:
                return "UNKNOWN"
            return "A" if dist_a <= dist_b else "B"

        if team_a_lab is not None:
            dist_a = ColorDistinguish.lab_distance(color, team_a_lab)
            if dist_a <= unknown_threshold:
                return "A"
            return "UNKNOWN"

        if team_b_lab is not None:
            dist_b = ColorDistinguish.lab_distance(color, team_b_lab)
            if dist_b <= unknown_threshold:
                return "B"
            return "UNKNOWN"

        if is_neutral or is_extreme or is_goalkeeper_like:
            return "UNKNOWN"

        return "UNKNOWN"

    @staticmethod
    def describe_color(lab_color):
        color = np.asarray(lab_color, dtype=np.float32)
        chroma = float(np.sqrt(color[1] ** 2 + color[2] ** 2))
        luma = float(color[0])

        if chroma < 0.05 and luma > 0.72:
            return "neutral_white"
        if chroma < 0.05:
            return "neutral_gray"
        if luma < 0.28:
            return "dark"
        if luma > 0.82:
            return "bright"
        if chroma > 0.16:
            return "saturated"
        return "normal"

    @staticmethod
    def parse_color(value):
        if value is None:
            return None

        text = value.strip().lower()
        if not text:
            return None

        if text in {"red", "blue", "green", "yellow", "cyan", "magenta", "white", "black", "orange", "purple", "pink"}:
            bgr = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0), "yellow": (0, 255, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255), "white": (255, 255, 255), "black": (0, 0, 0), "orange": (0, 165, 255), "purple": (128, 0, 128), "pink": (203, 192, 255)}[text]
            return ColorDistinguish.bgr_to_lab(bgr)

        if text.startswith("#") and len(text) == 7:
            hex_value = text[1:]
            try:
                bgr = tuple(int(hex_value[i:i + 2], 16) for i in (4, 2, 0))
                return ColorDistinguish.bgr_to_lab(bgr)
            except ValueError:
                return None

        if "," in text:
            try:
                parts = [int(p.strip()) for p in text.split(",")]
                if len(parts) == 3:
                    return ColorDistinguish.bgr_to_lab(tuple(parts))
            except ValueError:
                return None

        return None
