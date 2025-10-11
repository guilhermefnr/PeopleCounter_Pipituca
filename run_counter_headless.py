import os
import json
import cv2
import numpy as np
from collections import OrderedDict, deque
from ultralytics import YOLO
import torch
import math
import time
import datetime as dt
import signal
import sys
from typing import Optional

# Flag global para shutdown gracioso
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n⚠ Sinal {signum} recebido - iniciando shutdown gracioso...")
    shutdown_requested = True

# Registra handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ======================================================================
# CONTADOR DE PESSOAS - VERSÃO HEADLESS COM SALVAMENTO PERIÓDICO
# ======================================================================

W, H = 640, 480
FPS_TARGET = 15

# ---------- Detecção ----------
CONFIDENCE_THRESHOLD = 0.15
NMS_IOU = 0.45
MIN_DETECTION_SIZE = 15
MAX_DETECTION_SIZE = 280

# ---------- Tracking ROBUSTO ----------
MAX_TRACKING_DISTANCE = 120
MIN_TRACK_LENGTH = 4
MAX_LOST_FRAMES = 45
SMOOTHING_FACTOR = 0.30
MAX_PREDICT_STEP = 25

# ---------- Validação Centro/Bbox ----------
MAX_CENTER_BBOX_DISTANCE = 50
CENTER_BBOX_ALERT_DISTANCE = 35
CENTER_RESET_THRESHOLD = 70

# ---------- Gate ----------
GATE_WIDTH = 120
CROSSING_CONFIRMATION_FRAMES = 1

# ---------- Associação ----------
ASSOCIATION_COST_THRESHOLD = 220
IOU_WEIGHT = 0.55
LOW_IOU_PENALTY_THRESH = 0.08
LOW_IOU_PENALTY_FACTOR = 1.15

# ---------- Entrada pela DIREITA ----------
QUICK_BIRTH_ZONE_RIGHT = 140
QUICK_BIRTH_MIN_FRAMES = 1
QUICK_BIRTH_MIN_SIZE = 28

# ---------- Janela de graça ----------
RIGHT_EDGE_GRACE_FRAMES = 15
EARLY_CROSS_CONFIRM_FRAMES = 1
FORCE_INITIAL_SIDE_ON_QUICK_BIRTH = True
MIN_MOVE_AFTER_BIRTH = 4

# ---------- Cruzamento ----------
COOLDOWN_AFTER_CROSS = 15
MIN_DISTANCE_FROM_LINE = 10
MIN_MOVE_IN_GATE = 2
GATE_TIMEOUT = 45

# ---------- Nascimento ----------
BIRTH_MIN_FRAMES = 2
MERGE_RADIUS = 45
BIRTH_MAX_IOU = 0.22

# ---------- Zonas ----------
IGNORE_ZONES = []
RIGHT_EDGE_IGNORE = 0

# ---------- UI ----------
PANEL_ANCHOR = "bl"

# ---------- RTSP / Modelo ----------
RTSP_URL_DEFAULT = "rtsp://admin:111229@192.168.1.2:554/cam/realmonitor?channel=1&subtype=1"
RTSP_URL = os.getenv("RTSP_URL", RTSP_URL_DEFAULT)
MODEL_CANDIDATES = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "line_config.json")

# Headless flags
HEADLESS = os.getenv("HEADLESS", "0") == "1"
HEADLESS_MAX_SEC = int(os.getenv("HEADLESS_MAX_SEC", "3600"))

def _now_gmt3():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=3)

def _within_run_window_gmt3(now=None):
    if now is None:
        now = _now_gmt3()
    return 9 <= now.hour <= 21

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;10000000|max_delay;0|buffer_size;512000|"
    "rtsp_flags;prefer_tcp|fflags;nobuffer"
)


class KalmanTracker:
    def __init__(self, initial_pos):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = 0.05 * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = 0.10 * np.eye(2, dtype=np.float32)
        self.kf.statePre = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()

    def predict(self):
        p = self.kf.predict()
        return (int(p[0]), int(p[1]))

    def update(self, measurement):
        self.kf.correct(np.array([[measurement[0]], [measurement[1]]], dtype=np.float32))
        return (int(self.kf.statePost[0]), int(self.kf.statePost[1]))


class PersonTrack:
    def __init__(self, track_id, detection_box, initial_side, quick_birth=False):
        self.id = track_id
        self.bbox_history = deque([detection_box], maxlen=10)
        self.center_history = deque(maxlen=15)

        x, y, w, h = detection_box
        center = (x + w // 2, y + h // 2)
        self.center_history.append(center)

        self.kalman = KalmanTracker(center)

        self.last_seen_frame = 0
        self.lost_frames = 0
        self.confidence_scores = deque([1.0], maxlen=5)

        self.initial_side = initial_side
        self.current_side = initial_side
        self.crossing_confirmations = 0
        self.last_cross_frame = -999
        self.total_crosses = 0
        self.in_gate_area = False

        self.first_in_gate_frame = None
        self.cooldown = 0

        self.center_bbox_anomaly_count = 0
        self.is_anomalous = False

        self.quick_birth = quick_birth

        self.track_quality = 1.0
        self.direction_consistency = 0.0

        self.birth_frame = 0
        self.edge_grace_until = -1
        self.early_cross_used = False

        self.frames_in_gate = 0
        self.frames_outside_gate = 0
        
        self.last_crossing_type = None
        self.has_entrada = False
        self.has_saida = False

    def update_detection(self, bbox, confidence, frame_num):
        self.bbox_history.append(bbox)
        self.lost_frames = 0
        self.last_seen_frame = frame_num
        self.confidence_scores.append(confidence)

        x, y, w, h = bbox
        center = (x + w // 2, y + h // 2)

        self.kalman.predict()
        smoothed = self.kalman.update(center)

        final_center = (
            int(center[0] * (1 - SMOOTHING_FACTOR) + smoothed[0] * SMOOTHING_FACTOR),
            int(center[1] * (1 - SMOOTHING_FACTOR) + smoothed[1] * SMOOTHING_FACTOR)
        )

        bbox_center = (x + w // 2, y + h // 2)
        dist_to_bbox = math.hypot(final_center[0] - bbox_center[0],
                                  final_center[1] - bbox_center[1])

        if dist_to_bbox > CENTER_RESET_THRESHOLD:
            final_center = bbox_center
            self.kalman = KalmanTracker(bbox_center)
            self.center_bbox_anomaly_count += 3
            self.is_anomalous = True
        elif dist_to_bbox > MAX_CENTER_BBOX_DISTANCE:
            final_center = bbox_center
            self.center_bbox_anomaly_count += 2
            self.is_anomalous = True
        elif dist_to_bbox > CENTER_BBOX_ALERT_DISTANCE:
            self.center_bbox_anomaly_count += 1
        else:
            if self.center_bbox_anomaly_count > 0:
                self.center_bbox_anomaly_count = max(0, self.center_bbox_anomaly_count - 2)
            if self.center_bbox_anomaly_count == 0:
                self.is_anomalous = False

        self.center_history.append(final_center)
        self._update_track_quality()

    def predict_next_position(self):
        self.lost_frames += 1

        if self.lost_frames > 5:
            if self.center_history:
                self.center_history.append(self.center_history[-1])
            return self.center_history[-1] if self.center_history else (0, 0)

        p = self.kalman.predict()

        if self.center_history:
            lx, ly = self.center_history[-1]
            dx, dy = int(p[0]) - lx, int(p[1]) - ly
            norm = math.hypot(dx, dy)
            if norm > MAX_PREDICT_STEP and norm > 0:
                scale = MAX_PREDICT_STEP / norm
                p = (lx + int(dx * scale), ly + int(dy * scale))

            p = (max(0, min(W, int(p[0]))), max(0, min(H, int(p[1]))))

        self.center_history.append((int(p[0]), int(p[1])))
        return p

    def get_current_center(self):
        return self.center_history[-1] if self.center_history else None

    def get_current_bbox(self):
        return self.bbox_history[-1] if self.bbox_history else None

    def _update_track_quality(self):
        if len(self.center_history) < 3:
            return
        recent = list(self.center_history)[-5:]
        if len(recent) >= 3:
            velocities = []
            for i in range(1, len(recent)):
                dx = recent[i][0] - recent[i - 1][0]
                dy = recent[i][1] - recent[i - 1][1]
                velocities.append((dx, dy))
            if velocities:
                avg = np.mean(velocities, axis=0)
                c = 0
                for v in velocities:
                    dot = v[0] * avg[0] + v[1] * avg[1]
                    norms = np.linalg.norm(v) * np.linalg.norm(avg)
                    if norms > 0:
                        c += dot / norms
                self.direction_consistency = c / len(velocities)

        avg_conf = np.mean(list(self.confidence_scores))
        anomaly_penalty = max(0, 1 - self.center_bbox_anomaly_count * 0.15)

        self.track_quality = (avg_conf * 0.6 +
                              (self.direction_consistency + 1) * 0.15 +
                              min(len(self.center_history) / 10, 1) * 0.15 +
                              anomaly_penalty * 0.1)

    def is_reliable(self):
        if self.is_anomalous and self.center_bbox_anomaly_count > 2:
            return False

        if self.quick_birth:
            return (
                len(self.center_history) >= 2 and
                self.lost_frames < MAX_LOST_FRAMES and
                (self.track_quality > 0.35 or True)
            )

        return (len(self.center_history) >= MIN_TRACK_LENGTH and
                self.track_quality > 0.50 and
                self.lost_frames < MAX_LOST_FRAMES)

    def can_cross_again(self, current_frame):
        return (current_frame - self.last_cross_frame) >= COOLDOWN_AFTER_CROSS


class BirthBuffer:
    def __init__(self):
        self.items = []

    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _iou(self, b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections, frame_num):
        self.items = [it for it in self.items if frame_num - it['last_frame'] <= 6]
        for det in detections:
            c = det['center']
            b = det['bbox']
            matched = False
            for it in self.items:
                if self._dist(c, it['center']) <= 30 and self._iou(b, it['bbox']) >= 0.15:
                    it['center'] = c
                    it['bbox'] = b
                    it['hits'] += 1
                    it['last_frame'] = frame_num
                    it['quick_birth'] = det.get('quick_birth', False)
                    matched = True
                    break
            if not matched:
                self.items.append({
                    'center': c,
                    'bbox': b,
                    'hits': 1,
                    'last_frame': frame_num,
                    'quick_birth': det.get('quick_birth', False)
                })

    def ready_candidates(self):
        ready = []
        for it in self.items:
            if it['quick_birth'] and it['hits'] >= QUICK_BIRTH_MIN_FRAMES:
                ready.append(it)
            elif not it['quick_birth'] and it['hits'] >= BIRTH_MIN_FRAMES:
                ready.append(it)
        return ready


class PeopleCounter:
    def __init__(self):
        self.model = None
        self.line_config = None
        self.tracks = OrderedDict()
        self.next_track_id = 1
        self.frame_count = 0
        self.total_crossings = 0
        
        self.entrada_count = 0
        self.saida_count = 0

        self.fps_counter = deque(maxlen=30)
        self.detection_history = deque(maxlen=100)

        self.gate_width = GATE_WIDTH
        self.entry_sign = 1

        self.birth_buffer = BirthBuffer()

        self.anomaly_detections = 0
        self.quick_births = 0

        self._load_yolo_model()

    def _load_yolo_model(self):
        try:
            for model_name in MODEL_CANDIDATES:
                try:
                    print(f"Carregando modelo {model_name}...")
                    self.model = YOLO(model_name)
                    if torch.cuda.is_available():
                        self.model.to('cuda')
                        print("CUDA ativado")
                    else:
                        print("Usando CPU")
                    print(f"Modelo {model_name} carregado com sucesso")
                    break
                except Exception as e:
                    print(f"Falha ao carregar {model_name}: {e}")
            if self.model is None:
                raise Exception("Nenhum modelo YOLO encontrado")
        except Exception as e:
            print(f"Erro ao carregar YOLO: {e}")
            raise

    def load_line_config(self):
        try:
            with open(CONFIG_FILE, "r") as f:
                self.line_config = json.load(f)

            p1 = tuple(self.line_config["line_start"])
            p2 = tuple(self.line_config["line_end"])
            line_type = self.line_config.get("line_type", "horizontal")

            cfg_gate = int(self.line_config.get("gate_width", GATE_WIDTH))
            self.gate_width = max(GATE_WIDTH, cfg_gate)

            entry_side = self.line_config.get("entry_side", None)
            if entry_side is None:
                entry_side = "right" if line_type == "vertical" else "top"
            self.entry_sign = self._compute_entry_sign(entry_side, p1, p2, line_type)

            print(f"Linha: {p1} -> {p2} | Tipo: {line_type} | Gate: ±{self.gate_width}px")
            print(f"Lado de entrada: {entry_side} | Sinal: {self.entry_sign}")
            return True
        except FileNotFoundError:
            print(f"Arquivo {CONFIG_FILE} não encontrado")
            return False

    def _compute_entry_sign(self, entry_side, p1, p2, line_type):
        x1, y1 = p1
        x2, y2 = p2
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if line_type == "vertical":
            line_x = (x1 + x2) // 2
            test_pt = (line_x + 1, my) if entry_side == "right" else (line_x - 1, my)
        else:
            line_y = (y1 + y2) // 2
            test_pt = (mx, line_y - 1) if entry_side == "top" else (mx, line_y + 1)
        return 1 if self.side_of_line(test_pt, p1, p2) >= 0 else -1

    def detect_people(self, frame):
        results = self.model(frame,
                             conf=CONFIDENCE_THRESHOLD,
                             iou=NMS_IOU,
                             classes=[0],
                             verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                    if (MIN_DETECTION_SIZE <= w <= MAX_DETECTION_SIZE and
                        MIN_DETECTION_SIZE <= h <= MAX_DETECTION_SIZE):

                        cx, cy = x + w // 2, y + h // 2

                        ignore = False
                        for zx, zy, zw, zh in IGNORE_ZONES:
                            if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
                                ignore = True
                                break
                        if ignore:
                            continue

                        if RIGHT_EDGE_IGNORE > 0 and cx >= W - RIGHT_EDGE_IGNORE:
                            continue

                        quick_birth = (cx >= W - QUICK_BIRTH_ZONE_RIGHT and
                                       w >= QUICK_BIRTH_MIN_SIZE and
                                       h >= QUICK_BIRTH_MIN_SIZE)

                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'center': (cx, cy),
                            'quick_birth': quick_birth
                        })
        return detections

    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    def associate_detections_to_tracks(self, detections):
        if not detections or not self.tracks:
            return [], list(range(len(detections)))

        cost_matrix = np.full((len(self.tracks), len(detections)), 1000.0)
        track_ids = list(self.tracks.keys())

        for i, tid in enumerate(track_ids):
            t = self.tracks[tid]
            tc = t.get_current_center()
            tb = t.get_current_bbox()
            if tc is None or tb is None:
                continue
            for j, det in enumerate(detections):
                dc = det['center']
                db = det['bbox']
                distance = math.hypot(tc[0] - dc[0], tc[1] - dc[1])

                if distance < MAX_TRACKING_DISTANCE:
                    iou = self.calculate_iou(tb, db)
                    cost = distance * (1 - iou * IOU_WEIGHT)

                    if iou < LOW_IOU_PENALTY_THRESH:
                        cost *= LOW_IOU_PENALTY_FACTOR

                    cost_matrix[i, j] = cost

        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))

        for _ in range(min(len(track_ids), len(detections))):
            min_cost = np.inf
            bi = -1
            bj = -1
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    c = cost_matrix[i, j]
                    if c < min_cost:
                        min_cost = c
                        bi = i
                        bj = j
            if min_cost < ASSOCIATION_COST_THRESHOLD:
                matched_pairs.append((track_ids[bi], bj))
                unmatched_tracks.remove(bi)
                unmatched_dets.remove(bj)
            else:
                break
        return matched_pairs, unmatched_dets

    def side_of_line(self, point, p1, p2):
        x, y = point
        x1, y1 = p1
        x2, y2 = p2
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    def _projection_param(self, point, p1, p2):
        x0, y0 = point
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = (x2 - x1), (y2 - y1)
        denom = dx * dx + dy * dy
        if denom == 0:
            return 0.5
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / denom
        return t

    def perp_distance_to_line(self, point, p1, p2):
        x0, y0 = point
        x1, y1 = p1
        x2, y2 = p2
        denom = math.hypot(x2 - x1, y2 - y1)
        if denom == 0:
            return 1e9
        return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denom

    def point_in_gate(self, point, p1, p2, line_type):
        t = self._projection_param(point, p1, p2)
        if t < 0.0 or t > 1.0:
            return False
        return self.perp_distance_to_line(point, p1, p2) <= self.gate_width

    def update_tracks(self, detections):
        p1 = tuple(self.line_config["line_start"])
        p2 = tuple(self.line_config["line_end"])
        line_type = self.line_config.get("line_type", "horizontal")

        matched_pairs, unmatched_detections = self.associate_detections_to_tracks(detections)

        for track_id, det_idx in matched_pairs:
            det = detections[det_idx]
            t = self.tracks[track_id]
            t.update_detection(det['bbox'], det['confidence'], self.frame_count)

        for track_id, t in list(self.tracks.items()):
            if track_id not in [m[0] for m in matched_pairs]:
                if t.lost_frames < MAX_LOST_FRAMES:
                    t.predict_next_position()
                else:
                    if t.is_anomalous:
                        self.anomaly_detections += 1
                    del self.tracks[track_id]

        unmatched_dets = [detections[i] for i in unmatched_detections]
        self.birth_buffer.update(unmatched_dets, self.frame_count)

        ready = self.birth_buffer.ready_candidates()
        for cand in ready:
            c_center, c_bbox = cand['center'], cand['bbox']
            quick_birth = cand['quick_birth']

            too_close = False
            for t in self.tracks.values():
                tc = t.get_current_center()
                tb = t.get_current_bbox()
                if tc is None or tb is None:
                    continue
                if math.hypot(tc[0] - c_center[0], tc[1] - c_center[1]) < MERGE_RADIUS:
                    too_close = True
                    break
                if self.calculate_iou(tb, c_bbox) > BIRTH_MAX_IOU:
                    too_close = True
                    break
            if too_close:
                continue

            side_now = self.side_of_line(c_center, p1, p2)
            in_gate_now = self.point_in_gate(c_center, p1, p2, line_type)

            initial_side = side_now if not in_gate_now else 0

            if quick_birth and FORCE_INITIAL_SIDE_ON_QUICK_BIRTH:
                initial_side = self.entry_sign

            t = PersonTrack(self.next_track_id, c_bbox, initial_side, quick_birth)
            t.update_detection(c_bbox, 1.0, self.frame_count)

            if quick_birth:
                t.birth_frame = self.frame_count
                t.edge_grace_until = self.frame_count + RIGHT_EDGE_GRACE_FRAMES
                if t.first_in_gate_frame is None:
                    t.first_in_gate_frame = self.frame_count

            self.tracks[self.next_track_id] = t
            self.next_track_id += 1

            if quick_birth:
                self.quick_births += 1

        self._check_crossings(p1, p2, line_type)

    def _check_crossings(self, p1, p2, line_type):
        for t in list(self.tracks.values()):
            center = t.get_current_center()
            if center is None:
                continue

            in_gate = self.point_in_gate(center, p1, p2, line_type)
            current_side = self.side_of_line(center, p1, p2)
            dist_to_line = self.perp_distance_to_line(center, p1, p2)

            if in_gate:
                t.frames_in_gate += 1
                t.frames_outside_gate = 0
                if t.first_in_gate_frame is None:
                    t.first_in_gate_frame = self.frame_count
            else:
                t.frames_outside_gate += 1
                if t.frames_outside_gate > 3:
                    t.frames_in_gate = 0

            if in_gate and t.first_in_gate_frame is not None:
                time_in_gate = self.frame_count - t.first_in_gate_frame
                if time_in_gate > GATE_TIMEOUT:
                    t.initial_side = current_side
                    t.crossing_confirmations = 0
                    t.first_in_gate_frame = None

            if not in_gate and t.frames_outside_gate > 5:
                t.first_in_gate_frame = None
                if dist_to_line > MIN_DISTANCE_FROM_LINE:
                    t.initial_side = current_side
                    t.crossing_confirmations = 0

            t.in_gate_area = in_gate

            if t.is_anomalous:
                t.crossing_confirmations = 0
                continue

            reliable = t.is_reliable()
            if t.quick_birth and self.frame_count <= t.edge_grace_until:
                reliable = True

            if not reliable:
                t.crossing_confirmations = 0
                continue

            if in_gate:
                has_movement = True
                if len(t.center_history) >= 3:
                    p_a = t.center_history[-1]
                    p_b = t.center_history[-3]
                    move = math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])
                    has_movement = move >= MIN_MOVE_IN_GATE

                if not has_movement:
                    continue

                if not t.can_cross_again(self.frame_count):
                    continue

                if t.initial_side == 0:
                    t.initial_side = current_side

                if t.initial_side != 0 and abs(current_side) > 3:
                    initial_sign = np.sign(t.initial_side)
                    current_sign = np.sign(current_side)

                    if initial_sign != current_sign and current_sign != 0:
                        t.crossing_confirmations += 1

                        needed = CROSSING_CONFIRMATION_FRAMES
                        if t.quick_birth and self.frame_count <= t.edge_grace_until:
                            needed = EARLY_CROSS_CONFIRM_FRAMES

                        if t.crossing_confirmations >= needed:
                            if initial_sign == self.entry_sign:
                                direction = "ENTRADA"
                                if t.has_entrada:
                                    t.last_cross_frame = self.frame_count
                                    t.initial_side = current_side
                                    t.crossing_confirmations = 0
                                    t.first_in_gate_frame = None
                                    t.last_crossing_type = direction
                                    continue
                                self.entrada_count += 1
                                t.has_entrada = True
                            else:
                                direction = "SAÍDA"
                                if t.has_saida:
                                    t.last_cross_frame = self.frame_count
                                    t.initial_side = current_side
                                    t.crossing_confirmations = 0
                                    t.first_in_gate_frame = None
                                    t.last_crossing_type = direction
                                    continue
                                self.saida_count += 1
                                t.has_saida = True

                            t.total_crosses += 1
                            t.last_cross_frame = self.frame_count
                            self.total_crossings += 1
                            t.last_crossing_type = direction

                            t.initial_side = current_side
                            t.crossing_confirmations = 0
                            t.first_in_gate_frame = None

                            birth = "QB" if t.quick_birth else "NB"
                            print(f"✓ #{self.total_crossings} | ID:{t.id} | {direction} | Q:{t.track_quality:.2f} | {birth}")
            else:
                if t.initial_side == 0 and abs(current_side) > 15:
                    t.initial_side = current_side
                t.crossing_confirmations = 0

    def draw_interface(self, frame):
        if self.line_config is None:
            return frame

        p1 = tuple(self.line_config["line_start"])
        p2 = tuple(self.line_config["line_end"])
        line_type = self.line_config.get("line_type", "horizontal")

        cv2.line(frame, p1, p2, (0, 255, 255), 4)
        cv2.line(frame, p1, p2, (0, 0, 0), 1)

        if line_type == "vertical":
            line_x = int((p1[0] + p2[0]) / 2)
            gate_left = max(0, line_x - self.gate_width)
            gate_right = min(W, line_x + self.gate_width)

            overlay = frame.copy()
            cv2.rectangle(overlay, (gate_left, 0), (gate_right, H), (0, 255, 255), -1)
            alpha = 0.15
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.rectangle(frame, (gate_left, 0), (gate_right, H), (0, 255, 255), 2)

            quick_zone_left = W - QUICK_BIRTH_ZONE_RIGHT
            cv2.rectangle(frame, (quick_zone_left, 0), (W, H), (100, 255, 100), 1)
        else:
            line_y = int((p1[1] + p2[1]) / 2)
            gate_top = max(0, line_y - self.gate_width)
            gate_bottom = min(H, line_y + self.gate_width)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, gate_top), (W, gate_bottom), (0, 255, 255), -1)
            alpha = 0.15
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.rectangle(frame, (0, gate_top), (W, gate_bottom), (0, 255, 255), 2)

        panel_w, panel_h = 340, 155
        if PANEL_ANCHOR == "tl":
            px, py = 5, 5
        elif PANEL_ANCHOR == "tr":
            px, py = W - panel_w - 5, 5
        elif PANEL_ANCHOR == "br":
            px, py = W - panel_w - 5, H - panel_h - 5
        else:
            px, py = 5, H - panel_h - 5

        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (100, 100, 100), 2)

        cv2.putText(frame, f"ENTRADA: {self.entrada_count}", (px + 10, py + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        cv2.putText(frame, f"SAIDA: {self.saida_count}", (px + 10, py + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 3)

        active_tracks = len([t for t in self.tracks.values() if t.is_reliable()])
        anomalous_tracks = len([t for t in self.tracks.values() if t.is_anomalous])
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0

        cv2.putText(frame, f"Tracks: {active_tracks} | Anomalias: {anomalous_tracks}",
                    (px + 10, py + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {avg_fps:.1f} | QuickBirths: {self.quick_births}",
                    (px + 10, py + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        for t in self.tracks.values():
            center = t.get_current_center()
            bbox = t.get_current_bbox()
            if center is None or bbox is None:
                continue

            if t.is_anomalous:
                color = (0, 0, 255)
            elif t.in_gate_area:
                color = (0, 255, 255)
            elif t.total_crosses > 0:
                color = (0, 255, 0)
            else:
                intensity = int(255 * t.track_quality)
                color = (intensity, intensity, 255)

            x, y, w, h = bbox
            thickness = 3 if t.is_anomalous else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            bbox_center = (x + w // 2, y + h // 2)
            cv2.circle(frame, bbox_center, 3, (255, 255, 255), -1)
            cv2.circle(frame, center, 5, color, -1)

            label = f"ID:{t.id} Q:{t.track_quality:.1f}"
            if t.total_crosses > 0:
                label += f" X:{t.total_crosses}"
            if t.quick_birth:
                label += " QB"

            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

    def run(self):
        global CONFIDENCE_THRESHOLD, CROSSING_CONFIRMATION_FRAMES, COOLDOWN_AFTER_CROSS

        if not self.load_line_config():
            return

        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Falha ao conectar à câmera RTSP")
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        except Exception:
            pass

        print("=" * 70)
        print("CONTADOR DE PESSOAS - MODO HEADLESS")
        print("=" * 70)
        print(f"Duração máxima: {HEADLESS_MAX_SEC}s (~{HEADLESS_MAX_SEC//60} min)")
        print("=" * 70)

        paused = False
        last_frame = None
        start_ts = time.time()
        
        # Variáveis de controle
        last_save_time = start_ts
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 30
        frames_processed = 0
        SAVE_INTERVAL_SEC = int(os.getenv("SAVE_INTERVAL_SEC", "600"))

        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_ts
                
                # Verifica timeout
                if HEADLESS and elapsed >= HEADLESS_MAX_SEC:
                    print(f"\n⏱ Tempo máximo atingido ({HEADLESS_MAX_SEC}s)")
                    break
                
                # Verifica shutdown
                if shutdown_requested:
                    print(f"\n⚠ Shutdown solicitado")
                    break

                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_failures += 1
                        print(f"✗ Falha ao ler frame ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                        
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            print(f"✗ Muitas falhas consecutivas - encerrando")
                            break
                        
                        time.sleep(0.5)
                        continue
                    
                    consecutive_failures = 0

                    frame = cv2.resize(frame, (W, H))
                    last_frame = frame.copy()
                    self.frame_count += 1
                    frames_processed += 1

                    t0 = time.time()
                    detections = self.detect_people(frame)
                    self.detection_history.append(len(detections))
                    self.update_tracks(detections)
                    dt_iter = time.time() - t0
                    if dt_iter > 0:
                        self.fps_counter.append(1.0 / dt_iter)

                    frame = self.draw_interface(frame)
                    
                    # Salvamento periódico
                    if HEADLESS and (current_time - last_save_time) >= SAVE_INTERVAL_SEC:
                        print(f"\n💾 Salvamento automático (t={int(elapsed)}s, frames={frames_processed})")
                        try:
                            from reporter import Reporter
                            now_gmt3 = _now_gmt3()
                            if _within_run_window_gmt3(now_gmt3):
                                rep = Reporter()
                                rep.write_hourly(self.entrada_count, when=now_gmt3)
                                print(f"✓ Dados salvos: ENTRADA={self.entrada_count}")
                                last_save_time = current_time
                            else:
                                print(f"⏭ Fora da janela 10-21h (hora: {now_gmt3.hour}h)")
                        except Exception as e:
                            print(f"⚠ Erro ao salvar: {e}")
                    
                    # Log de progresso
                    if HEADLESS and frames_processed % 100 == 0:
                        fps = frames_processed / elapsed if elapsed > 0 else 0
                        remaining = HEADLESS_MAX_SEC - elapsed
                        print(f"   [{int(elapsed)}s] Frames: {frames_processed} | FPS: {fps:.1f} | "
                              f"ENTRADA: {self.entrada_count} | SAÍDA: {self.saida_count} | "
                              f"Restante: {int(remaining)}s")
                else:
                    if last_frame is not None:
                        frame = self.draw_interface(last_frame.copy())

                if not HEADLESS:
                    cv2.imshow("Contador de Pessoas", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

        finally:
            cap.release()
            if not HEADLESS:
                cv2.destroyAllWindows()

            # Salvamento final
            print(f"\n💾 Salvamento final...")
            try:
                from reporter import Reporter
                now_gmt3 = _now_gmt3()
                rep = Reporter()
                rep.write_hourly(self.entrada_count, when=now_gmt3)
                print(f"✓ Reporte final enviado: ENTRADAS={self.entrada_count}")
            except Exception as e:
                print(f"⚠ Erro no salvamento final: {e}")

            # Relatório final
            end_time = time.time()
            total_time = end_time - start_ts
            
            print("\n" + "=" * 70)
            print("RELATÓRIO FINAL")
            print("=" * 70)
            print(f"Duração: {int(total_time)}s (~{total_time/60:.1f} min)")
            print(f"Frames processados: {frames_processed}")
            if total_time > 0:
                print(f"FPS médio: {frames_processed/total_time:.2f}")
            print(f"---")
            print(f"✓ ENTRADA: {self.entrada_count}")
            print(f"✓ SAÍDA: {self.saida_count}")
            print(f"Total de cruzamentos: {self.total_crossings}")
            print(f"Tracks criados: {self.next_track_id - 1}")
            print(f"Quick births: {self.quick_births}")
            print(f"Anomalias: {self.anomaly_detections}")
            print("=" * 70)


def main():
    try:
        counter = PeopleCounter()
        counter.run()
    except KeyboardInterrupt:
        print("\n✗ Programa interrompido pelo usuário")
    except Exception as e:
        print(f"\n✗ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()