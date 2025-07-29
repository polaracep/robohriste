#!/usr/bin/env python3
"""
ball_tracker_gui.py – Blue-ball tracker with a clean, panel-free GUI.

All controls now reside in keyboard shortcuts:
    Space … reset scores + timer
    Esc/Q  … quit

Score (Top / Bottom) and elapsed time are rendered directly onto the video
frame. No OpenCV trackbars or Qt slider widgets remain.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

# =============================================================================
#                              Tracking utilities
# =============================================================================
@dataclass
class Params:
    """Tweakable parameters (now fixed – no sliders)."""
    width: int = 1280
    height: int = 720
    line_y: float = 0.56

    min_r: int = 10
    max_r: int = 60
    hough_p2: int = 25

    max_dist: int = 120
    ghost_frames: int = 7
    kf_q: float = 5e-4
    kf_r: float = 5e-2

    roi: Tuple[int, int, int, int] = (0, 200, 1280, 400)


class Timer:
    def __init__(self) -> None:
        self.restart()

    def restart(self) -> None:
        self._start = time.time()

    def seconds(self) -> int:
        return int(time.time() - self._start)


class KalmanTracker:
    """Minimal ID-assigning, Kalman-filter-based multi-object tracker."""
    def __init__(self, ghost_frames: int, max_dist: int, q: float, r: float):
        self.next_id = 0
        self.kfs: Dict[int, cv2.KalmanFilter] = OrderedDict()
        self.pred: Dict[int, Tuple[int, int]] = OrderedDict()
        self.lost: Dict[int, int] = OrderedDict()
        self.ghost_frames = ghost_frames
        self.max_dist = max_dist
        self.q = q
        self.r = r

    # ---------------- internal helpers ----------------
    @staticmethod
    def _new_kf(x: float, y: float, q: float, r: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix    = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix   = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov     = q * np.eye(4, dtype=np.float32)
        kf.measurementNoiseCov = r * np.eye(2, dtype=np.float32)
        kf.errorCovPost        = np.eye(4, dtype=np.float32)
        kf.statePost           = np.array([[x], [y], [0], [0]], np.float32)
        return kf

    def _register(self, pt: Tuple[int, int]) -> None:
        self.kfs[self.next_id] = self._new_kf(*pt, self.q, self.r)
        self.pred[self.next_id] = pt
        self.lost[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid: int) -> None:
        self.kfs.pop(oid, None)
        self.pred.pop(oid, None)
        self.lost.pop(oid, None)

    # ------------------ public API --------------------
    def update(self, detections: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        # 1) predict existing tracks
        for oid, kf in self.kfs.items():
            p = kf.predict()
            self.pred[oid] = (int(p[0]), int(p[1]))

        ids   = list(self.kfs.keys())
        predC = np.array(list(self.pred.values())) if ids else np.empty((0, 2))
        detC  = np.array(detections)

        # 2) no detections: age tracks
        if detC.size == 0:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > self.ghost_frames:
                    self._deregister(oid)
            return self.pred

        # 3) first frame: register everything
        if predC.size == 0:
            for pt in detections:
                self._register(pt)
            return self.pred

        # 4) greedy matching
        D = np.linalg.norm(predC[:, None] - detC[None, :], axis=2)
        used_r, used_c = set(), set()
        while True:
            if np.isinf(D).all():
                break
            r, c = divmod(D.argmin(), D.shape[1])
            if D[r, c] > self.max_dist:
                break
            if r in used_r or c in used_c:
                D[r, c] = np.inf
                continue
            oid = ids[r]
            self.kfs[oid].correct(np.array([[detC[c, 0]], [detC[c, 1]]], np.float32))
            self.pred[oid] = tuple(detC[c])
            self.lost[oid] = 0
            used_r.add(r)
            used_c.add(c)
            D[r, :] = np.inf
            D[:, c] = np.inf

        # 5) age unmatched predictions
        for r, oid in enumerate(ids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > self.ghost_frames:
                    self._deregister(oid)

        # 6) register unmatched detections
        for c, pt in enumerate(detections):
            if c not in used_c:
                self._register(pt)

        return self.pred

    def is_visible(self, oid: int) -> bool:
        return self.lost.get(oid, 1) == 0


# =============================================================================
#                         Colour mask & circle detection
# =============================================================================
LOWER_BLUE = np.array([90,  60,  30])
UPPER_BLUE = np.array([135, 255, 255])


def gray_world(img: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img.astype(np.float32))
    m = (b.mean() + g.mean() + r.mean()) / 3
    b *= m / b.mean()
    g *= m / g.mean()
    r *= m / r.mean()
    return cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)


def circles_nms(circles: np.ndarray, max_overlap: float = .5):
    if circles.size == 0:
        return []
    circles = circles[circles[:, 2].argsort()[::-1]]
    kept: list[tuple[int, int, int]] = []
    for x, y, r in circles:
        accept = True
        for kx, ky, kr in kept:
            d = np.hypot(x - kx, y - ky)
            if d >= r + kr:
                continue
            if d <= abs(kr - r):
                overlap = np.pi * min(kr, r) ** 2
            else:
                φ = 2 * np.arccos((d ** 2 + kr ** 2 - r ** 2) / (2 * d * kr))
                θ = 2 * np.arccos((d ** 2 + r ** 2 - kr ** 2) / (2 * d * r))
                overlap = .5 * kr ** 2 * (φ - np.sin(φ)) + .5 * r ** 2 * (θ - np.sin(θ))
            if overlap / (np.pi * min(kr, r) ** 2) > max_overlap:
                accept = False
                break
        if accept:
            kept.append((int(x), int(y), int(r)))
    return kept


def detect_circles(frame: np.ndarray, p: Params):
    img  = gray_world(frame)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    g    = cv2.medianBlur(cv2.bitwise_and(img, img, mask=mask)[:, :, 0], 5)
    cir  = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2,
                            minDist=p.min_r * 1.5, param1=50, param2=p.hough_p2,
                            minRadius=p.min_r, maxRadius=p.max_r)
    if cir is None:
        return [], mask
    cir = np.round(cir[0]).astype(int)
    cir = circles_nms(cir, .4)
    return [(x, y) for x, y, _ in cir], mask


# =============================================================================
#                                   GUI
# =============================================================================
class BallTrackerWidget(QtWidgets.QWidget):
    def __init__(self, source: int | str = 0, params: Params | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Ball Tracker')
        self.params = params or Params()

        # ------------ video source -----------
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError('Cannot open video source')

        # ------------ video display ----------
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                       QtWidgets.QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet('background:#111;')

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)

        # ------------ state ------------------
        self.tracker = KalmanTracker(self.params.ghost_frames,
                                     self.params.max_dist,
                                     self.params.kf_q,
                                     self.params.kf_r)
        self.timer   = Timer()
        self.score_t = 0
        self.score_b = 0

        # ------------ event loop -------------
        self.qtimer = QtCore.QTimer(self)
        self.qtimer.timeout.connect(self._update_frame)
        self.qtimer.start(15)   # ~66 FPS max

    # ------------------------------------------------------------------
    def _update_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            return

        frame = cv2.resize(frame, (self.params.width, self.params.height))
        # ROI
        x, y, w, h = self.params.roi
        roi = frame[y:y + h, x:x + w].copy()

        # detection & tracking
        circles, _ = detect_circles(roi, self.params)
        det_centers = [(cx + x, cy + y) for cx, cy in circles]
        tracks = self.tracker.update(det_centers)

        # score update
        line_y_px = int(self.params.line_y * self.params.height)
        for oid, (cx, cy) in tracks.items():
            if not self.tracker.is_visible(oid):
                continue
            if cy < line_y_px and self.tracker.lost[oid] == 0:
                self.score_t += 1
            elif cy > line_y_px and self.tracker.lost[oid] == 0:
                self.score_b += 1

        # ------------- overlay ----------------
        disp = frame.copy()
        # crossing line
        cv2.line(disp, (0, line_y_px), (disp.shape[1], line_y_px), (0, 255, 255), 2)
        # detections + tracks
        for cx, cy in det_centers:
            cv2.circle(disp, (cx, cy), 4, (0, 0, 255), -1)
        for oid, (cx, cy) in tracks.items():
            color = (0, 255, 0) if self.tracker.is_visible(oid) else (120, 120, 120)
            cv2.circle(disp, (cx, cy), 10, color, 2)
            cv2.putText(disp, str(oid), (cx + 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)

        # on-screen labels (upper-left corner)
        cv2.putText(disp, f'Top: {self.score_t}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f'Bottom: {self.score_b}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f'{self.timer.seconds()} s', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # PyQt expects RGB
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = disp.shape
        qimg = QtGui.QImage(disp.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == QtCore.Qt.Key.Key_Space:
            self.score_t = self.score_b = 0
            self.timer.restart()
        elif e.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
            self.close()
        super().keyPressEvent(e)

    # ------------------------------------------------------------------
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.qtimer.stop()
        self.cap.release()
        super().closeEvent(e)


# =============================================================================
#                                   main
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Blue-Ball Tracker GUI (panel-free)')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--src',  type=int, help='camera index (default 0)')
    src.add_argument('--file', type=str, help='video file')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source = args.file if args.file else (args.src if args.src is not None else 0)

    app = QtWidgets.QApplication(sys.argv)
    w   = BallTrackerWidget(source)
    w.resize(1280, 720)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
