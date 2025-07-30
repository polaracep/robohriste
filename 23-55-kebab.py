from __future__ import annotations

"""ball_tracker.py — Kalman‑filter (mellow) edition + ROI support
============================================================
Adds an optional Region‑Of‑Interest (ROI) crop to accelerate circle
 detection and draws the ROI rectangle live on screen.
"""

import argparse
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

class GameState: 
    score_paused = True

class Timer:
    INTERVAL = 90
    def __init__(self, elapsed_callback): 
        self.callback = elapsed_callback
        self.start_time = 0
        self.running = False
        self.time_s = Timer.INTERVAL

    def start_timer(self):
        if self.time_s == Timer.INTERVAL:
            self.start_time = time.time()
        else: 
            self.start_time = time.time() - Timer.INTERVAL + self.time_s
        self.running = True
    
    def clear_timer(self):
        self.start_time = time.time()
        self.time_s = Timer.INTERVAL

    def stop_timer(self):
        self.running = False

    def get_timer(self):
        if self.running:
            self.time_s = self.start_time + Timer.INTERVAL - time.time()
            if self.time_s <= 0 and self.running:
                self.time_s = 0
                self.stop_timer()
                self.callback()
                return 0

        return self.time_s



def seconds_to_mmss(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(minutes):02}:{int(remaining_seconds):02}"

# -----------------------------------------------------------------------------
#                           Kalman‑based, mellow Tracker
# -----------------------------------------------------------------------------
class KalmanTracker:
    """Per‑object constant‑velocity Kalman filters with early ghost cleanup."""

    def __init__(
        self,
        ghost_frames: int = 10,          # frames to keep an unmatched track
        max_dist: int = 120,             # gating radius (proc px)
        q: float = 5e-4,                 # process‑noise scalar (lower = calmer)
        r: float = 5e-2                  # measurement‑noise scalar (lower = trust detections)
    ):
        self.next_id = 0
        self.kfs: Dict[int, cv2.KalmanFilter] = OrderedDict()
        self.pred: Dict[int, Tuple[int, int]] = OrderedDict()
        self.lost: Dict[int, int] = OrderedDict()
        self.ghost_frames = ghost_frames
        self.max_dist = max_dist
        self.q = q; self.r = r

    # ---------------- Kalman helpers ---------------- #
    def _new_kf(self, x: float, y: float):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.measurementMatrix = np.eye(2,4, dtype=np.float32)
        kf.processNoiseCov      = self.q * np.eye(4, dtype=np.float32)
        kf.measurementNoiseCov  = self.r * np.eye(2, dtype=np.float32)
        kf.errorCovPost         = np.eye(4, dtype=np.float32)
        kf.statePost            = np.array([[x],[y],[0],[0]], np.float32)
        return kf

    def _register(self, pt: Tuple[int,int]):
        self.kfs[self.next_id] = self._new_kf(*pt)
        self.pred[self.next_id] = pt
        self.lost[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid: int):
        self.kfs.pop(oid, None); self.pred.pop(oid, None); self.lost.pop(oid, None)

    def update(self, detections: List[Tuple[int,int]]):
        
        """Return dict id→(x,y) of *predicted* positions (post‑correction)."""
        # 1. Predict ahead
        for oid, kf in self.kfs.items():
            p = kf.predict(); self.pred[oid] = (int(p[0]), int(p[1]))

        ids   = list(self.kfs.keys())
        predC = np.array(list(self.pred.values())) if ids else np.empty((0,2))
        detC  = np.array(detections)

        # 2. Handle no detections
        if detC.size == 0:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > self.ghost_frames:
                    self._deregister(oid)
            return self.pred

        # 3. First frame
        if predC.size == 0:
            for pt in detections: self._register(pt)
            return self.pred

        # 4. Distance matrix on predictions
        D = np.linalg.norm(predC[:,None] - detC[None,:], axis=2)
        used_r, used_c = set(), set()
        while True:
            if np.isinf(D).all(): break
            r,c = divmod(D.argmin(), D.shape[1])
            if D[r,c] > self.max_dist: break
            if r in used_r or c in used_c: D[r,c]=np.inf; continue
            oid = ids[r]
            meas = np.array([[detC[c,0]],[detC[c,1]]], np.float32)
            self.kfs[oid].correct(meas)
            self.pred[oid] = tuple(detC[c])
            self.lost[oid] = 0
            used_r.add(r); used_c.add(c)
            D[r,:]=np.inf; D[:,c]=np.inf

        # 5. unmatched tracks
        for r, oid in enumerate(ids):
            if r not in used_r:
                self.lost[oid] += 1
                if self.lost[oid] > self.ghost_frames:
                    self._deregister(oid)

        # 6. unmatched detections
        for c, pt in enumerate(detections):
            if c not in used_c: self._register(pt)

        return self.pred

    def is_visible(self, oid: int) -> bool:
        return self.lost.get(oid, 1) == 0

# -----------------------------------------------------------------------------
#                                CLI
# -----------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser("Track blue balls & score crossings (mellow KF) with ROI support")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--src", type=int, default=0)
    g.add_argument("--file", type=str)

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--line_y", type=float, default=0.56)

    p.add_argument("--score_top", type=int, default=12)
    p.add_argument("--score_bottom", type=int, default=12)

    p.add_argument("--min_r", type=int, default=10)
    p.add_argument("--max_r", type=int, default=60)
    p.add_argument("--hough_p2", type=int, default=25)

    p.add_argument("--max_dist", type=int, default=120)
    p.add_argument("--ghost_frames", type=int, default=7)
    p.add_argument("--kf_q", type=float, default=5e-4)
    p.add_argument("--kf_r", type=float, default=5e-2)

    # ROI rectangle: X Y W H in processed‑frame coordinates
    p.add_argument("--roi", type=int, nargs=4, metavar=("X","Y","W","H"),
                   help="ROI rectangle (processed resolution). Default is full frame.")

    p.add_argument("--wb", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p

# -----------------------------------------------------------------------------
#          Colour mask & Hough‑circle detection
# -----------------------------------------------------------------------------
LOWER_BLUE = np.array([90,60,30]); UPPER_BLUE = np.array([135,255,255])


def gray_world(img):
    b,g,r = cv2.split(img.astype(np.float32)); m=(b.mean()+g.mean()+r.mean())/3
    b*=m/b.mean(); g*=m/g.mean(); r*=m/r.mean(); return cv2.merge([b,g,r]).clip(0,255).astype(np.uint8)

# ─── 1. Make sure this helper is in the file (put it near the other utils) ───
def circles_nonmax_suppression(circles: np.ndarray, max_overlap: float = 0.5):
    """
    Greedy NMS that keeps circles whose pair-wise overlap area
    is ≤ `max_overlap` of the *smaller* circle’s area.
    """
    if circles.size == 0:
        return []

    # largest radius first → bigger balls keep priority
    circles = circles[circles[:, 2].argsort()[::-1]]

    kept: list[tuple[int, int, int]] = []
    for x, y, r in circles:
        accept = True
        for kx, ky, kr in kept:
            d = np.hypot(x - kx, y - ky)
            if d >= r + kr:                 # separate – OK
                continue

            # lens–lens intersection
            if d <= abs(kr - r):            # one fully inside the other
                overlap = np.pi * min(kr, r) ** 2
            else:
                φ = 2 * np.acos((d**2 + kr**2 - r**2) / (2 * d * kr))
                θ = 2 * np.acos((d**2 +  r**2 - kr**2) / (2 * d *  r))
                overlap = 0.5 * kr**2 * (φ - np.sin(φ)) \
                        + 0.5 *  r**2 * (θ - np.sin(θ))

            if overlap / (np.pi * min(kr, r) ** 2) > max_overlap:
                accept = False
                break
        if accept:
            kept.append((int(x), int(y), int(r)))
    return kept


# ─── 2. Replace your current detect_circles() with this version ───
def detect_circles(frame, p):
    img  = gray_world(frame) if p.wb else frame
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    g = cv2.medianBlur(
            cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask),
                         cv2.COLOR_BGR2GRAY), 5)

    cir = cv2.HoughCircles(
            g, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=p.min_r * 1.5, param1=50, param2=p.hough_p2,
            minRadius=p.min_r, maxRadius=p.max_r)

    # ── NEW: non-max suppression so balls don’t overlap >50 % ──
    if cir is None:
        return [], mask

    cir = np.round(cir[0]).astype(int)              # (N,3) → ints
    cir = circles_nonmax_suppression(cir, 0.40)     # filter

    return [(x, y) for x, y, _ in cir], mask


# -----------------------------------------------------------------------------
#                             Drawing helper
# -----------------------------------------------------------------------------

def draw_ui(img, tracker: KalmanTracker, sx, sy, mid_px, b_top, b_bot, s_top, s_bot, timer: Timer, roi_rect: Tuple[int,int,int,int], debug: bool):
    h,w = img.shape[:2]
    cv2.line(img,(0,mid_px),(w-1,mid_px),(0,255,255),2)

    # ROI rectangle (green)
    rx,ry,rw,rh = roi_rect
    cv2.rectangle(img,
                  (int(rx*sx), int(ry*sy)),
                  (int((rx+rw)*sx), int((ry+rh)*sy)),
                  (0,255,0), 2)

    for oid,(cx,cy) in tracker.pred.items():
        if not tracker.is_visible(oid): continue  # skip ghosts
        px,py = int(cx*sx), int(cy*sy)
        cv2.circle(img,(px,py),12,(0,0,255),2)
        cv2.putText(img,str(oid),(px-10,py-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        
    
    time = seconds_to_mmss(timer.get_timer())
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thicc = 2


    text= f"Kryptokoule: {b_top}"
    size = cv2.getTextSize(text, font, font_scale, font_thicc)
    cv2.putText(img,text,(w-10-size[0][0],40),font,font_scale,(255,0,0),font_thicc)

    text= f"Kryptokoule: {b_bot}"
    size = cv2.getTextSize(text, font, font_scale, font_thicc)
    cv2.putText(img,text,(w-10-size[0][0],h-20),font,font_scale,(0,0,255),font_thicc)

    cv2.putText(img,f"Body: {s_top}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.putText(img,f"Body: {s_bot}",(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
    size = cv2.getTextSize(time, font, font_scale, font_thicc)

    cv2.putText(img,f"{time}",((w-size[0][0])//2,40),font,font_scale,(255,0,0),font_thicc)

    #cv2.putText(img,f"{seconds_to_mmss(timer.get_timer())}",(w-100,h-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    if debug:
        cv2.putText(img,"DEBUG",(w-120,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    
    if not timer.running:
        center_x = (w//2)
        center_y = (h//2)
        cv2.line(img, (center_x - 30, center_y - 60), (center_x - 30, center_y + 60), (0, 0, 255), 20)
        cv2.line(img, (center_x + 30, center_y - 60), (center_x + 30, center_y + 60), (0, 0, 255), 20)

# -----------------------------------------------------------------------------
#                                    Main
# -----------------------------------------------------------------------------

def nothing():
    pass

def pause_game():
    GameState.score_paused = True

def main():
    args = make_parser().parse_args()
    if not 0<args.line_y<1: sys.exit("line_y must be in (0,1)")
    cap = cv2.VideoCapture(args.file if args.file else args.src)
    if not cap.isOpened(): sys.exit("Cannot open video source")

    tracker = KalmanTracker(args.ghost_frames, args.max_dist, args.kf_q, args.kf_r)
    mid_y_proc = int(args.height*args.line_y); BUFFER=3
    balls_t, balls_b = args.score_top, args.score_bottom
    score_t, score_b = 0, 0
    last_y: Dict[int,int]={}

    # ROI defaults to full processed frame if not provided
    if args.roi:
        roi_x, roi_y, roi_w, roi_h = args.roi
    else:
        roi_x, roi_y, roi_w, roi_h = 0, 200, args.width, 400
    if not (0 <= roi_x < args.width and 0 <= roi_y < args.height):
        sys.exit("ROI origin outside frame")


    # slidery
    cv2.namedWindow("Ball Tracker", cv2.WINDOW_NORMAL)      # already exists, but

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_EXPOSURE, 20)

    t0,fc=time.perf_counter(),0

    timer = Timer(pause_game)
    timer.start_timer()

    GameState.score_paused = False

    last_s = 0
    
    while True:
        # -------------------------------------------------------- key handling
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            break
        elif key == 10 or key == 13:  # Enter to restart
            if timer.running: 
                timer.stop_timer()
                GameState.score_paused = True
            else:
                timer.start_timer()
                GameState.score_paused = False
        elif key == ord(' '):  # Space to pause/resume
            timer.stop_timer()
            GameState.score_paused = True
            timer.clear_timer()
            balls_b = args.score_bottom
            balls_t = args.score_top
            score_b = -12
            score_t = -12

        # -------------------------------------------------------- zbytek 

        ret, frame = cap.read()
        if not ret: break
        proc = cv2.resize(frame,(args.width,args.height),cv2.INTER_AREA)

        # -------------------------------------------------------- ROI processing
        roi = proc[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        dets_roi, mask_roi = detect_circles(roi,args)
        dets = [(x+roi_x, y+roi_y) for x,y in dets_roi]
        # Embed ROI mask back into full mask for debug view
        mask_full: Optional[np.ndarray] = None
        if args.debug:
            mask_full = np.zeros(proc.shape[:2], dtype=np.uint8)
            mask_full[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = mask_roi

        # ----------------------------------------------------------------------

        tracker.update(dets)

        # crossings only for visible IDs
        for oid,(cx,cy) in tracker.pred.items():
            if not tracker.is_visible(oid): continue
            prev = last_y.get(oid)
            if GameState.score_paused == False:
                if prev is not None:
                    if prev < mid_y_proc-BUFFER <= cy:
                        balls_t -= 1; balls_b += 1
                    elif prev > mid_y_proc+BUFFER >= cy:
                        balls_t += 1; balls_b -= 1
                last_y[oid] = cy
        for oid in list(last_y):
            if oid not in tracker.pred or not tracker.is_visible(oid): last_y.pop(oid)

        # -------------------------------------------------------------- UI Rendering

        sx = frame.shape[1]/args.width
        sy = frame.shape[0]/args.height

        # a second has passed, add the points
        if not int(timer.get_timer()) == int(last_s):
            last_s = int(timer.get_timer())
            score_t += 24-balls_t
            score_b += 24-balls_b

        draw_ui(frame, tracker, sx, sy, int(frame.shape[0]*args.line_y),
                balls_t, balls_b, score_t, score_b, timer, (roi_x,roi_y,roi_w,roi_h), args.debug)

        fc+=1
        fps = fc/(time.perf_counter()-t0); t0,fc=time.perf_counter(),0
        fps_text = f"FPS:{fps:.1f}"
        size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.putText(frame,fps_text,((frame.shape[1]-size[0][0])//2,frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.3,(0xff,0xff,0xff),1)

        cv2.imshow("Ball Tracker", frame)
        if args.debug and mask_full is not None:
            cv2.imshow("Mask", mask_full)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
