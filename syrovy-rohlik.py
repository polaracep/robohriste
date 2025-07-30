#!/usr/bin/env python3
"""
ball_tracker_gui.py – Minimal, full-screen Ball-Tracker GUI.

PyQt-only overlays (scores & timer) float over the video; no sliders or
side-panels remain.

Run:
    python ball_tracker_gui.py             # default camera
    python ball_tracker_gui.py --file clip.mp4
"""
from __future__ import annotations
import argparse, sys, time, cv2, numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets

# ───────────────────────────── Tracking params ──────────────────────────────
@dataclass
class Params:
    width: int = 1280
    height: int = 720
    line_y: float = 0.56        # fraction of height (for score line)
    min_r: int = 10             # HoughCircles
    max_r: int = 60
    hough_p2: int = 25
    max_dist: int = 120
    ghost_frames: int = 7
    kf_q: float = 5e-4
    kf_r: float = 5e-2
    roi: Tuple[int, int, int, int] = (0, 200, 1280, 400)

class Timer:                                               # simple stopwatch
    def __init__(self): self.restart()
    def restart(self): self._t0 = time.time()
    def seconds(self): return int(time.time() - self._t0)

class KalmanTracker:
    def __init__(self, ghost_frames=10, max_dist=120, q=5e-4, r=5e-2):
        self.next_id = 0
        self.kfs: Dict[int, cv2.KalmanFilter] = OrderedDict()
        self.pred, self.lost = OrderedDict(), OrderedDict()
        self.ghost_frames, self.max_dist, self.q, self.r = ghost_frames, max_dist, q, r

    @staticmethod
    def _init_kf(x,y,q,r):
        k=cv2.KalmanFilter(4,2); k.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        k.measurementMatrix=np.eye(2,4,dtype=np.float32)
        k.processNoiseCov=q*np.eye(4,dtype=np.float32)
        k.measurementNoiseCov=r*np.eye(2,dtype=np.float32)
        k.errorCovPost=np.eye(4,dtype=np.float32); k.statePost=np.array([[x],[y],[0],[0]],np.float32); return k
    def _register(self,pt): self.kfs[self.next_id]=self._init_kf(*pt,self.q,self.r); self.pred[self.next_id]=pt; self.lost[self.next_id]=0; self.next_id+=1
    def _deregister(self,i): self.kfs.pop(i,None); self.pred.pop(i,None); self.lost.pop(i,None)
    def update(self,dets:List[Tuple[int,int]]):
        for i,k in self.kfs.items(): p=k.predict(); self.pred[i]=(int(p[0]),int(p[1]))
        ids=list(self.kfs); predC=np.array(list(self.pred.values())) if ids else np.empty((0,2)); detC=np.array(dets)
        if detC.size==0:
            for i in list(self.lost): self.lost[i]+=1; self._deregister(i) if self.lost[i]>self.ghost_frames else None; return self.pred
        if predC.size==0: [self._register(pt) for pt in dets]; return self.pred
        D=np.linalg.norm(predC[:,None]-detC[None,:],2,axis=2); used_r=used_c=set(),set()
        while not np.isinf(D).all():
            r,c=divmod(D.argmin(),D.shape[1]);  # greedy
            if D[r,c]>self.max_dist: break
            if r in used_r or c in used_c: D[r,c]=np.inf; continue
            i=ids[r]; self.kfs[i].correct(np.array([[detC[c,0]],[detC[c,1]]],np.float32))
            self.pred[i]=tuple(detC[c]); self.lost[i]=0; used_r.add(r); used_c.add(c); D[r,:]=D[:,c]=np.inf
        for r,i in enumerate(ids):
            if r not in used_r: self.lost[i]+=1; self._deregister(i) if self.lost[i]>self.ghost_frames else None
        [self._register(tuple(detC[c])) for c in range(len(dets)) if c not in used_c]
        return self.pred
    def is_visible(self,i): return self.lost.get(i,1)==0

# ───────────────────────────── Detection helpers ────────────────────────────
LOWER_BLUE=np.array([90,60,30]); UPPER_BLUE=np.array([135,255,255])
def gray_world(img):
    b,g,r=cv2.split(img.astype(np.float32)); m=(b.mean()+g.mean()+r.mean())/3
    return cv2.merge([*(c*(m/c.mean()) for c in (b,g,r))]).clip(0,255).astype(np.uint8)
def nonmax(cir,max_overlap=.5):
    if not len(cir): return []
    cir=cir[cir[:,2].argsort()[::-1]]; keep=[]
    for x,y,r in cir:
        if all(np.hypot(x-kx,y-ky)>=r+kr or np.hypot(x-kx,y-ky)<=abs(kr-r) and
               (np.pi*min(kr,r)**2)/(np.pi*min(kr,r)**2)<=max_overlap for kx,ky,kr in keep):
            keep.append((int(x),int(y),int(r)))
    return keep
def detect(frame,p):
    img=gray_world(frame); hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,LOWER_BLUE,UPPER_BLUE); g=cv2.medianBlur(cv2.bitwise_and(img,img,mask=mask)[:,:,0],5)
    cir=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.2,minDist=p.min_r*1.5,
                         param1=50,param2=p.hough_p2,minRadius=p.min_r,maxRadius=p.max_r)
    return ([] if cir is None else [(x+p.roi[0],y+p.roi[1]) for x,y,_ in nonmax(np.round(cir[0]).astype(int))], mask)

# ─────────────────────────────────── GUI ────────────────────────────────────
class BallTrackerWidget(QtWidgets.QWidget):
    def __init__(self,src:int|str=0,p:Params|None=None,parent=None):
        super().__init__(parent); self.setWindowTitle('Ball Tracker'); self.setStyleSheet('background:#000;')
        self.params=p or Params()
        self.cap=cv2.VideoCapture(src); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.params.width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.params.height)
        if not self.cap.isOpened(): raise RuntimeError('Cannot open source')
        # video label
        self.video=QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter); self.video.setStyleSheet('background:#000;')
        lay=QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.addWidget(self.video)
        # overlay labels
        def make_lbl(): 
            lbl=QtWidgets.QLabel('--',self.video)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet('font-size:64px; font-weight:700; color:#fff; background:rgba(0,0,0,0%); text-shadow:0 0 8px #000;')
            return lbl
        self.lbl_top, self.lbl_time, self.lbl_bot = make_lbl(), make_lbl(), make_lbl()
        # overlay layout inside video label
        ovl=QtWidgets.QVBoxLayout(self.video); ovl.setContentsMargins(30,30,90,30)
        ovl.addWidget(self.lbl_top,0,QtCore.Qt.AlignmentFlag.AlignTop|QtCore.Qt.AlignmentFlag.AlignHCenter)
        ovl.addStretch()
        ovl.addWidget(self.lbl_time,0,QtCore.Qt.AlignmentFlag.AlignRight)
        ovl.addStretch()
        ovl.addWidget(self.lbl_bot,0,QtCore.Qt.AlignmentFlag.AlignBottom|QtCore.Qt.AlignmentFlag.AlignHCenter)
        # tracking state
        self.track=KalmanTracker(self.params.ghost_frames,self.params.max_dist,self.params.kf_q,self.params.kf_r)
        self.tmr=Timer(); self.score_t=self.score_b=12
        # loop
        self.qtimer=QtCore.QTimer(self); self.qtimer.timeout.connect(self.refresh); self.qtimer.start(15)
    def refresh(self):
        ok,frame=self.cap.read()
        if not ok: return
        frame=cv2.resize(frame,(self.params.width,self.params.height))
        dets,_=detect(frame,self.params); trks=self.track.update(dets)
        line=int(self.params.line_y*self.params.height)
        for oid,(x,y) in trks.items():
            if not self.track.is_visible(oid): continue
            if y<line: self.score_t+=1; self.track.lost[oid]+=1  # force ghost to prevent double count
            elif y>line: self.score_b+=1; self.track.lost[oid]+=1
        disp=frame.copy(); cv2.line(disp,(0,line),(disp.shape[1],line),(0,255,255),2)
        for x,y in dets: cv2.circle(disp,(x,y),4,(0,0,255),-1)
        for oid,(x,y) in trks.items():
            color=(0,255,0) if self.track.is_visible(oid) else (120,120,120)
            cv2.circle(disp,(x,y),10,color,2); cv2.putText(disp,str(oid),(x+12,y-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,1,cv2.LINE_AA)
        disp=cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)
        h,w,ch=disp.shape; self.video.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(disp.data,w,h,ch*w,QtGui.QImage.Format.Format_RGB888)))
        self.lbl_top.setText(f'Skóre A: {self.score_t}')
        self.lbl_bot.setText(f'Skóre B: {self.score_b}')
        self.lbl_time.setText(f'{self.tmr.seconds()} s')
    def keyPressEvent(self,e):
        if e.key() in (QtCore.Qt.Key.Key_Space,): self.score_t=self.score_b=0; self.tmr.restart()
        elif e.key() in (QtCore.Qt.Key.Key_Escape,QtCore.Qt.Key.Key_Q): self.close()
    def closeEvent(self,e): self.qtimer.stop(); self.cap.release(); super().closeEvent(e)

# ────────────────────────────────── main ────────────────────────────────────
def arg_parse():
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group()
    g.add_argument('--src',type=int); g.add_argument('--file',type=str); return ap.parse_args()
def main():
    a=arg_parse(); src=a.file if a.file else (a.src if a.src is not None else 0)
    app=QtWidgets.QApplication(sys.argv); w=BallTrackerWidget(src); w.show(); sys.exit(app.exec())
if __name__=='__main__': main()
