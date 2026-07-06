import cv2, json, sys, os, numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE = "/private/tmp/claude-501/-Users-cheng-Maxgent-maxgent-worktree-flywheel/a51db06a-c38b-46e1-bb67-0b38f77be762/scratchpad/"
VIDEO = os.environ.get("VIDEO", "/Users/cheng/Downloads/IMG_6258.MOV")
MODEL = BASE + "pose_landmarker_lite.task"
OUT   = os.environ.get("OUT", BASE + "pose_signal.json")

PROC_W, PROC_H = 540, 960      # aspect-correct portrait (matches render + browser)
MW, MH = 80, 60                 # rope motion-diff resolution (mirrors browser)

base = python.BaseOptions(model_asset_path=MODEL)
opts = vision.PoseLandmarkerOptions(base_options=base,
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
lm_er = vision.PoseLandmarker.create_from_options(opts)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps} frames={total}", flush=True)

prev_gray = None
records = []
idx = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    t_ms = int(idx / fps * 1000)
    small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = lm_er.detect_for_video(mp_img, t_ms)

    # rope motion map (grayscale frame diff at MW x MH), top-left origin
    g = cv2.cvtColor(cv2.resize(small, (MW, MH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = None
    if prev_gray is not None:
        d = np.abs(g - prev_gray)
        d[d <= 12] = 0
        diff = d
    prev_gray = g

    rec = {"t": t_ms, "present": False, "rope": 0.0}
    if res.pose_landmarks:
        lm = res.pose_landmarks[0]
        def vis(i): return getattr(lm[i], "visibility", 1.0)
        rec["present"] = True
        rec["nose_y"] = lm[0].y
        rec["sh_y"]  = (lm[11].y + lm[12].y) / 2
        rec["hip_y"] = (lm[23].y + lm[24].y) / 2
        rec["kn_y"]  = (lm[25].y + lm[26].y) / 2
        rec["ank_y"] = (lm[27].y + lm[28].y) / 2
        rec["vis_nose"] = vis(0)
        rec["vis_sh"]   = min(vis(11), vis(12))
        rec["vis_hip"]  = min(vis(23), vis(24))
        rec["vis_kn"]   = min(vis(25), vis(26))
        rec["vis_ank"]  = min(vis(27), vis(28))
        # arm geometry (trip = arms fling wide off the body)
        rec["sh_cx"] = (lm[11].x + lm[12].x) / 2
        rec["sh_w"]  = abs(lm[11].x - lm[12].x)
        rec["wl_x"], rec["wl_y"] = lm[15].x, lm[15].y   # left wrist
        rec["wr_x"], rec["wr_y"] = lm[16].x, lm[16].y   # right wrist
        rec["el_x"], rec["er_x"] = lm[13].x, lm[14].x   # elbows x
        rec["vis_wr"] = min(vis(15), vis(16))
        # rope pass near feet
        if vis(27) > 0.35 and vis(28) > 0.35 and diff is not None:
            ax = (lm[27].x + lm[28].x) / 2
            ay = (lm[27].y + lm[28].y) / 2
            x0, y0, x1, y1 = ax - 0.17, ay - 0.05, ax + 0.17, ay + 0.17
            gx0, gx1 = max(0, int(x0 * MW)), min(MW, int(np.ceil(x1 * MW)))
            gy0, gy1 = max(0, int(y0 * MH)), min(MH, int(np.ceil(y1 * MH)))
            if gx1 > gx0 and gy1 > gy0:
                sub = diff[gy0:gy1, gx0:gx1]
                rec["rope"] = float(sub.mean() / 255.0)
    records.append(rec)
    idx += 1
    if idx % 200 == 0:
        print(f"  {idx}/{total}", flush=True)

cap.release()
json.dump({"fps": fps, "records": records}, open(OUT, "w"))
present = sum(1 for r in records if r["present"])
print(f"done: {len(records)} frames, present={present}  -> {OUT}", flush=True)
