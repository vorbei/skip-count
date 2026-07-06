// Pose inference off the main thread. The UI/video/skeleton stay smooth even when
// a single inference is slow (esp. on CPU). Main thread transfers an ImageBitmap
// per frame; we return only the landmark array it needs.
import { PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
let landmarker = null, ready = false;

async function init(){
  try{
    const fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    const opts = (delegate) => ({
      baseOptions:{ modelAssetPath: MODEL_URL, delegate },
      runningMode:'VIDEO', numPoses:1,
      minPoseDetectionConfidence:0.5, minPosePresenceConfidence:0.5, minTrackingConfidence:0.5
    });
    let delegate = 'GPU';
    try{ landmarker = await PoseLandmarker.createFromOptions(fileset, opts('GPU')); }
    catch(e){ landmarker = await PoseLandmarker.createFromOptions(fileset, opts('CPU')); delegate = 'CPU'; }
    ready = true;
    self.postMessage({ type:'ready', delegate });
  }catch(e){
    self.postMessage({ type:'error', message: String((e && e.message) || e) });
  }
}

self.onmessage = (e) => {
  const m = e.data;
  if(m.type === 'init'){ init(); return; }
  if(m.type === 'frame'){
    const bmp = m.bitmap;
    if(!ready){ if(bmp && bmp.close) bmp.close(); self.postMessage({ type:'result', lm:null, ts:m.ts, ms:0 }); return; }
    const t0 = performance.now();
    let res = null;
    try{ res = landmarker.detectForVideo(bmp, m.ts); }catch(err){}
    const ms = performance.now() - t0;
    if(bmp && bmp.close) bmp.close();
    let out = null;
    if(res && res.landmarks && res.landmarks[0]){
      const a = res.landmarks[0];
      out = new Array(a.length);
      for(let i=0;i<a.length;i++) out[i] = { x:a[i].x, y:a[i].y, visibility:a[i].visibility };
    }
    self.postMessage({ type:'result', lm:out, ts:m.ts, ms });
  }
};
