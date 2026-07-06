// Pose inference off the main thread. The UI/video/skeleton stay smooth even when
// a single inference is slow (esp. on CPU). Main thread transfers an ImageBitmap
// per frame; we return only the landmark array it needs.
import { PoseLandmarker, FilesetResolver } from './mediapipe/vision_bundle.mjs';

// self-hosted assets, resolved relative to this worker's URL (works at root OR /subpath/)
const WASM_PATH = new URL('mediapipe/wasm', location.href).href;
const MODEL_URL = new URL('mediapipe/pose_landmarker_lite.task', location.href).href;
let landmarker = null, ready = false;

async function init(){
  try{
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
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
