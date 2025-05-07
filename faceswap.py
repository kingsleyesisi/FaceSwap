import os
import cv2
import gdown
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from insightface.app import FaceAnalysis
import insightface
from onnxruntime.quantization import quantize_dynamic, QuantType

# === Configuration ===
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_DIR = 'models'
FP32_MODEL = os.path.join(MODEL_DIR, 'inswapper_128.onnx')
INT8_MODEL = os.path.join(MODEL_DIR, 'inswapper_128_int8.onnx')
DRIVE_URL = 'https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF'
MAX_DIM = 512  # max width/height to reduce memory footprint

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Download FP32 model if missing
def ensure_model():
    if not os.path.exists(FP32_MODEL):
        gdown.download(DRIVE_URL, FP32_MODEL, quiet=False)
    if not os.path.exists(INT8_MODEL):
        quantize_dynamic(
            model_input=FP32_MODEL,
            model_output=INT8_MODEL,
            weight_type=QuantType.QUInt8,
            per_channel=False
        )
ensure_model()
model_path = INT8_MODEL if os.path.exists(INT8_MODEL) else FP32_MODEL

# Setup InsightFace with lighter detector
face_analyzer = FaceAnalysis(
    name='antelope',               # smaller model (~100MB)
    allowed_modules=['detection', 'landmark']
)
face_analyzer.prepare(ctx_id=-1, det_size=(320, 320))  # CPU only
swapper = insightface.model_zoo.get_model(model_path)

# FastAPI app\ app = FastAPI()

# Utility: save UploadFile to disk
async def save_upload(file: UploadFile, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file.filename)
    with open(path, 'wb') as f:
        f.write(await file.read())
    return path

# Utility: downscale image to max dimension
def resize_image(img):
    h, w = img.shape[:2]
    scale = min(MAX_DIM / max(h, w), 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

@app.post('/swap')
async def swap_faces(target: UploadFile = File(...), source: UploadFile = File(...)):
    # Save uploads
    tgt_path = await save_upload(target, UPLOAD_FOLDER)
    src_path = await save_upload(source, UPLOAD_FOLDER)

    # Read and resize
    img_tgt = cv2.imread(tgt_path)
    img_src = cv2.imread(src_path)
    if img_tgt is None or img_src is None:
        raise HTTPException(status_code=400, detail='Invalid image files')
    img_tgt = resize_image(img_tgt)
    img_src = resize_image(img_src)

    # Detect faces
    faces_tgt = face_analyzer.get(img_tgt)
    faces_src = face_analyzer.get(img_src)
    if not faces_tgt or not faces_src:
        raise HTTPException(status_code=400, detail='No faces detected')

    # Perform swap
    face_t = faces_tgt[0]
    face_s = faces_src[0]
    out_img = swapper.get(img_tgt.copy(), face_t, face_s, paste_back=True)

    # Encode to JPEG
    success, buffer = cv2.imencode('.jpg', out_img)
    if not success:
        raise HTTPException(status_code=500, detail='Failed to encode output')
    return Response(content=buffer.tobytes(), media_type='image/jpeg')

@app.get('/health')
def health():
    return {'status': 'ok'}
