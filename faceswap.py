import os
import cv2
import gdown
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import insightface
from insightface.app import FaceAnalysis
from onnxruntime.quantization import quantize_dynamic, QuantType

# === Configuration ===
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_DIR = 'models'
FP32_MODEL = os.path.join(MODEL_DIR, 'inswapper_128.onnx')
INT8_MODEL = os.path.join(MODEL_DIR, 'inswapper_128_int8.onnx')
DRIVE_URL = 'https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF'

# === App Setup ===
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === Download and Quantize ONNX Model ===
if not os.path.exists(FP32_MODEL):
    print(f"[+] Downloading FP32 model to {FP32_MODEL}")
    gdown.download(DRIVE_URL, FP32_MODEL, quiet=False)

if not os.path.exists(INT8_MODEL):
    print(f"[+] Quantizing model to INT8: {INT8_MODEL}")
    quantize_dynamic(
        model_input=FP32_MODEL,
        model_output=INT8_MODEL,
        weight_type=QuantType.QUInt8,
        per_channel=False,
        op_types_to_quantize=["MatMul"]
    )

model_path = INT8_MODEL if os.path.exists(INT8_MODEL) else FP32_MODEL

# === Load InsightFace ===
face_analyzer = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition', 'landmark'])
face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))  # use CPU with ctx_id = -1
swapper = insightface.model_zoo.get_model(model_path)

# === Helpers ===
def save_file(file, folder):
    filename = secure_filename(file.filename)
    path = os.path.join(folder, filename)
    file.save(path)
    return path

# === Routes ===
@app.route('/swap', methods=['POST'])
def swap_faces():
    if 'target' not in request.files or 'source' not in request.files:
        return jsonify({'error': 'Missing image(s). Upload both target and source.'}), 400

    target_path = save_file(request.files['target'], UPLOAD_FOLDER)
    source_path = save_file(request.files['source'], UPLOAD_FOLDER)

    img_target = cv2.imread(target_path)
    img_source = cv2.imread(source_path)

    if img_target is None or img_source is None:
        return jsonify({'error': 'Could not read one or both images.'}), 400

    faces_target = face_analyzer.get(img_target)
    faces_source = face_analyzer.get(img_source)

    if not faces_target or not faces_source:
        return jsonify({'error': 'Could not detect face(s) in one or both images.'}), 400

    face_target = faces_target[0]
    face_source = faces_source[0]

    swapped_img = swapper.get(img_target.copy(), face_target, face_source, paste_back=True)

    result_path = os.path.join(RESULT_FOLDER, f"swapped_{os.path.basename(target_path)}")
    cv2.imwrite(result_path, swapped_img)

    return send_file(result_path, mimetype='image/jpeg')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# === Run App (For local testing only) ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=False)
