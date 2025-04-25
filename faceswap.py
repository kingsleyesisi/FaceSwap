import os
import cv2
import gdown
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import insightface
from insightface.app import FaceAnalysis
from onnxruntime.quantization import quantize_dynamic, QuantType

# Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_DIR = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model paths and download URL
fp32_model = os.path.join(MODEL_DIR, 'inswapper_128.onnx')
int8_model = os.path.join(MODEL_DIR, 'inswapper_128_int8.onnx')
drive_url = 'https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF'

# Download FP32 model if missing
if not os.path.exists(fp32_model):
    print(f"[+] Downloading FP32 model to {fp32_model}")
    gdown.download(drive_url, fp32_model, quiet=False)

# Quantize to INT8 if missing
if not os.path.exists(int8_model):
    print(f"[+] Quantizing {fp32_model} to INT8 -> {int8_model}")
    quantize_dynamic(
        model_input='models/inswapper_128.onnx',
        model_output='models/inswapper_128_int8.onnx',
        weight_type=QuantType.QInt8
        )


# Choose the quantized model if available
model_path = int8_model if os.path.exists(int8_model) else fp32_model

# Load InsightFace models
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(model_path)

# Helper to save uploaded files
def save_file(file, folder):
    filename = secure_filename(file.filename)
    path = os.path.join(folder, filename)
    file.save(path)
    return path

@app.route('/swap', methods=['POST'])
def swap_faces():
    if 'target' not in request.files or 'source' not in request.files:
        return jsonify({'error': 'Missing image(s). Upload both target and source.'}), 400

    target_path = save_file(request.files['target'], UPLOAD_FOLDER)
    source_path = save_file(request.files['source'], UPLOAD_FOLDER)

    img_target = cv2.imread(target_path)
    img_source = cv2.imread(source_path)

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

if __name__ == '__main__':
    app.run(debug=True)
