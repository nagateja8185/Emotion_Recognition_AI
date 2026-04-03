"""
Sky Blue Laptop-Friendly Emotion Recognition Server
- Serves both Text & Image Emotion Detection Web UI
- Supports real-time webcam updates (auto every 2s)
- Handles both /predict_text and /predict_image endpoints
"""

import os
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email.parser import BytesParser
from email.policy import default as default_policy
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ---------- AUTO-DETECT PATHS (Cross-Platform) ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_IMG_PATH = os.path.join(BASE_DIR, "models", "image_emotion.h5")
MODEL_TEXT_PIPELINE = os.path.join(BASE_DIR, "models", "text_emotion", "pipeline.joblib")
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "data", "images", "fer2013", "train")
WEB_DEMO_DIR = os.path.join(BASE_DIR, "web_demo")
PORT = 8000
# ----------------------------

FALLBACK_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']

# ---------- HELPERS ----------
def load_image_labels():
    try:
        labels = sorted([d for d in os.listdir(TRAIN_IMAGE_DIR)
                         if os.path.isdir(os.path.join(TRAIN_IMAGE_DIR, d))])
        return labels if labels else FALLBACK_LABELS
    except Exception:
        return FALLBACK_LABELS

def preprocess_image_bytes(image_bytes, target_size=(48, 48)):
    """Convert raw image bytes → preprocessed grayscale face tensor"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces):
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        crop = gray[y:y+h, x:x+w]
    else:
        crop = gray
    crop = cv2.resize(crop, target_size)
    arr = crop.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

# ---------- SERVER ----------
class EmotionHandler(BaseHTTPRequestHandler):
    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            index_path = os.path.join(WEB_DEMO_DIR, "index.html")
            if not os.path.exists(index_path):
                self.send_error(500, f"HTML file not found: {index_path}")
                return
            with open(index_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/predict_text":
            self.handle_text()
        elif self.path == "/predict_image":
            self.handle_image()
        else:
            self.send_error(404)

    def handle_text(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode())
            text = data.get("text", "")
            if not text:
                self._json(400, {"error": "No text provided"})
                return

            pred = self.server.text_model.predict([text])[0]
            probs = self.server.text_model.predict_proba([text])[0].tolist()
            classes = list(self.server.text_model.classes_)
            self._json(200, {"label": pred, "probs": probs, "classes": classes})
        except Exception as e:
            self._json(500, {"error": f"Text prediction error: {str(e)}"})

    def handle_image(self):
        """Handle webcam frame upload robustly across browsers."""
        try:
            ctype = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            img_bytes = None

            # ---- Case 1: multipart form (browser FormData) ----
            if "multipart/form-data" in ctype:
                boundary = ctype.split("boundary=")[-1].encode()
                # split the multipart content
                parts = body.split(b"--" + boundary)
                for part in parts:
                    if b"Content-Disposition" in part and b"name=\"file\"" in part:
                        # extract after header's blank line
                        idx = part.find(b"\r\n\r\n")
                        if idx != -1:
                            img_bytes = part[idx + 4 :].strip().rstrip(b"--")
                            break

            # ---- Case 2: raw image bytes (fallback) ----
            if not img_bytes and body:
                img_bytes = body

            if not img_bytes:
                self._json(400, {"error": "No file provided"})
                return

            # ---- Preprocess and predict ----
            arr = preprocess_image_bytes(img_bytes)
            probs = self.server.img_model.predict(arr)[0].tolist()
            label = self.server.img_labels[int(np.argmax(probs))]

            self._json(200, {"label": label, "probs": probs})

        except Exception as e:
            self._json(500, {"error": str(e)})



def run(port=PORT):
    print("Loading models...")
    
    # Validate model files exist
    if not os.path.exists(MODEL_IMG_PATH):
        print(f"❌ Image model not found: {MODEL_IMG_PATH}")
        print("   Please run: python src/train_image.py")
        return
    
    if not os.path.exists(MODEL_TEXT_PIPELINE):
        print(f"❌ Text model not found: {MODEL_TEXT_PIPELINE}")
        print("   Please run: python src/train_text.py")
        return
    
    try:
        img_model = load_model(MODEL_IMG_PATH)
        text_model = joblib.load(MODEL_TEXT_PIPELINE)
        img_labels = load_image_labels()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    print(f"✅ Models loaded successfully!")
    print(f"   - Image model: {MODEL_IMG_PATH}")
    print(f"   - Text model: {MODEL_TEXT_PIPELINE}")
    print(f"   - Image labels: {img_labels}")
    print(f"\n🚀 Server running at http://localhost:{port}")
    print(f"   Open in browser: http://localhost:{port}\n")

    server = ThreadingHTTPServer(("", port), EmotionHandler)
    server.img_model = img_model
    server.text_model = text_model
    server.img_labels = img_labels
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        server.server_close()


if __name__ == "__main__":
    run()
