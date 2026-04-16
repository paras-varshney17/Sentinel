import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time
import psycopg2
from psycopg2 import pool
import re


class ANPR:
    def __init__(self, model_path="anpr_best.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)

        self.reader = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available(),
            verbose=False
        )

        # Cache
        self.last_text = ""
        self.last_time = 0
        self.cooldown = 5  # seconds

        # Temporal voting buffer
        self.text_buffer = []

        # PostgreSQL Pool
        self.db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 5,
            host="10.68.174.21",
            database="anpr_db",
            user="postgres",
            password="1234"
        )

    def get_conn(self):
        return self.db_pool.getconn()

    def release_conn(self, conn):
        self.db_pool.putconn(conn)

    def save_to_db(self, plate_text, camera_id="CAM_1"):
        conn = None
        try:
            conn = self.get_conn()
            cursor = conn.cursor()

            query = """
            INSERT INTO plates (plate_text, camera_id)
            VALUES (%s, %s)
            """
            cursor.execute(query, (plate_text, camera_id))
            conn.commit()

            print(f"💾 Saved to DB: {plate_text}")

        except Exception as e:
            print("DB Error:", e)

        finally:
            if conn:
                self.release_conn(conn)

    def detect_plates(self, frame):
        results = self.model.predict(frame, conf=0.4, verbose=False)

        if results and results[0].boxes is not None:
            return results[0].boxes.xyxy.cpu().numpy()
        return []

    # 🔥 Improved preprocessing
    def preprocess_roi(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        gray = cv2.equalizeHist(gray)

        # Noise reduction
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        gray = cv2.filter2D(gray, -1, kernel)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return thresh

    # 🔥 Regex cleanup
    def clean_plate(self, text):
        text = text.replace("O", "0").replace("I", "1")
        match = re.findall(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}', text)
        return match[0] if match else text

    # 🔥 Multi-variant OCR
    def extract_text(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        # Padding
        pad = 5
        x1, y1 = x1 - pad, y1 - pad
        x2, y2 = x2 + pad, y2 + pad

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return ""

        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        variants = [
            roi,
            self.preprocess_roi(roi),
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ]

        best_text = ""
        best_conf = 0

        for var in variants:
            results = self.reader.readtext(
                var,
                detail=1,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            for (_, txt, conf) in results:
                if conf > best_conf:
                    best_conf = conf
                    best_text = txt

        return self.clean_plate(best_text.strip())

    def infer_ipcam(self, ip_url, display=True, duration=None):
        cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise ValueError("❌ Cannot open IP camera stream")

        frame_count = 0
        skip_frames = 4
        start_time = time.time()

        print("🚀 Running Optimized ANPR... Press 'q' to quit")

        while True:
            if duration and (time.time() - start_time > duration):
                print("⏹ Time limit reached")
                break

            ret, frame = cap.read()
            if not ret:
                continue

            cap.grab()
            frame = cv2.resize(frame, (640, 480))

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            boxes = self.detect_plates(frame)
            ann = Annotator(frame, line_width=2)

            for bbox in boxes:
                current_time = time.time()

                text = self.extract_text(frame, bbox)

                if text:
                    self.text_buffer.append(text)

                # 🔥 Temporal voting
                if len(self.text_buffer) >= 5:
                    final_text = max(set(self.text_buffer), key=self.text_buffer.count)
                    self.text_buffer.clear()

                    if (
                        len(final_text) >= 6 and
                        final_text != self.last_text and
                        current_time - self.last_time > self.cooldown
                    ):
                        self.last_text = final_text
                        self.last_time = current_time

                        print(f"[{time.strftime('%H:%M:%S')}] Plate: {final_text}")
                        self.save_to_db(final_text)

                display_text = self.last_text if self.last_text else text
                ann.box_label(bbox, label=display_text, color=colors(17, True))

            if display:
                cv2.imshow("ANPR Optimized", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ip_camera_url = "http://10.68.174.129:8080/video"

    anpr = ANPR("anpr_best.pt")
    anpr.infer_ipcam(ip_camera_url, display=True, duration=None)