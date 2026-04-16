import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time
import psycopg2
from psycopg2 import pool


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

        # 🔥 Neon PostgreSQL Connection Pool
        NEON_DB_URL = "postgresql://neondb_owner:npg_T8bhfCs5tZeI@ep-red-mode-amjsx3r0.c-5.us-east-1.aws.neon.tech/neondb?sslmode=require"

        self.db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=NEON_DB_URL
        )

    # ✅ Get connection safely
    def get_conn(self):
        return self.db_pool.getconn()

    def release_conn(self, conn):
        self.db_pool.putconn(conn)

    # ✅ Save to PostgreSQL
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

    def preprocess_roi(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh

    def extract_text(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return ""

        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        processed = self.preprocess_roi(roi)

        text = self.reader.readtext(processed, detail=0)
        return " ".join(text).strip()

    def infer_ipcam(self, ip_url, display=True, duration=None):
        cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise ValueError("❌ Cannot open IP camera stream")

        frame_count = 0
        skip_frames = 4
        start_time = time.time()

        print("🚀 Running ANPR with PostgreSQL... Press 'q' to quit")

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

                if current_time - self.last_time > self.cooldown:
                    text = self.extract_text(frame, bbox)

                    if len(text) >= 6 and text != self.last_text:
                        self.last_text = text
                        self.last_time = current_time

                        # ✅ Terminal output
                        print(f"[{time.strftime('%H:%M:%S')}] Plate: {text}")

                        # ✅ Save to PostgreSQL
                        self.save_to_db(text)

                else:
                    text = self.last_text

                ann.box_label(bbox, label=text, color=colors(17, True))

            if display:
                cv2.imshow("ANPR PostgreSQL", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ip_camera_url = "http://10.68.174.129:8080/video"

    anpr = ANPR("anpr_best.pt")
    anpr.infer_ipcam(ip_camera_url, display=True, duration=None)