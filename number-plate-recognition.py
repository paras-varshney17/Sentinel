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

        # 🔥 Unique filtering (DB spam control)
        self.plate_history = {}
        self.unique_window = 10  # seconds

        # 🔥 DB (replace with your real credentials)
        NEON_DB_URL = "postgresql://neondb_owner:npg_T8bhfCs5tZeI@ep-red-mode-amjsx3r0.c-5.us-east-1.aws.neon.tech/neondb?sslmode=require"

        self.db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=NEON_DB_URL
        )

    def get_conn(self):
        return self.db_pool.getconn()

    def release_conn(self, conn):
        self.db_pool.putconn(conn)

    def save_to_db(self, plate_text):
        conn = None
        try:
            conn = self.get_conn()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO plates (plate_text) VALUES (%s)",
                (plate_text,)
            )
            conn.commit()

            print(f"💾 Saved: {plate_text}")

        except Exception as e:
            print("DB Error:", e)

        finally:
            if conn:
                self.release_conn(conn)

    def normalize_text(self, text):
        text = text.replace(" ", "").upper()
        text = text.replace("O", "0").replace("I", "1")
        return text

    def detect_plates(self, frame):
        results = self.model.predict(frame, conf=0.4, verbose=False)
        if results and results[0].boxes is not None:
            return results[0].boxes.xyxy.cpu().numpy()
        return []

    def preprocess(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return gray

    def extract_text(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return ""

        # resize for better OCR
        roi = cv2.resize(roi, None, fx=2, fy=2)

        processed = self.preprocess(roi)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        variants = [roi, processed]

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

        return best_text.strip()

    def infer_ipcam(self, ip_url):
        cap = cv2.VideoCapture(ip_url)

        if not cap.isOpened():
            raise ValueError("❌ Camera not opened")

        print("🚀 ANPR LIVE RUNNING...")

        while True:
            ret, frame = cap.read()

            # 🔥 FIX: no fake frames
            if not ret or frame is None:
                print("⚠️ Camera disconnected")
                break

            frame = cv2.resize(frame, (960, 720))

            boxes = self.detect_plates(frame)
            ann = Annotator(frame, line_width=2)

            for bbox in boxes:
                text = self.extract_text(frame, bbox)

                # 🔥 Always display live text (NO FREEZE)
                ann.box_label(bbox, label=text, color=colors(17, True))

                # 🔥 DB logic separate from display
                if len(text) >= 6:
                    norm_text = self.normalize_text(text)
                    current_time = time.time()

                    last_seen = self.plate_history.get(norm_text, 0)

                    if current_time - last_seen > self.unique_window:
                        self.plate_history[norm_text] = current_time

                        print(f"[{time.strftime('%H:%M:%S')}] Plate: {norm_text}")
                        self.save_to_db(norm_text)

            # clean old entries
            self.plate_history = {
                k: v for k, v in self.plate_history.items()
                if time.time() - v < 60
            }

            try:
                cv2.imshow("ANPR LIVE", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except:
                pass

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ip_camera_url = "http://172.20.50.202:8080/video"

    anpr = ANPR("anpr_best.pt")
    anpr.infer_ipcam(ip_camera_url)