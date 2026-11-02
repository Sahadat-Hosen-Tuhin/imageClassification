import cv2
import os

VIDEO_DIR = "data/raw/videos"
FRAME_DIR = "data/processed/frames"

os.makedirs(FRAME_DIR, exist_ok=True)

frame_rate = 30   # every 30th frame

for video_name in os.listdir(VIDEO_DIR):
    if not video_name.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        continue

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_name = f"{video_name}_frame_{saved}.jpg"
            cv2.imwrite(os.path.join(FRAME_DIR, frame_name), frame)
            saved += 1

        count += 1

    cap.release()

print("✅ Frame Extraction Completed")
