import cv2
import os

video_path = "run_7.mp4"
output_dir = "Images/run_7"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("output/run_7", exist_ok=True)
os.makedirs("colmap/run_7", exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit(1)

count = 0
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % 5 == 0:
        frame_name = f"frame_{frame_id:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_id += 1
    count += 1

cap.release()
print(f"Extracted {frame_id} frames to {output_dir}")
