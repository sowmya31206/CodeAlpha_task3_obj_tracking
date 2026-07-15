!pip install ultralytics opencv-python

import os

for root, dirs, files in os.walk("/content/runs"):
    print(root)
    print(files)

import cv2
import os

from ultralytics import YOLO
from google.colab import files
from IPython.display import Video, display

model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully!")

!rm -rf /content/runs

uploaded = files.upload()

video_path = list(uploaded.keys())[0]

print("Uploaded Video:", video_path)

results = model.track(
    source=video_path,
    save=True,
    show=False,
    persist=True
)

print("Detection completed!")

!ffprobe /content/runs/detect/track/*.avi

!ffmpeg -i /content/runs/detect/track/*.avi -vcodec libx264 output.mp4

from IPython.display import Video
Video("output.mp4", embed=True)