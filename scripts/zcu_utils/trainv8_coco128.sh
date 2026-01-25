#! /bin/bash
source /home/choneil/repos/ultralytics/venv/bin/activate
yolo train detect model='weights/yolov8/split/yolov8n_split.pt' data=coco128.yaml epochs=20 imgsz=640
cp "$(ls -d /home/choneil/ultralytics/runs/detect/train* | sort -V | tail -1)/weights/best.pt" ./weights/yolov8/tuned/unfused/yolov8n_zcu102_unfused.pt
python fuse_weights.py ./weights/yolov8/tuned/unfused/yolov8n_zcu102_unfused.pt 
yolo train detect model='weights/yolov8/split/yolov8s_split.pt' data=coco128.yaml epochs=20 imgsz=640
cp "$(ls -d /home/choneil/ultralytics/runs/detect/train* | sort -V | tail -1)/weights/best.pt" ./weights/yolov8/tuned/unfused/yolov8s_zcu102_unfused.pt
python fuse_weights.py ./weights/yolov8/tuned/unfused/yolov8s_zcu102_unfused.pt
yolo train detect model='weights/yolov8/split/yolov8m_split.pt' data=coco128.yaml epochs=20 imgsz=640
cp "$(ls -d /home/choneil/ultralytics/runs/detect/train* | sort -V | tail -1)/weights/best.pt" ./weights/yolov8/tuned/unfused/yolov8m_zcu102_unfused.pt
python fuse_weights.py ./weights/yolov8/tuned/unfused/yolov8m_zcu102_unfused.pt
