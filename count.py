# dbName = "fcCollege"

RTSP = "10.30.161.131"
frame_skip = 2

## set line coords here.
x1, y1, x2, y2 = 520, 479, 1481, 503


import time
import numpy as np
from collections import defaultdict
import cv2
from ultralytics import YOLO
import datetime
from dataLogging.utils import smooth_points, point_position, add_reading_to_db





import logging

import os

# Set the CUDA_MODULE_LOADING environment variable to LAZY
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
rtsp_url = f"rtsp://admin:HuaWei123@{RTSP}/LiveMedia/ch1/Media1"
# path_to_model = "/home/lums/new_folder/vehicle_counting/yolov8n_vehicle_detection.pt"
path_to_model = "/lums/new_folder/vehicle_counting/yolov8m_AugmentedData_new_2.pt"
# path_to_model = "/lums/new_folder/vehicle_counting/countingCode/yolov8x_7Classes.engine"

model = YOLO(path_to_model)


# cv2.namedWindow('track', cv2.WINDOW_NORMAL)
# cv2.resizeWindow("track", (1000, 1000))

label_names = ['car', 'motorcycle', 'van', 'rickshaw', 'bus', 'truck', 'driver']


num_frames = 0

# Store the track history
track_history = defaultdict(lambda: [])

count_dict = {'car': 0, 'motorcycle': 0, 'van': 0, 'rickshaw': 0, 'bus': 0, 'truck': 0, 'driver': 0}


line_point1 = (x1, y1)
line_point2 = (x2, y2)

window_size = 5
threshold=0.6

cooldown_dict = {}  # Dictionary to store cooldown timers for each object
cooldown_time = 1200  # Cooldown time in frames


# Initial setup for timing
while True:

    try: 
        start_time = time.time()
        cap = cv2.VideoCapture(rtsp_url)
        print('cap: ', cap)
        _fps= int(cap.get(cv2.CAP_PROP_FPS))
        print("_fps: ", _fps)
        
        while time.time() - start_time < 3600: # run for 1hr
            ret, frame = cap.read()
            
            num_frames+=1
            if num_frames % frame_skip != 0:
                continue
            if num_frames%100==0:
                end_time=time.time()
                fps = num_frames / (end_time-start_time)
                print("fps: ", fps)
            results = model.track(frame, verbose=False, persist=True, conf=0.3, iou=0.8, max_det=300, half=False, imgsz=640, show_conf=False, show_labels=False)
            # Get the boxes and track IDs

            try:
                boxes = results[0].boxes.xywh.cpu()
                labels = results[0].boxes.cls.int().cpu().tolist()
                # print(labels)
                track_ids = results[0].boxes.id.int().cpu().tolist()
            except Exception as e:
                # print("missed result: ", e)
                continue
            ano_frame = results[0].plot()
            cv2.line(ano_frame, line_point1, line_point2, (0, 0, 255), 3)  # draw a line
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                if cooldown_dict.get(track_id, 0) > 0:
                    cooldown_dict[track_id] -= 1
                    continue
                
                position = point_position(x1, y1, x2, y2, x, y)
                # print("x, y", x.item(), y.item())
                # where=(int(x.item()+10), int(y.item()+10))
                # cv2.putText(ano_frame, f"Position: {position}", where, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                smoothed_track = smooth_points(track, window_size=window_size)
                first_point = smoothed_track[0]
                last_point = smoothed_track[-1]
                # Check the positions
                pos_first = point_position(x1, y1, x2, y2, first_point[0], first_point[1])
                pos_last = point_position(x1, y1, x2, y2, last_point[0], last_point[1])

                if pos_first == "above" and pos_last == "below":
                    confidence_above_below = sum(1 for p in smoothed_track if point_position(x1, y1, x2, y2, p[0], p[1]) == "above") / window_size
                    if confidence_above_below >= threshold:
                        
                        label = label_names[labels[track_ids.index(track_id)]]
                        # print(label)
                        count_dict[label] += 1  # Increment the count of the detected label
                        cooldown_dict[track_id] = cooldown_time
                        # print(f"Object with ID {track_id} and label {label} crossed the line from above to below. Total count: {count_dict}")
                        
                # if pos_first == "above" and pos_last == "below":
                #     count+=1
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(smoothed_track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(ano_frame, [points], isClosed=False, color=(0, 255,0), thickness=3)

            current_time = time.time()
            
            display_count_text = ", ".join([f"{k}: {v}" for k, v in count_dict.items()])
            cv2.putText(ano_frame, f"Counts: {display_count_text}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

            cv2.imshow('track', ano_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        track_history = defaultdict(lambda: [])
        cooldown_dict = {}
        num_frames = 0
        print('resting for 5 mins...')
        time.sleep(300)
    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        logging.error(error_message)
        cap.release()
        track_history = defaultdict(lambda: [])
        cooldown_dict = {}
        num_frames = 0
        time.sleep(150)
        continue
cv2.destroyAllWindows()
