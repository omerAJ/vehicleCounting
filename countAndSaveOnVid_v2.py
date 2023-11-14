# dbName = "GovernorHouse"

RTSP = "10.30.161.131"
frame_skip = 2
index = 6
video_path = f'/lums/new_folder/gatherFrames/gatherVids/night/New_output_N2.avi'
# csv_filepath = f"/lums/new_folder/vehicle_counting/countingCode/GT/vid_{index}.csv"
csv_filepath = f"/lums/new_folder/vehicle_counting/countingCode/GT/vid_N2.csv"

## set line coords here.
x1, y1, x2, y2 = 450, 436, 1610, 441


import time
import numpy as np
from collections import defaultdict
import cv2
from ultralytics import YOLO
import datetime
import sys
sys.path.append('/lums/new_folder/vehicle_counting/countingCode/')
from dataLogging.utils import smooth_points, point_position, add_reading_to_db
import threading
from pynput import keyboard

def putTextMultiline(count_dict, start_pos):
    start_x, start_y = start_pos
    line_height = 30  # Adjust as needed for spacing

    for i, (k, v) in enumerate(count_dict.items()):
        text = f"{k}: {v}"
        y = start_y + i * line_height
        cv2.putText(ano_frame, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 255), 3)


from pynput import keyboard

class KeyCounter:
    def __init__(self):
        self.GTdict = {'car': 0, 'motorcycle': 0, 'rickshaw': 0, 'van': 0, 'bus': 0, 'truck': 0}

    def on_press(self, key):
        # Check if the key is a numpad key
        try:
            if key.char == '1':  # Numpad 1
                self.GTdict['car'] += 1
            elif key.char == '2':  # Numpad 2
                self.GTdict['motorcycle'] += 1
            elif key.char == '3':  # Numpad 3
                self.GTdict['rickshaw'] += 1
            elif key.char == '4':  # Numpad 4
                self.GTdict['van'] += 1
            elif key.char == '8':  # Numpad 5
                self.GTdict['bus'] += 1
            elif key.char == '6':  # Numpad 6
                self.GTdict['truck'] += 1
        except AttributeError:
            pass

    def run(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()


key_counter = KeyCounter()
    
# Create a new thread for key listening
thread = threading.Thread(target=key_counter.run)
thread.daemon = True  # Daemon threads exit when the program does
thread.start()


model_count_dict = {'car': 0, 'motorcycle': 0, 'rickshaw': 0, 'van': 0, 'bus': 0, 'truck': 0}


import csv

def setup_csv():
    with open(csv_filepath, "w", newline='') as file:
        writer = csv.writer(file)
        # Define the headers based on the dictionary keys
        headers = ['timestamp'] + \
                  [f'{key}_model' for key in model_count_dict.keys()] + \
                  [f'{key}_GT' for key in key_counter.GTdict.keys()]
        writer.writerow(headers)

def write_to_csv(timestamp, model_counts, GT_counts):
    with open(csv_filepath, "a", newline='') as file:
        writer = csv.writer(file)
        # Create a row with the timestamp followed by values from both dictionaries
        row = [timestamp] + list(model_counts.values()) + list(GT_counts.values())
        writer.writerow(row)

setup_csv()

import logging

import os

# Set the CUDA_MODULE_LOADING environment variable to LAZY
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# rtsp_url = f"rtsp://admin:HuaWei123@{RTSP}/LiveMedia/ch1/Media1"
# path_to_model = "/lums/new_folder/vehicle_counting/vehicles_6Classes__OnlyOnNew/v8x_part2/weights/best.pt"

# path_to_model = "/lums/new_folder/vehicle_counting/yolov8m_AugmentedData_new_2.pt"
path_to_model = "/lums/new_folder/vehicle_counting/yolov8x_augmentedData_new2.engine"
# path_to_model = "/lums/new_folder/vehicle_counting/yolov8x_augmentedData_new2.pt"

# path_to_model = "/lums/new_folder/vehicle_counting/vehicles_7Classes_augmented_new/v8m4/weights/best.pt"



model = YOLO(path_to_model)

def save_frame(frame, n, base_name="frame_"):
    file_name = f"/lums/new_folder/gatherFrames/{base_name}_{n}.jpg"
    cv2.imwrite(file_name, frame)
    print(f"Frame saved as {file_name}")
# cv2.namedWindow('track', cv2.WINDOW_NORMAL)
# cv2.resizeWindow("track", (1000, 1000))

label_names = ['car', 'motorcycle', 'van', 'rickshaw', 'bus', 'truck']


num_frames = 0

# Store the track history
track_history = defaultdict(lambda: [])

line_point1 = (x1, y1)
line_point2 = (x2, y2)

window_size = 5
threshold=0.6

cooldown_dict = {}  # Dictionary to store cooldown timers for each object
cooldown_time = 1200  # Cooldown time in frames

num=len(os.listdir('/lums/new_folder/gatherFrames'))
# print('\n num: ', nucount_m)
# Initial setup for timing


start_time = time.time()
cap = cv2.VideoCapture(video_path)
_fps= int(cap.get(cv2.CAP_PROP_FPS))
print("_fps: ", _fps)
frames_per_5_seconds = (5 * _fps)
frames_since_last_write = 0
print('cap: ', cap)
_fps= int(cap.get(cv2.CAP_PROP_FPS))
print("_fps: ", _fps)

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        num_frames+=1
        # if num_frames % frame_skip != 0:
        #     continue
        frames_since_last_write+=1
        if num_frames%100==0:
            end_time=time.time()
            fps = num_frames / (end_time-start_time)
            print("fps: ", fps)
        results = model.track(frame, task='detect', verbose=False, persist=True, conf=0.3, iou=0.8, max_det=300, half=False, imgsz=640, show_conf=False, show_labels=False)
        # Get the boxes and track IDs
        # If 's' key is pressed, save the current frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(num)
            save_frame(frame, n=num)
            num+=1
        try:
            boxes = results[0].boxes.xywh.cpu()
            labels = results[0].boxes.cls.int().cpu().tolist()
            # print(labels)
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except Exception as e:
            # print("missed result: ", e)
            continue
        # ano_frame = results[0].plot()
        ano_frame = frame
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
                    model_count_dict[label] += 1  # Increment the count of the detected label
                    cooldown_dict[track_id] = cooldown_time
                    # print(f"Object with ID {track_id} and label {label} crossed the line from above to below. Total count: {count_dict}")
                    
            # if pos_first == "above" and pos_last == "below":
            #     count+=1
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            ## Draw the tracking lines
            # points = np.hstack(smoothed_track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(ano_frame, [points], isClosed=False, color=(0, 255,0), thickness=3)

        current_time = time.time()
        
        # display_count_text = "\n".join([f"{k}: {v}" for k, v in count_dict.items()])
        # cv2.putText(ano_frame, f"Counts: \n{display_count_text}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (10, 10, 255), 1)
        # putTextMultiline(model_count_dict, (20, 100))
        putTextMultiline(key_counter.GTdict, (300, 100))
        if frames_since_last_write >= frames_per_5_seconds:
            current_time += 5
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(current_time))
            with open(csv_filepath, "a", newline='') as file:
                write_to_csv(formatted_time, model_count_dict, key_counter.GTdict)
            frames_since_last_write = 0
        time.sleep(0.06)

        cv2.imshow('track', ano_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        if cv2.waitKey(1) & 0xFF == ord('z'):
            cv2.waitKey(0)
                
            
    else:
        print('missed frame')
        