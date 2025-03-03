from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
import cv2
import insightface
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Float, create_engine, Column, Integer, String, LargeBinary, Text, TIMESTAMP, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import psycopg2
import base64
import datetime
import threading
import pytz
import torch
import numpy as np
import threading
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from strongsort.strong_sort import StrongSORT
from strongsort.utils.parser import get_config
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tracking = True
CONFIDENCE_THRESHOLD = 0.4
WHITE = (255, 255, 255)
OUTPUT_SIZE = (640, 640)
YOLO_MODEL_PATH = "yolov8n.pt"
# TRACKER_CONFIG_PATH = "config.yaml"
TRACKER_WEIGHTS_PATH = "osnet_x0_25_market1501.pt"
HOMOGRAPHY_PATH = "homography_matrix_640.txt"
BACKGROUND_IMAGE_PATH = "new_cafeteria.jpg"

# Utility Functions
def load_yolo_model(model_path="yolov8n.pt"):
    """Load YOLO model."""
    return YOLO(model_path)

def initialize_tracker(weights_path, device, max_age=150, fp16=False):
    """Initialize StrongSORT tracker."""
    # config = get_config()
    # config.merge_from_file(config_path)
    return StrongSORT(model_weights=Path(weights_path), device=device, max_age=max_age, fp16=fp16)

def load_homography_matrix(homography_file):
    """Load homography matrix."""
    return np.loadtxt(homography_file)

def load_background_image(image_path):
    """Load background image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Background image not found at {image_path}")
    return img

def transform_to_image(x, y, h_matrix):
    """Transform video coordinates to room image coordinates."""
    video_coords = np.array([[x, y, 1]]).T  # Homogeneous coordinates
    image_coords = np.dot(h_matrix, video_coords)
    image_coords /= image_coords[2]  # Normalize by z-coordinate
    return int(image_coords[0]), int(image_coords[1])

def get_unique_color(track_id):
    """Generate unique color based on track ID."""
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())

def draw_dashed_line(image, start, end, color, thickness, dash_length=10):
    """Draw dashed line."""
    x1, y1 = start
    x2, y2 = end
    line_length = int(np.linalg.norm((x2 - x1, y2 - y1)))
    for i in range(0, line_length, dash_length * 2):
        start_pos = i / line_length
        end_pos = (i + dash_length) / line_length
        start_x = int(x1 + start_pos * (x2 - x1))
        start_y = int(y1 + start_pos * (y2 - y1))
        end_x = int(x1 + end_pos * (x2 - x1))
        end_y = int(y1 + end_pos * (x2 - y1))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

def draw_dotted_line(image, start, end, color, radius=3):
    """Draw dotted line."""
    x1, y1 = start
    x2, y2 = end
    line_length = int(np.linalg.norm((x2 - x1, y2 - y1)))
    for i in range(0, line_length, radius * 4):
        t = i / line_length
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (x2 - y1))
        cv2.circle(image, (px, py), radius, color, -1)
        
yolo_model = load_yolo_model(YOLO_MODEL_PATH)
tracker = initialize_tracker(TRACKER_WEIGHTS_PATH, DEVICE)
h_matrix = load_homography_matrix(HOMOGRAPHY_PATH)
bg_image = load_background_image(BACKGROUND_IMAGE_PATH)


app = FastAPI()

face_model = insightface.app.FaceAnalysis()
face_model.prepare(ctx_id=-1, det_size=(640, 640), det_thresh= .5)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()
# engine_ = create_engine('postgresql://postgres:postgres@172.16.200.43:32101/xenonstack')
# Session_ = sessionmaker(bind=engine_)
engine_ = create_engine('postgresql://postgres:postgres@10.0.0.210:5432/xenonai')
Session_ = sessionmaker(bind=engine_)
kolkata_zone = pytz.timezone('Asia/Kolkata')

conn = psycopg2.connect(
    dbname="mydb",
    user="cvdemo",
    password="cvdemo",
    host="172.16.200.229",
    port="30878"
)

class AsyncFrameCapture:
    def __init__(self, rtsp_link):
        self.rtsp_link = rtsp_link
        self.cap = cv2.VideoCapture(rtsp_link)
        self.frame = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def _capture_frames(self):
        while self.running:
            success, frame = self.cap.read()
            if frame is not None:
                frame = cv2.resize(frame, (640,640))
            if success:
                with self.lock:
                    self.frame = frame
                    self.condition.notify_all()

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class KnownFace(Base):
    __tablename__ = 'employee_details'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    employee_id = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone_number = Column(String(255), nullable=True)
    face_image_base64 = Column(Text, nullable=True)
    seat = Column(String(10), nullable=True)
    is_restricted_in = Column(Text, nullable=True)
    embedding = Column(Vector(512), nullable=True)

class DetectedLog(Base):
    __tablename__ = 'detected_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    emp_id = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)
    area = Column(String(225), nullable=True)
    face_image_base64 = Column(Text, nullable=True)
    detected_at = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False)
    seat = Column(String(10), nullable=True)

class DetectedUnknownLog(Base):
    __tablename__ = 'detected_unknown_logs'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), default='Unknown')
    area = Column(String(255), nullable=True)
    frame = Column(Text, nullable=True)
    detected_at = Column(TIMESTAMP, default=func.current_timestamp(), nullable=False)


class Area(Base):
    __tablename__ = 'cam_areas'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    rtsp = Column(String(255))

# class AreaHeadcount(Base):
#     __tablename__ = 'area_headcount'
    
#     id = Column(Integer, primary_key=True)
#     area = Column(String(255), nullable=False)
#     current_headcount = Column(Integer, nullable=False, default=0)
#     recognized_count = Column(Integer, nullable=False, default=0)
#     unrecognized_count = Column(Integer, nullable=False, default=0)  


def get_area_by_rtsp_url(rtsp_url):
    with Session_() as session:
        area = session.query(Area.name).filter(Area.rtsp == rtsp_url).first()
        print(area)
        return area[0]

def create_fresh_frame_instance(rtsp_link):
    if rtsp_link == "webcam":
        rtsp_link = 0
    cap = cv2.VideoCapture(rtsp_link)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return AsyncFrameCapture

# def generate_frames(rtsp_link, yolo_model):
#     print("Inside generate_frames")
#     if rtsp_link == "webcam":
#         rtsp_link = 0
#         area_name = "IT_room"
#     else:
#         area_name = get_area_by_rtsp_url(rtsp_link)
#     fresh_frame = AsyncFrameCapture(rtsp_link)
#     trajectory_points = {}
#     try:
#         while True:
#             try:
#                 frame = fresh_frame.get_frame()
#                 if frame is not None:

#                     # YOLO Predictions
#                     detections = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
#                     results = [
#                         [int(data[0]), int(data[1]), int(data[2]), int(data[3]), data[4], int(data[5])]
#                         for data in detections.boxes.data.tolist()
#                         if int(data[5]) == 0 and data[4] >= CONFIDENCE_THRESHOLD
#                     ]

#                     # Tracker Update
#                     detection_tensor = torch.tensor(
#                         results, dtype=torch.float32, device=DEVICE
#                     ) if results else torch.empty((0, 6), dtype=torch.float32, device=DEVICE)
#                     tracks = tracker.update(detection_tensor, frame)

#                     # Process Tracks
#                     for track in tracks:
#                         if isinstance(track, np.ndarray):
#                             ltrb, track_id = track[:4], int(track[4])
#                         else:
#                             ltrb, track_id = track.to_ltrb(), track.track_id

#                         xmin, ymin, xmax, ymax = map(int, ltrb)
#                         x, y = (xmin + xmax) // 2, (ymin + ymax) // 2
#                         mapped_point = transform_to_image(x, y, h_matrix)
#                         color = get_unique_color(track_id)

#                         roi = frame[ymin:ymax, xmin:xmax]
#                         # rgb_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#                         faces = face_model.get(roi)
#                         if faces:
#                             frame2 = recognize_faces(roi, faces, rtsp_link, area_name, similarity_threshold=0.3)
#                             frame[ymin:ymax, xmin:xmax] = frame2
#                         # Update Trajectories
#                         trajectory_points.setdefault(track_id, []).append(mapped_point)
#                         for i in range(1, len(trajectory_points[track_id])):
#                             draw_dotted_line(bg_image, trajectory_points[track_id][i - 1], trajectory_points[track_id][i], color)

#                         # Draw Bounding Box
#                         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#                         cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

#                         # Highlight on Background
#                         cv2.circle(bg_image, mapped_point, 5, color, -1)
#                         cv2.putText(bg_image, str(track_id), mapped_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2) 
                    
#                         # for bbox in results:
#                         #     x1, y1, x2, y2 = bbox[:4]
#                         #     # Crop the frame to the ROI for face recognition
#                         #     roi = frame[y1:y2, x1:x2]  # Crop the detected area (person)
                            
#                         #     # Perform face recognition on the cropped ROI (the detected person area)
#                         #     frame = recognize_faces(roi, rtsp_link, area_name, similarity_threshold=0.3)
                            
#                         #     # After face recognition, you may want to place the recognized face back in the frame (optional)
#                         #     # For example, if `recognize_faces` returns the processed ROI:
#                         #     frame[y1:y2, x1:x2] = frame
                    
#                     # combined_frame = np.hstack((cv2.resize(bg_image, OUTPUT_SIZE), cv2.resize(frame, OUTPUT_SIZE)))
#                     ret, buffer = cv2.imencode('.jpg', frame)
#                     if ret:
#                         frame_bytes = buffer.tobytes()
#                         try:
#                             yield (b'--frame\r\n'
#                                     b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#                         except Exception as e:
#                             print(f"Error sending frame data: {e}")
#                             break
#             except Exception as e:
#                 print(f"Error processing frame: {e}")
#     finally:
#         fresh_frame.release()
#         print("Released video capture resource")

import json
import datetime

tracking_time = {}

def generate_frames(rtsp_link, yolo_model):
    print("Inside generate_frames")
    if rtsp_link == "webcam":
        rtsp_link = 0
        area_name = "IT_room"
    else:
        area_name = get_area_by_rtsp_url(rtsp_link)
    
    fresh_frame = AsyncFrameCapture(rtsp_link)
    trajectory_points = {}
    detected_objects = {}
    
    try:
        while True:
            try:
                frame = fresh_frame.get_frame()
                if frame is not None:
                    # YOLO Predictions
                    detections = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
                    results = [
                        [int(data[0]), int(data[1]), int(data[2]), int(data[3]), data[4], int(data[5])]
                        for data in detections.boxes.data.tolist()
                        if int(data[5]) == 0 and data[4] >= CONFIDENCE_THRESHOLD
                    ]

                    # Tracker Update
                    detection_tensor = torch.tensor(
                        results, dtype=torch.float32, device=DEVICE
                    ) if results else torch.empty((0, 6), dtype=torch.float32, device=DEVICE)
                    tracks = tracker.update(detection_tensor, frame)

                    # Process Tracks
                    for track in tracks:
                        if isinstance(track, np.ndarray):
                            ltrb, track_id = track[:4], int(track[4])
                        else:
                            ltrb, track_id = track.to_ltrb(), track.track_id

                        xmin, ymin, xmax, ymax = map(int, ltrb)
                        x, y = (xmin + xmax) // 2, (ymin + ymax) // 2
                        mapped_point = transform_to_image(x, y, h_matrix)
                        color = get_unique_color(track_id)

                        # Initialize or update tracking time
                        current_time = datetime.datetime.now()
                        if track_id not in tracking_time:
                            tracking_time[track_id] = {'start_time': current_time, 'end_time': current_time}
                        else:
                            tracking_time[track_id]['end_time'] = current_time

                        # Calculate time period for the current bounding box
                        start_time = tracking_time[track_id]['start_time']
                        end_time = tracking_time[track_id]['end_time']
                        time_period = (end_time - start_time).total_seconds()

                        # Face Recognition
                        roi = frame[ymin:ymax, xmin:xmax]
                        faces = face_model.get(roi)
                        face_info = None
                        if faces:
                            frame2, name, emp_id = recognize_faces(roi, faces, rtsp_link, area_name, similarity_threshold=0.3)
                            frame[ymin:ymax, xmin:xmax] = frame2

                            # Store face details in the dictionary
                            face_info = {
                                'name': name,
                                'employee_id': emp_id
                            }

                        # Update detected_objects dictionary with bbox, face info, and time period
                        detected_objects[track_id] = {
                            'bbox': [xmin, ymin, xmax, ymax],
                            'face_info': face_info,
                            'time_period': time_period
                        }

                        # Draw bounding box and trajectory
                        trajectory_points.setdefault(track_id, []).append(mapped_point)
                        for i in range(1, len(trajectory_points[track_id])):
                            draw_dotted_line(bg_image, trajectory_points[track_id][i - 1], trajectory_points[track_id][i], color)

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                        cv2.circle(bg_image, mapped_point, 5, color, -1)
                        cv2.putText(bg_image, str(track_id), mapped_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                    
                    # Save detected objects and their times to a JSON file
                    with open("detected_objects.json", "w") as f:
                        json.dump(detected_objects, f, indent=4)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {e}")
    finally:
        fresh_frame.stop()
        print("Released video capture resource")



def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def recognize_faces(frame, faces, rtsp_link, area_name, similarity_threshold=0.1):
    global latest_metadata, main_data
    latest_metadata = []
    main_data = []
    recognized = False
    cursor = conn.cursor() 

    face_count = len(faces)

    if face_count==1:
        face = faces[0]
        embedding = face.embedding
        embedding_list = embedding.tolist()
        cursor.execute(
            f"""SELECT employee_id, company, name, department, seat FROM employee_details
             WHERE embedding <=> '{embedding_list}'<.6 order by embedding <=> '{embedding_list}' asc;"""
        )
        results = cursor.fetchone()
        bbox = face['bbox'].astype(int)
        recognized = False  # Initialize the recognized flag here
        # recognized_count = 0
        if results:
            recognized = True
            # recognized_count += 1 
            emp_id, company, name, department, seat = results
            label = f"{name} - {company}" if company else name
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            latest_metadata.append({
                "name": name,
                "company": company,
                "emp_id": emp_id,
                "department": department,
                "area": area_name,
                "bbox": bbox.tolist(),
                "seat": seat
            })
    
            return frame, name, emp_id 
        if not recognized:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            latest_metadata.append({
                "name": "Unknown",
                "area": area_name,
                "bbox": bbox.tolist()
            })

            return frame, "unknown", None
        # unrecognized_count = face_count - recognized_count
    # if latest_metadata:
    #     with Session_() as session:
    #         for data in latest_metadata:
    #             if "emp_id" in data:  # Recognized faces
    #                 record = DetectedLog(
    #                     name=data["name"],
    #                     company=data["company"],
    #                     emp_id=data["emp_id"],
    #                     department=data["department"],
    #                     area=area_name,
    #                     face_image_base64=frame_to_base64(frame),
    #                     detected_at=datetime.datetime.now(kolkata_zone),
    #                     seat=data["seat"]
    #                 )
    #             else:  # Unrecognized faces
    #                 record = DetectedUnknownLog(
    #                     name=data["name"],
    #                     area=area_name,
    #                     frame=None,  # Replace with `frame_to_base64(frame)` if needed
    #                     detected_at=datetime.datetime.now(kolkata_zone)
    #                 )
    #             session.add(record)
    #         session.commit()

        # with Session_() as session:
        #     # Check if there's an existing record for this area
        #     existing_area = session.query(AreaHeadcount).filter(AreaHeadcount.area == area_name).first()
            
        #     if existing_area:
        #         # If the record exists, update the current headcount with the new face count
        #         existing_area.current_headcount = face_count
        #         existing_area.recognized_count = recognized_count
        #         existing_area.unrecognized_count = unrecognized_count
        #     else:
        #         # If the record doesn't exist, insert a new one with the face count
        #         print("I am hitted")
        #         new_area = AreaHeadcount(
        #             area=area_name, 
        #             current_headcount=face_count, 
        #             recognized_count=recognized_count, 
        #             unrecognized_count=unrecognized_count
        #         )

        #         session.add(new_area)
            
            # session.commit()

    # return frame


@app.get("/")
async def hello_world():
    return PlainTextResponse("Hello, World!")

@app.get("/video")
async def video(yolo_model=yolo_model, rtsp_link: str = Query(..., description="RTSP link to stream video")):
    try:
        return StreamingResponse(generate_frames(rtsp_link, yolo_model), media_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video streaming: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
