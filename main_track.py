from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
import cv2
import insightface
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.orm import declarative_base, sessionmaker
import psycopg2
import base64
import datetime
import numpy as np
from ultralytics import YOLO
import threading
from deep_sort_realtime.deepsort_tracker import DeepSort

app = FastAPI()

# Face detection model
face_model = insightface.app.FaceAnalysis()
face_model.prepare(ctx_id=-1, det_size=(640, 640), det_thresh= .5)

# Person detection model using YOLOv8
person_detection_model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=50)

CONFIDENCE_THRESHOLD = 0.8
WHITE = (255, 255, 255)
OUTPUT_SIZE = (640, 640)

# Areas of interest in the surveillance system
area_data = {
    "cafeteria": {
        "image_path": "/path/to/cafeteria.jpg",
        "homography_matrix_path": "/path/to/cafeteria/homography_matrix_cafeteria.txt"
    },
    "602": {
        "image_path": "/Users/xs496-jassin/Desktop/Surveillance/room.jpg",
        "homography_matrix_path": "/Users/xs496-jassin/Desktop/Surveillance/homography_matrix_cafeteria.txt"
    },
    "webcam": {
        "image_path": "/Users/xs496-jassin/Desktop/Surveillance/room.jpg",
        "homography_matrix_path": "/Users/xs496-jassin/Desktop/Surveillance/homography_matrix_ai_room.txt"
    },
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()

# PostgreSQL connection setup
engine_ = create_engine('postgresql://postgres:postgres@10.0.0.210:5432/xenonai')
Session_ = sessionmaker(bind=engine_)

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="172.16.200.229",
    port="31937"
)

# Threaded frame capture class
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

# Define database models
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

class AreaHeadcount(Base):
    __tablename__ = 'area_headcount'
    id = Column(Integer, primary_key=True)
    area = Column(String(255), nullable=False)
    current_headcount = Column(Integer, nullable=False, default=0)
    recognized_count = Column(Integer, nullable=False, default=0)
    unrecognized_count = Column(Integer, nullable=False, default=0)

def get_area_by_rtsp_url(rtsp_url):
    with Session_() as session:
        area = session.query(Area.name).filter(Area.rtsp == rtsp_url).first()
        return area[0]

def create_fresh_frame_instance(rtsp_link):
    if rtsp_link == "webcam":
        rtsp_link = 0
    cap = cv2.VideoCapture(rtsp_link)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return AsyncFrameCapture

# Function for detecting persons using YOLO
def detect_person(frame, area_image, model, CONFIDENCE_THRESHOLD, h_matrix, trajectory_points):
    # Run the YOLO model on the frame
    detections = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Prepare detection results
    results = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        class_id = int(data[5])
        if class_id != 0 or confidence < CONFIDENCE_THRESHOLD:  # Skip non-person detections
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    # Update the tracker
    tracks = tracker.update_tracks(results, frame=frame)

    # Draw detections and trajectories
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # x, y = (xmin + xmax) // 2, ymax  # Bottom-center point
        center_y = ymin + (ymax - ymin) // 2  # Center of the bounding box
        y = (ymax + center_y) // 2  # Midpoint between the bottom and the center
        x = (xmin + xmax) // 2  

        # Get unique color for the track
        color = get_unique_color(track_id)

        # Map the point to the room image
        mapped_point = transform_to_image(x, y, h_matrix)

        # Update the trajectory for this track
        if track_id not in trajectory_points:
            trajectory_points[track_id] = []
        trajectory_points[track_id].append(mapped_point)

        # Draw the dashed trajectory line
        points = trajectory_points[track_id]
        for i in range(1, len(points)):
            draw_dotted_line(area_image, points[i - 1], points[i], color)

        # Draw bounding box and track ID
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    # Resize images for display
    frame_resized_original = cv2.resize(area_image, OUTPUT_SIZE)
    frame_resized = cv2.resize(frame, OUTPUT_SIZE)

    # Stack frames side by side
    stacked_frame = np.hstack((frame_resized_original, frame_resized))

    return person_boxes

# Function to transform video coordinates to image coordinates using the homography matrix
def transform_to_image(x, y, h_matrix):
    video_coords = np.array([[x, y, 1]]).T  # Homogeneous coordinates
    image_coords = np.dot(h_matrix, video_coords)
    image_coords /= image_coords[2]  # Normalize by z-coordinate
    return int(image_coords[0]), int(image_coords[1])


def draw_dotted_line(image, start_point, end_point, color, radius=3):
    """Draws a line with dots between two points."""
    x1, y1 = start_point
    x2, y2 = end_point
    line_length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    
    # Place dots along the line at regular intervals
    for i in range(0, line_length, radius * 4):  # Adjust spacing by modifying radius * 4
        start = i / line_length
        start_x = int(x1 + start * (x2 - x1))
        start_y = int(y1 + start * (y2 - y1))
        cv2.circle(image, (start_x, start_y), radius, color, -1)

# Function to transform coordinates using the homography matrix
def transform_to_image(x, y, h_matrix):
    video_coords = np.array([[x, y, 1]]).T  # Homogeneous coordinates
    image_coords = np.dot(h_matrix, video_coords)
    image_coords /= image_coords[2]  # Normalize by z-coordinate
    return int(image_coords[0]), int(image_coords[1])

# Function to generate a unique color for each track ID
def get_unique_color(track_id):
    try:
        track_id = int(track_id)  # Ensure track_id is an integer
    except ValueError:
        raise TypeError(f"Track ID must be convertible to an integer, but got {track_id}")

    np.random.seed(track_id)  # Seed the random number generator
    color = tuple(np.random.randint(0, 255, 3).tolist())
    return color

# Function to generate frames for video stream
def generate_frames(rtsp_link):
    person_trajectories = {}

    trajectory_points = []
    track_frame = None
    show = True

    if rtsp_link == "webcam":
        rtsp_link = 0
        area_name = "webcam"
    else:
        area_name = get_area_by_rtsp_url(rtsp_link)
    
    # Get corresponding area map and homographic matrix
    area_info = area_data.get(area_name)
    if area_info:
        image_path = area_info["image_path"]
        homography_matrix_path = area_info["homography_matrix_path"]
    else:
        print(f"Area {area_name} not found in the dictionary.")
        return None
    
    h_matrix = np.loadtxt(homography_matrix_path)
    area_image = cv2.imread(image_path)

    fresh_frame = AsyncFrameCapture(rtsp_link)
    try:
        while True:
            try:
                frame = fresh_frame.get_frame()
                if frame is not None:
                    # # Detect persons
                    person_boxes = detect_person(frame, area_image, person_detection_model, CONFIDENCE_THRESHOLD, h_matrix, trajectory_points)
                    if show:
                        for box in person_boxes:
                            x1, y1, x2, y2, bottom_center = box
                            # Draw the rectangle (bounding box) on the frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color and thickness 2
                            # Optionally, add a label above the bounding box
                            label = "Person"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


                    # Implement tracking
                    for person_bbox in person_boxes:
                        track_frame = show_tracking(h_matrix, image, person_bbox)

                        frame = cv2.resize(frame, (640, 640))
                        track_frame = cv2.resize(track_frame, (640, 640))
                        frame2 = cv2.hconcat([frame, track_frame])

                    # Convert the frame to bytes for streaming
                    ret, buffer = cv2.imencode('.jpg', frame2)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        try:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        except Exception as e:
                            print(f"Error sending frame data: {e}")
            except Exception as e:
                print(f"Error processing frame: {e}")
    finally:
        if hasattr(fresh_frame, "release") and callable(fresh_frame.release):
            fresh_frame.release()

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/")
async def hello_world():
    return PlainTextResponse("Hello, World!")

@app.get("/video")
async def video(rtsp_link: str = Query(..., description="RTSP link to stream video")):
    try:
        return StreamingResponse(generate_frames(rtsp_link), media_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video streaming: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
