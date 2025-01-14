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

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="172.16.200.229",
    port="31937"
)

import threading

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

class KnownFace(Base):
    __tablename__ = 'user_details_new'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    employee_id = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone_number = Column(String(255), nullable=True)
    embedding = Column(Vector, nullable=True)
    face_image_base64 = Column(Text, nullable=True)

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
        print(area)
        return area[0]

def create_fresh_frame_instance(rtsp_link):
    if rtsp_link == "webcam":
        rtsp_link = 0
    cap = cv2.VideoCapture(rtsp_link)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return AsyncFrameCapture

def generate_frames(rtsp_link):
    print("Inside generate_frames")
    if rtsp_link == "webcam":
        rtsp_link = 0
        area_name = "cafeteria"
    else:
        area_name = get_area_by_rtsp_url(rtsp_link)
    fresh_frame = AsyncFrameCapture(rtsp_link)
    try:
        while True:
            try:
                frame = fresh_frame.get_frame()
                if frame is not None:
                    frame = recognize_faces(frame,rtsp_link,area_name, similarity_threshold=0.3)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        try:
                            yield (b'--frame\r\n'
                                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        except Exception as e:
                            print(f"Error sending frame data: {e}")
                            break
            except Exception as e:
                print(f"Error processing frame: {e}")
    finally:
        fresh_frame.release()
        print("Released video capture resource")

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def recognize_faces(frame, rtsp_link, area_name, similarity_threshold=0.1):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_model.get(rgb_frame)
    global latest_metadata, main_data
    latest_metadata = []
    main_data = []
    cursor = conn.cursor()

    if not faces:  # If no faces are detected, we can directly return the frame
        return frame

    face_count = len(faces)

    for face in faces:
        embedding = face.embedding
        embedding_list = embedding.tolist()
        cursor.execute(
            f"""SELECT employee_id, company, name, department, seat FROM user_details_new
             WHERE embedding <=> '{embedding_list}'<.6 order by embedding <=> '{embedding_list}' asc;"""
        )
        results = cursor.fetchone()
        bbox = face['bbox'].astype(int)
        recognized = False  # Initialize the recognized flag here
        recognized_count = 0
        if results:
            recognized_count += 1  
            emp_id, company, name, department, seat = results
            label = f"{name} - {company}" if company else name
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            recognized = True  # Set recognized to True if face is recognized
            latest_metadata.append({
                "name": name,
                "company": company,
                "emp_id": emp_id,
                "department": department,
                "area": area_name,
                "bbox": bbox.tolist(),
                "seat": seat
            })
            recognized = True
        if not recognized:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            latest_metadata.append({
                "name": "Unknown",
                "area": area_name,
                "bbox": bbox.tolist()
            })
        unrecognized_count = face_count - recognized_count
    if latest_metadata:
        with Session_() as session:
            for data in latest_metadata:
                if "emp_id" in data:  # Recognized faces
                    record = DetectedLog(
                        name=data["name"],
                        company=data["company"],
                        emp_id=data["emp_id"],
                        department=data["department"],
                        area=area_name,
                        face_image_base64=frame_to_base64(frame),
                        detected_at=datetime.datetime.now(),
                        seat=data["seat"]
                    )
                else:  # Unrecognized faces
                    record = DetectedUnknownLog(
                        name=data["name"],
                        area=area_name,
                        frame=None,  # Replace with `frame_to_base64(frame)` if needed
                        detected_at=datetime.datetime.now()
                    )
                session.add(record)
            session.commit()

        with Session_() as session:
            # Check if there's an existing record for this area
            existing_area = session.query(AreaHeadcount).filter(AreaHeadcount.area == area_name).first()
            
            if existing_area:
                # If the record exists, update the current headcount with the new face count
                existing_area.current_headcount = face_count
                existing_area.recognized_count = recognized_count
                existing_area.unrecognized_count = unrecognized_count
            else:
                # If the record doesn't exist, insert a new one with the face count
                print("I am hitted")
                new_area = AreaHeadcount(
                    area=area_name, 
                    current_headcount=face_count, 
                    recognized_count=recognized_count, 
                    unrecognized_count=unrecognized_count
                )

                session.add(new_area)
            
            session.commit()

    return frame


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
