import cv2
import time
import numpy as np
import streamlit as st
from geopy.distance import geodesic
from collections import deque
import matplotlib.pyplot as plt

# --- Streamlit UI ---
st.set_page_config(page_title="🪨 Rock Fall & Human Detection", layout="wide")
st.title("🪨 Rock Fall Detection & Nearby Alert System")

# Sidebar: Camera settings
sources_input = st.sidebar.text_area(
    "Enter camera indices or URLs (comma separated)",
    value="0",
    help="Example: 0,1,rtsp://192.168.1.10:554/stream"
)

sources = [s.strip() for s in sources_input.split(",") if s.strip()]

# Sidebar: Detection thresholds
st.sidebar.header("Detection Settings")
min_area = st.sidebar.slider("Min Object Area", 500, 5000, 1500, step=100)
speed_thresh = st.sidebar.slider("Min Vertical Speed (px/sec)", 5, 50, 15)

# Sidebar: Camera location
st.sidebar.header("Camera GPS Location")
camera_lat = st.sidebar.number_input("Camera Latitude", value=28.6139)
camera_lon = st.sidebar.number_input("Camera Longitude", value=77.2090)

# Nearby people (mock DB)
people = [
    {"name": "Worker 1", "lat": 28.6150, "lon": 77.2080, "phone": "+911234567890"},
    {"name": "Engineer 2", "lat": 28.6200, "lon": 77.2000, "phone": "+919876543210"},
    {"name": "Supervisor", "lat": 28.5800, "lon": 77.2500, "phone": "+919999999999"},
]

# Setup human detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Setup camera captures
caps = {}
for src in sources:
    try:
        idx = int(src)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(src)
    if cap.isOpened():
        caps[src] = cap
    else:
        st.warning(f"❌ Cannot open source: {src}")

if not caps:
    st.stop()

# Setup layout
cols = st.columns(len(caps))
video_placeholders = {src: cols[i].empty() for i, src in enumerate(caps)}
stats_placeholders = {src: cols[i].empty() for i, src in enumerate(caps)}

# Object trackers and stats per camera
fall_counts = {src: 0 for src in caps}
fall_times = {src: deque(maxlen=100) for src in caps}
tracks = {src: {} for src in caps}
next_ids = {src: 0 for src in caps}
start_times = {src: time.time() for src in caps}

# Helper: find nearby people
def alert_nearby_people(camera_location, people_list, radius_km=3):
    nearby = []
    for person in people_list:
        dist = geodesic(camera_location, (person["lat"], person["lon"])).km
        if dist <= radius_km:
            nearby.append(person)
    return nearby

# --- Main loop ---
while True:
    for src, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            video_placeholders[src].text("No feed")
            continue

        frame = cv2.resize(frame, (640, 480))
        fgmask = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40).apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect humans
        human_rects, _ = hog.detectMultiScale(frame, winStride=(8, 8))
        humans = [tuple(r) for r in human_rects]

        falling_now = False
        new_tracks = {}
        timestamp = time.time()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cy = y + h // 2

            # Human check
            is_human = False
            for (hx, hy, hw, hh) in humans:
                if (x < hx + hw and x + w > hx and y < hy + hh and y + h > hy):
                    is_human = True
                    break

            if is_human:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "👤 Human", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                continue

            # Rock tracking
            matched_id = None
            for tid, (prev_y, prev_t) in tracks[src].items():
                if abs(prev_y - cy) < 80:
                    matched_id = tid
                    break

            if matched_id is None:
                matched_id = next_ids[src]
                next_ids[src] += 1

            new_tracks[matched_id] = (cy, timestamp)

            # Falling detection
            if matched_id in tracks[src]:
                dy = cy - tracks[src][matched_id][0]
                dt = timestamp - tracks[src][matched_id][1]
                speed = dy / dt if dt > 0 else 0

                if cy < frame.shape[0] * 0.2 and speed > speed_thresh:
                    falling_now = True
                    fall_counts[src] += 1
                    fall_times[src].append(timestamp)
                    cv2.putText(frame, "⚠ ROCK FALLING!", (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "🪨 Rock", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Alert nearby people
                    camera_location = (camera_lat, camera_lon)
                    nearby_people = alert_nearby_people(camera_location, people)
                    for person in nearby_people:
                        st.warning(f"🚨 ALERT: {person['name']} is within 3km of {src}! 📍 Phone: {person['phone']}")
                        # Optional: call send_sms_alert(person) here

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Rock", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        tracks[src] = new_tracks

        # Stats
        rate = len([t for t in fall_times[src] if timestamp - t <= 60])
        elapsed = (timestamp - start_times[src]) / 60
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholders[src].image(frame_rgb, channels="RGB", caption=f"Source: {src}")

        stats_placeholders[src].markdown(f"""
        <div style='padding:1em;border:1px solid #ddd;border-radius:8px;'>
        <h4>📊 Stats for Camera {src}</h4>
        <ul>
            <li><b>Total Falls:</b> {fall_counts[src]}</li>
            <li><b>Falls/Min:</b> {rate}</li>
            <li><b>Nearby People Alerted:</b> {len(nearby_people) if falling_now else 0}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(0.1)
