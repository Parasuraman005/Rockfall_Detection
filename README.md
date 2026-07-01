# Rock Fall Detection & Nearby Alert System

## Overview
This project is a **real‑time rock‑fall detection and human detection** web application built with **Streamlit**, **OpenCV**, and **geopy**. It leverages computer‑vision techniques to:

1. Capture video streams from one or multiple cameras (local device indices or RTSP URLs).
2. Detect humans using OpenCV’s HOG + SVM detector and ignore them from rock‑fall analysis.
3. Track moving objects (potential rocks) across frames, compute vertical speed, and trigger a *fall* alert when:
   - The object reaches the upper 20 % of the frame (near the top).
   - Its vertical speed exceeds a configurable threshold.
4. When a fall is detected, the app finds nearby personnel (mock database) within a 3 km radius and displays an alert with their contact details.

The UI is built with Streamlit, offering an intuitive sidebar for camera configuration, detection thresholds, and GPS location of the camera.

---

## Repository Structure
```
Rockfall_Detection/
├─ rock_fall_dashboard.py   # Main Streamlit application
├─ requirements.txt          # Python dependencies
├─ process.txt               # Basic installation commands (pip install ...)
└─ README.md                # This documentation (you are reading it now)
```

---

## Setup & Installation
1. **Clone the repository** (already done) and navigate to the project folder.
2. **Install dependencies** – you can either follow `process.txt` or install from `requirements.txt`:
   ```bash
   # Using process.txt instructions
   pip install streamlit
   pip install opencv-python-headless
   pip install numpy
   pip install matplotlib
   # Additional dependencies required by the app
   pip install geopy
   ```
   Or simply:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**:
   ```bash
   streamlit run rock_fall_dashboard.py
   ```
   The Streamlit interface will open in your browser.

**About how to run it**: After installing dependencies, execute `streamlit run rock_fall_dashboard.py` from the project root.

---

## Usage Guide
1. **Camera Sources** – Input a comma‑separated list of camera indices (e.g., `0,1`) or RTSP URLs in the sidebar.
2. **Detection Settings** – Adjust:
   - *Min Object Area*: filters out small contours.
   - *Min Vertical Speed*: defines the speed threshold for a rock‑fall.
3. **Camera GPS Location** – Provide latitude and longitude of the camera to enable nearby‑person alerts.
4. **Monitoring** – The main dashboard shows live video feeds, bounding boxes for detected rocks/humans, and a stats panel summarising total falls, falls per minute, and alerts sent.

---

## Code Walk‑through (Key Sections)
| Section | Description |
|---|---|
| **Imports (lines 1‑8)** | Imports required libraries – OpenCV for CV, Streamlit for UI, `geopy` for distance calculations, and `deque` for a rolling window of fall timestamps. |
| **Sidebar UI (lines 13‑31)** | Collects user input for camera sources, detection thresholds, and GPS location. |
| **Human Detector (lines 39‑42)** | Sets up OpenCV HOG descriptor with a pre‑trained people detector. |
| **Camera Initialization (lines 44‑56)** | Opens each source, handling integer indices vs URL strings, and stores valid captures in `caps`. |
| **Main Loop (lines 81‑180)** | Continuously reads frames, applies background subtraction, finds contours, filters by area, checks for human overlap, tracks objects with simple vertical position tracking, calculates speed, triggers fall alerts, and updates the UI. |
| **Alert Logic (lines 150‑155)** | When a fall is detected, `alert_nearby_people` finds persons within 3 km and displays a warning. This is where you could integrate an SMS/email service. |
| **Statistics (lines 163‑177)** | Shows total falls, falls per minute, and number of nearby people alerted for each camera. |

---

## Extending the Project
- **Real People DB** – Replace the mock `people` list with a database or API.
- **SMS/Email Alerts** – Hook the placeholder `send_sms_alert` to an actual notification service (Twilio, SendGrid, etc.).
- **Model‑Based Detection** – Swap the HOG human detector with a deep‑learning model (e.g., YOLO) for higher accuracy.
- **Docker Support** – Add a `Dockerfile` to containerise the app for easy deployment.

---


---

**Happy coding!**
