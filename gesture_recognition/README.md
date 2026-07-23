# 🦴 NoseCheck AI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-nosecheck.onrender.com-2563eb?style=for-the-badge&logo=render)](https://nosecheck.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarks-0f9d58?style=for-the-badge&logo=google)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-Research%20Only-orange?style=for-the-badge)](#license)

> **A smartphone-based screening tool for detecting nasal asymmetry and Deviated Nasal Septum (DNS) using computer vision and symptom assessment.**

🔗 **Try it live → [nosecheck.onrender.com](https://nosecheck.onrender.com)**

---

## 📸 How It Works

Upload a frontal face photo → answer a short symptom questionnaire → get a deviation score with severity classification (Normal / Mild / Moderate / Severe).

```
Photo Upload  →  MediaPipe Face Landmarks  →  Asymmetry Metrics  →  Deviation Score  →  Result
```

---

## ✨ Features

- **Automated face landmark detection** using Google's MediaPipe Face Landmarker
- **4 nasal asymmetry metrics**: lateral deviation, septal angle, nostril asymmetry, bridge alignment
- **Symptom questionnaire** integrated into the scoring pipeline
- **Mobile-friendly web interface** — works from any smartphone browser
- **Research-grade scoring** with Normal / Mild / Moderate / Severe classification
- **Calibration framework** for validating against 3D-printed reference models

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- pip

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/sairamvottikonda-tech/NoseCheckAI.git
cd NoseCheckAI

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-production.txt   # minimal (web app only)
# pip install -r requirements.txt            # full (includes analysis tools)

# 4. Download the MediaPipe model
mkdir -p models
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# 5. Start the server
python run_server.py
```

Open **http://localhost:5001** in your browser.

---

## 🧠 Key Metrics

| Metric | Description |
|---|---|
| **Lateral Deviation** | Distance from nose tip to facial midline |
| **Septal Angle** | Angle of nasal septum from vertical |
| **Nostril Asymmetry** | Left vs. right nostril size/shape differences |
| **Bridge Alignment** | Straightness of the nasal bridge |

These combine into a weighted **Deviation Score** that maps to a severity classification.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | OpenCV, MediaPipe Face Landmarker |
| Web Framework | Flask + Gunicorn |
| Numerical | NumPy |
| Analysis (dev) | Pandas, SciPy, scikit-learn, Matplotlib |
| Deployment | Render (free tier) |

---

## 📁 Project Structure

```
NoseCheckAI/
├── src/
│   ├── app/                   # Flask web application
│   ├── landmark_detection/    # MediaPipe face landmark detection
│   ├── measurement/           # Nasal asymmetry calculations
│   ├── scoring/               # Deviation score algorithm
│   ├── questionnaire/         # Symptom checklist module
│   └── data_management/       # Data storage and retrieval
├── data/
│   ├── calibration_models/    # Known deviations of 3D reference models
│   └── results/               # Measurement data and scores
├── scripts/                   # Calibration and analysis utilities
├── templates/                 # HTML templates
├── static/                    # CSS, JS assets
├── models/                    # MediaPipe model (downloaded at build)
├── docs/                      # Research notes and documentation
├── render.yaml                # Render deployment config
├── requirements-production.txt
└── requirements.txt
```

---

## 🐍 Python API

```python
from src.landmark_detection.detector import detect_landmarks
from src.measurement.asymmetry_calculator import calculate
from src.scoring.score_calculator import calculate_score

image_path = "path/to/face_photo.jpg"
landmarks = detect_landmarks(image_path)
if landmarks:
    measurements = calculate(landmarks)
    result = calculate_score(measurements)
    print(f"Deviation Score: {result['deviation_score']}")
    print(f"Classification:  {result['classification']}")
```

---

## ☁️ Deploying to Render

The repo includes a `render.yaml` for one-click deployment:

1. Fork this repo
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your fork — Render will auto-detect `render.yaml`
4. The build command downloads the MediaPipe model and installs deps automatically

---

## 🔬 Research Context

NoseCheck is designed to:
- Validate correlation between calculated asymmetry scores and known deviations (e.g., 3D-printed models, CT data)
- Assess measurement repeatability (target CV < 10%)
- Serve as an educational screening tool for adolescent nasal health awareness

**Expected outcomes:** High r² correlation with ground truth, reliable 4-class severity distinction.

---

## 🛠 Troubleshooting

**Face detection fails** — Use a well-lit frontal photo with the full face visible. Avoid angles, shadows, or obstructions.

**Port 5001 in use** — Change the port in `run_server.py`. (Port 5000 is avoided by default due to macOS AirPlay conflicts.)

**Model download fails** — Download the `.task` file manually using the `curl` command in step 4 above.

**Import errors** — Make sure you're in the project root and the venv is activated.

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is not a medical device and is not intended for clinical diagnosis or treatment decisions. Always consult a qualified healthcare professional for medical advice.

---

## 🤝 Contributing

This is an active research project. For questions, bug reports, or collaboration inquiries, open an Issue or reach out via GitHub.

---

## 📄 License

To be determined based on research institution requirements. All rights reserved pending publication.
