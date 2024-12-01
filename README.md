# VigilAI
A real-time computer vision system that leverages facial feature analysis and machine learning to detect driver/operator fatigue and prevent accidents through timely alerts

# Fatigue Detection System

A real-time computer vision system for detecting driver fatigue using facial landmarks, eye tracking, and drowsiness detection.

## Overview

This system monitors a person's face in real-time to detect signs of fatigue by:
- Tracking eye closure patterns
- Calculating Eye Aspect Ratio (EAR)
- Detecting yawning using Mouth Aspect Ratio (MAR)
- Providing real-time audio-visual alerts
- Logging drowsiness events and capturing screenshots

## Project Structure
```
fatigue_detection/
├── data/
│   ├── raw/
│   │   └── CEW Dataset/    # Dataset for validation
│   └── processed/
│       ├── logs/           # Event logs
│       ├── screenshots/    # Drowsiness event captures
│       └── validation/     # Validation results
├── src/
│   └── feature_extraction.py    # Main detection code
├── requirements.txt
└── alert.wav               # Alert sound file
```

## Prerequisites

### Hardware Requirements
- Webcam
- Audio output device
- Minimum 4GB RAM
- CPU with SSE4.1 or higher support

### Software Requirements
- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aarchishya/VigilAI.git
cd fatigue_detection
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Required Packages
```
opencv-python==4.8.0
mediapipe==0.10.0
numpy==1.24.3
scipy==1.11.1
pygame==2.5.0
tqdm==4.65.0
```

## Dataset
For validation, the system uses the CEW (Closed Eyes in the Wild) dataset:
- Download from: https://www2.fpce.uc.pt/~lmv/downloads/databases/CEW_v2/
- Extract to: `data/raw/CEW Dataset/`
- Structure:
  ```
  CEW Dataset/
  ├── Open Eyes/
  └── Closed Eyes/
  ```

## Usage

### Real-time Fatigue Detection
```bash
python src/feature_extraction.py
```

### Dataset Validation
```bash
python src/feature_extraction.py --validate
```

### Controls
- Press 'q' to quit
- Press 'p' to pause/resume
- Press 'm' to toggle metrics display

## Features

### Real-time Detection
- Face mesh detection using MediaPipe
- Eye closure monitoring
- Yawn detection
- Visual alerts on screen
- Audio alerts for drowsiness

### Data Logging
- Timestamps of drowsy events
- EAR and MAR values
- Screenshots of drowsy moments
- Validation metrics

### Validation System
- Dataset-based validation
- Accuracy metrics
- Threshold optimization
- Performance analysis

## Output Files

### Logs
- Location: `data/processed/logs/drowsiness_log.csv`
- Format: CSV with columns:
  - Timestamp
  - Event Type
  - EAR Value
  - Screenshot Path

### Screenshots
- Location: `data/processed/screenshots/`
- Format: JPEG images
- Naming: `drowsy_YYYYMMDD_HHMMSS.jpg`

### Validation Results
- Location: `data/processed/validation/validation_results.txt`
- Contains:
  - Accuracy metrics
  - EAR statistics
  - Suggested thresholds

## Troubleshooting

1. **Camera not detected**
   - Check camera connections
   - Verify camera permissions
   - Try different camera index in code

2. **Audio alerts not working**
   - Check audio device
   - Verify 'alert.wav' exists
   - Check pygame installation

3. **High CPU usage**
   - Lower camera resolution
   - Increase frame skip
   - Close background applications

