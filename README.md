# WildGuardAI: Real-Time Wildlife Conflict Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv12](https://img.shields.io/badge/YOLOv12n-Detection-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

**WildGuardAI** is an AI-driven real-time wildlife conflict prediction system designed to proactively prevent human–wildlife conflict through intelligent detection, tracking, and threat assessment. Rather than simply alerting on animal presence, WildGuardAI predicts conflict risk by analyzing animal trajectory, proximity, and behavior patterns, enabling early intervention before dangerous encounters occur.

The system integrates state-of-the-art computer vision models with spatial analysis to provide actionable intelligence for forest rangers, agricultural communities, and wildlife conservationists in remote regions.

## Problem Statement

Human–Wildlife Conflict (HWC) is escalating globally due to:
- Deforestation and habitat fragmentation
- Agricultural expansion into wildlife territories
- Growing human settlements at forest edges

**Current Limitations:**
- Traditional systems (camera traps, manual surveys) are **reactive**, documenting incidents only after they occur
- Existing AI models trigger alerts based solely on animal presence, causing frequent false alarms and alert fatigue
- No integration of spatial analysis, movement tracking, or behavioral prediction
- Cloud-dependent solutions are impractical in remote forest areas with limited connectivity

## Key Features

✅ **Real-Time Detection** – Detects 22 wildlife species with 91.2% mAP using YOLOv12n  
✅ **Intelligent Tracking** – BoT-SORT multi-object tracking with consistent identity maintenance  
✅ **Conflict Prediction** – Threat Index (Ω) calculation based on trajectory, proximity, and approach angle  
✅ **Dual-Threshold Alerting** – Ω > 0.8 triggers alert; Ω < 0.3 confirms safety (reduces false alarms)  
✅ **Lightweight Architecture** – 6.3 GFLOPs, 5.11 MB, 2.56M parameters (edge-deployable)  
✅ **REST API Backend** – Flask API with SQLite for scalable integration  
✅ **Interactive Dashboard** – Real-time visualization with trajectories, object counts, and alert history  
✅ **Geofence Monitoring** – Automatic SMS/IoT alerts when animals breach predefined zones  

## System Architecture

```
┌─────────────────────┐
│   INPUT LAYER       │
│ • Video Upload      │
│ • Webcam Stream     │
│ • Image Upload      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   AI INFERENCE ENGINE               │
│ • YOLOv12n Detection                │
│ • BoT-SORT Tracking                 │
│ • Conflict Prediction Engine        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   APPLICATION SERVER                │
│ • Flask REST API                    │
│ • SQLite Database                   │
│ • Alert Generation & Logging        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   PRESENTATION LAYER                │
│ • Detection Dashboard               │
│ • Trajectory Visualization          │
│ • Alert Panel & Logs                │
└─────────────────────────────────────┘
```

## Technical Specifications

### Hardware Requirements
- **Processor:** Intel Core i5 / AMD Ryzen 5 or higher
- **GPU:** NVIDIA T4 (for model training) or compatible CUDA device
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** 50 GB+ SSD for datasets and model weights
- **Input:** CCTV cameras, recorded videos, or live webcam feeds

### Software Stack
- **Language:** Python 3.9+
- **Deep Learning:** PyTorch 2.x, Ultralytics YOLOv12
- **Tracking:** BoT-SORT (Kalman Filter + Hungarian Algorithm)
- **Backend:** Flask REST API
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript
- **Training Environment:** Google Colab with NVIDIA T4 GPU
- **Dataset:** Roboflow Universe – Wild Animals CV Dataset (22 classes)

## Model Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 91.2% |
| mAP@0.50–0.95 | 77.19% |
| Wildlife Species Detected | 22 classes |
| Computational Cost | 6.3 GFLOPs |
| Model Size | 5.11 MB |
| Parameters | 2.56M |
| Training Epochs | 50 |
| Training Hardware | NVIDIA T4 |

## Installation & Setup

### Prerequisites
```bash
# Install Python 3.9 or higher
python --version

# Create virtual environment (recommended)
python -m venv wildguardai_env
source wildguardai_env/bin/activate  # On Windows: wildguardai_env\Scripts\activate
```

### Clone Repository
```bash
git clone https://github.com/SiddardhaShayini/WildGuardAI-Real-Time-Wildlife-Conflict-Prediction.git
cd WildGuardAI-Real-Time-Wildlife-Conflict-Prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
flask>=2.3.0
numpy>=1.24.0
matplotlib>=3.7.0
numpy>=1.24.0
```

### Download Pre-trained Model
```bash
# Download YOLOv12n weights trained on wildlife dataset
# Place best.pt in the models/ directory
```

## Usage

### 1. Run the Flask Application
```bash
python app.py
```
The application will start at `http://localhost:5000`

### 2. Upload Input
- **Video:** Upload recorded video file
- **Webcam:** Enable live camera stream
- **Image:** Upload single image for detection

### 3. Monitor Real-Time Results
- View detection bounding boxes and object IDs
- Observe animal trajectories
- Check Threat Index values
- Monitor alert status and geofence breaches

### 4. Export Logs
- Download session logs as CSV
- Contains timestamps, species, Track IDs, and threat levels

## Core Modules

### Detection Module (YOLOv12n)
- Performs real-time object detection for 22 wildlife species
- Outputs bounding boxes, confidence scores, and class labels
- Lightweight architecture with Localized Area Attention for efficient inference

### Tracking Module (BoT-SORT)
- Assigns unique Track IDs to detected animals
- Uses Kalman Filter to predict motion across frames
- Employs Hungarian Algorithm for optimal data association
- Maintains identity during occlusions using ReID features

### Conflict Prediction Engine
- Calculates Threat Index (Ω) using:
  - **Time-to-Collision:** Estimated time until animal reaches geofence
  - **Approach Angle:** Direction of animal movement relative to human zones
  - **Proximity:** Current distance from protected area
- Alert Logic:
  - **Ω > 0.8:** HIGH RISK – Alert triggered
  - **0.3 < Ω ≤ 0.8:** MEDIUM RISK – Monitor
  - **Ω < 0.3:** SAFE – No alert

### Alert & Notification Module
- Generates alerts when Threat Index exceeds threshold
- Logs events with timestamps, species, and Track IDs
- SMS/IoT integration for remote notifications (future)

### Dashboard (Presentation Layer)
- Real-time visualization of detections
- Movement trajectory display
- Object count and FPS metrics
- Alert history and event logs
- Session export functionality

## Methodology

### 1. Data Acquisition & Preprocessing
- Dataset sourced from Roboflow Universe (Wild Animals CV Dataset)
- Images resized to 640×640 resolution
- Normalization and augmentation applied for robust training

### 2. Model Training
- **Framework:** PyTorch with Ultralytics YOLOv12
- **Hardware:** NVIDIA T4 GPU (Google Colab)
- **Duration:** 50 epochs
- **Optimization:** Adam optimizer with learning rate scheduling
- **Output:** Best model saved as `best.pt`

### 3. Detection & Tracking Pipeline
- YOLOv12n detects wildlife in each video frame
- BoT-SORT assigns unique IDs and maintains consistent tracking
- Kalman Filter predicts motion in subsequent frames
- Occlusion handling through ReID feature comparison

### 4. Risk Evaluation
- Threat Index (Ω) computed for each tracked animal
- Geofence proximity monitored continuously
- Alert generated when Ω > 0.8 within protected zone

### 5. Alert & Visualization
- Alerts processed through Flask REST API
- Events logged in SQLite database
- Results displayed on interactive dashboard

## Future Enhancements

🚀 **Edge Deployment** – Deploy on NVIDIA Jetson Nano for offline operation in remote areas  
🚀 **IoT Integration** – Real-time SMS/push notifications to forest rangers  
🚀 **Multi-Camera Fusion** – Coordinate detections across multiple camera feeds  
🚀 **Behavior Analysis** – Advanced animal behavior classification (hunting, migrating, etc.)  
🚀 **Species-Specific Risk Profiles** – Customized threat assessment per species  
🚀 **Mobile Application** – Native mobile apps for field personnel  

## Project Structure

```
WildGuardAI/
├── models/
│   ├── best.pt                    # Pre-trained YOLOv12n weights
│   └── tracking_config.yaml       # BoT-SORT configuration
├── data/
│   ├── videos/                    # Sample video files
│   ├── images/                    # Sample images
│   └── dataset/                   # Training dataset (Roboflow)
├── app.py                         # Flask application
├── inference.py                   # Detection & tracking engine
├── conflict_engine.py             # Threat Index calculation
├── database.py                    # SQLite operations
├── templates/
│   ├── index.html                # Main dashboard
│   ├── upload.html               # Upload interface
│   └── results.html              # Results display
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── requirements.txt              # Python dependencies
├── config.yaml                   # Configuration parameters
└── README.md                      # This file
```

## Performance Comparison

| Model | mAP@0.5 | Parameters | GFLOPs | Size (MB) | Advantage |
|-------|---------|------------|--------|-----------|-----------|
| YOLOv12n (WildGuardAI) | 91.2% | 2.56M | 6.3 | 5.11 | **Lightweight, Edge-Ready** |
| RT-DETR | 90.5% | 32M+ | 35+ | 110+ | Higher accuracy but resource-heavy |
| YOLOv8n | 89.8% | 3.2M | 8.7 | 6.3 | Baseline comparison |

## Configuration

Edit `config.yaml` to customize:

```yaml
# Detection Settings
CONFIDENCE_THRESHOLD: 0.5
NMS_THRESHOLD: 0.45

# Tracking Settings
MAX_AGE: 30                    # Maximum frames without detection
MIN_HITS: 3                    # Minimum detections to confirm track

# Threat Assessment
THREAT_THRESHOLD_HIGH: 0.8     # Alert trigger
THREAT_THRESHOLD_LOW: 0.3      # Safe confirmation
TIME_TO_COLLISION_WEIGHT: 0.5
PROXIMITY_WEIGHT: 0.3
APPROACH_ANGLE_WEIGHT: 0.2

# Geofence Settings
GEOFENCE_COORDINATES: [[x1, y1], [x2, y2], ...]
```

## Results & Discussion

WildGuardAI demonstrates superior performance in real-world wildlife monitoring scenarios:

- **Accuracy:** 91.2% mAP across 22 species, handling occlusion and camouflage effectively
- **Efficiency:** ~92% fewer parameters than competing models while maintaining comparable accuracy
- **Practical Value:** Distinction between normal movement and threat trajectories reduces false alarms
- **Deployability:** Lightweight footprint enables edge deployment on resource-constrained devices
- **Scalability:** Modular architecture supports integration with multiple camera feeds and alert systems

## Limitations & Scope

**Current Scope:**
- Conceptual framework and YOLOv12n training pipeline development
- Simulation-based threat assessment using synthetic trajectories
- Lab-based testing with recorded video and curated datasets

**Not Included (Future Work):**
- Large-scale field deployment in actual forest environments
- Real-time SMS/IoT integration (framework prepared)
- Jetson Nano edge deployment (architecture optimized for this)
- Multi-camera coordination and sensor fusion

## Contributing

This is an academic research project. If you're interested in contributing or extending this work, please feel free to fork the repository and submit pull requests.

## Citation

If you use WildGuardAI in your research, please cite as:

```bibtex
@project{wildguardai2025,
  title={WildGuardAI: Real-Time Wildlife Conflict Prediction},
  author={Siddardha, S. and Others},
  year={2025},
  url={https://github.com/SiddardhaShayini/WildGuardAI-Real-Time-Wildlife-Conflict-Prediction}
}
```

## References

[1] Mughal, A. B., et al. (2025). Deep Learning for Dynamic Wildlife Monitoring: A Real-Time Approach. IEEE Access, 13.

[2] Ke, W., et al. (2025). IECA-YOLOv7: Lightweight Model with Enhanced Attention for Aerial Wildlife Detection. Scientific Reports, 15.

[3] Dave, B., et al. (2023). Wild Animal Detection Using YOLOv8. Procedia Computer Science, 230, 100–111.

[4] Dertien, J. S., et al. (2023). Mitigating Human–Wildlife Conflict Using Real-Time Camera-Based Alert System. BioScience, 2023.

[5] Pathak, A. A., et al. (2024). Leveraging AI to Reduce Human–Animal Conflict in India. Communications of ASIS, 55, 336–350.

[6] Wang, C.-Y., et al. (2024). YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information. arXiv.

[7] Deepika, R., et al. (2024). Agro Guard Edge AI — IoT Framework for Wildlife Intrusion Detection. J. Electrical Eng. & Automation, 6(4).

[8] Feng, L., et al. (2021). Action Recognition Using a Spatial-Temporal Network for Wild Felines. Animals, 11(2), 485.

[9] Chen, H., et al. (2021). Robust Visual Tracking with Reliable Object Information and Kalman Filter. Sensors, 21(3), 889.

[10] Yoon, K., et al. (2019). Data Association for Multi-Object Tracking via Deep Neural Networks. Sensors, 19(3), 559.

[11] Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. CVPR.

[12] Dataset: Wild Animals Computer Vision Dataset. Roboflow Universe.

## Acknowledgments

This project was developed as an academic research initiative. We acknowledge the Roboflow community for providing the Wild Animals dataset and the open-source communities for PyTorch, YOLOv12, and BoT-SORT.

## License

This project is licensed under the MIT License – see LICENSE file for details.

---

**For questions or suggestions, please open an issue or contact the project maintainers.**

**Last Updated:** March 2025
