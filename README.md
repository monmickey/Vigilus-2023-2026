# 🛡️ Vigilus Attention Tracker

Vigilus is a high-performance, AI-driven attention monitoring system designed to help users maintain focus and productivity. It leverages advanced computer vision to track gaze, head position, and emotional state in real-time, providing both a desktop interface and a remote web-based dashboard.

![Vigilus Interface Mockup](https://raw.githubusercontent.com/monmickey/Vigilus/main/assets/banner.png)

## ✨ Core Features

- **Real-time Gaze Tracking**: Monitor where you're looking with high precision.
- **Head Orientation Analysis**: Detect distractions based on head pitch and yaw.
- **Emotion Recognition**: Gain insights into your cognitive state through facial expressions.
- **Multi-Interface Support**: 
  - **Desktop (PySide6)**: Robust control center with glassmorphism UI.
  - **Web (Flask)**: Secure remote monitoring via a dedicated stream.
- **Customizable Modes**: Choose between *Basic*, *Intermediate*, and *Strict* monitoring levels.
- **Face Alerts**: Instant detection of "No Face" or "Multiple Faces" scenarios.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Webcam
- MediaPipe Landmarker model (`models/face_landmarker.task`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/monmickey/Vigilus.git
   cd Vigilus
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r current_requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

## 🛠️ Build from Source

To create a standalone executable for Windows:

```bash
pip install pyinstaller
pyinstaller VigilusAttentionTracker.spec
```

The executable will be generated in the `dist/` directory.

## 📐 Architecture

- **`engine/`**: The core `VigilusEngine` powered by MediaPipe for facial landmarking and attention scoring.
- **`app.py`**: The main PySide6 desktop application with a modern, glassmorphic UI.
- **`streamer.py`**: Handles MJPEG streaming for remote web views.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed with focus for focused creators.*
