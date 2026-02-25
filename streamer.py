
import threading
import cv2
import socket
import time
from flask import Flask, Response, render_template_string

class Streamer:
    def __init__(self):
        self.app = Flask(__name__)
        self.frame_bytes = None
        self.raw_frame = None
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.ip_address = self.get_ip_address()
        self.hostname = socket.gethostname()
        self.port = 5000
        self.running = True

        # Routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/metrics', 'metrics', self.metrics_feed)
        self.metrics = {}

        # Start background encoder
        self.encoder_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self.encoder_thread.start()

    def get_ip_address(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def update_frame(self, frame):
        # Quick copy to avoid blocking main thread with encoding
        if frame is None:
            return
        with self.lock:
            self.raw_frame = frame.copy()
        self.frame_event.set()

    def update_metrics(self, data):
        with self.lock:
            self.metrics = data

    def _encoder_loop(self):
        while self.running:
            self.frame_event.wait(timeout=1.0)
            self.frame_event.clear()
            
            frame_to_encode = None
            with self.lock:
                if self.raw_frame is not None:
                    frame_to_encode = self.raw_frame
                    self.raw_frame = None # Consume it
            
            if frame_to_encode is not None:
                # Optimized JPEG quality (75 is standard for web, significantly faster than 85)
                ret, buffer = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    with self.lock:
                        self.frame_bytes = buffer.tobytes()

    def generate(self):
        while True:
            current_bytes = None
            with self.lock:
                current_bytes = self.frame_bytes
            
            if current_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_bytes + b'\r\n')
            time.sleep(0.03) # Limit stream FPS slightly to save bandwidth


    def video_feed(self):
        return Response(self.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def metrics_feed(self):
        import json
        with self.lock:
            return self.app.response_class(
                response=json.dumps(self.metrics),
                mimetype='application/json'
            )

    def index(self):
        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>VIGILUS — Remote Monitor</title>
                <link href="https://fonts.googleapis.com/css2?family=Jura:wght@400;600;700&family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
                <style>
                    :root {
                        --bg-color: #121212;
                        --card-bg: #1E1E1E;
                        --accent-green: #00E676;
                        --accent-red: #FF5252;
                        --accent-blue: #00B0FF;
                        --text-primary: #FFFFFF;
                        --text-secondary: #AAAAAA;
                    }

                    body { 
                        background-color: var(--bg-color); 
                        color: var(--text-primary); 
                        font-family: 'Roboto', sans-serif; 
                        margin: 0; 
                        padding: 20px; 
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }

                    h1 { 
                        font-family: 'Jura', sans-serif;
                        font-weight: 700; 
                        letter-spacing: 4px;
                        margin-bottom: 5px; 
                        color: var(--accent-green);
                        text-shadow: 0 0 10px rgba(0, 230, 118, 0.4);
                    }

                    h2 { 
                        margin-top: 0; 
                        font-size: 14px; 
                        color: var(--text-secondary); 
                        letter-spacing: 1px; 
                        margin-bottom: 30px; 
                        font-weight: 300;
                    }

                    /* Video Container */
                    .video-container {
                        position: relative;
                        padding: 5px;
                        background: linear-gradient(45deg, #333, #111);
                        border-radius: 16px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
                        margin-bottom: 30px;
                    }

                    img { 
                        display: block;
                        max-width: 100%; 
                        width: 800px;
                        border-radius: 12px; 
                        border: 1px solid #333;
                    }

                    /* Status Indicators */
                    .status-dot {
                        position: absolute;
                        top: 15px;
                        right: 15px;
                        width: 12px;
                        height: 12px;
                        background: var(--accent-red);
                        border-radius: 50%;
                        box-shadow: 0 0 10px var(--accent-red);
                        animation: pulse 2s infinite;
                    }
                    
                    @keyframes pulse {
                        0% { opacity: 1; transform: scale(1); }
                        50% { opacity: 0.5; transform: scale(1.2); }
                        100% { opacity: 1; transform: scale(1); }
                    }

                    .status-dot.active {
                        background: var(--accent-green);
                        box-shadow: 0 0 10px var(--accent-green);
                    }

                    /* Metrics Grid */
                    .metrics-container { 
                        display: grid; 
                        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); 
                        gap: 20px; 
                        width: 100%;
                        max-width: 800px; 
                    }

                    .card {
                        background: var(--card-bg);
                        padding: 20px;
                        border-radius: 12px;
                        text-align: center;
                        border: 1px solid #333;
                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                    }

                    .card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 5px 15px rgba(0,0,0,0.4);
                        border-color: #444;
                    }

                    .label { 
                        font-family: 'Jura', sans-serif;
                        font-size: 12px; 
                        color: var(--text-secondary); 
                        text-transform: uppercase; 
                        letter-spacing: 2px; 
                        margin-bottom: 10px; 
                    }

                    .value { 
                        font-size: 20px; 
                        font-weight: 700; 
                    }

                    /* Colors */
                    .good { color: var(--accent-green); text-shadow: 0 0 8px rgba(0,230,118,0.2); }
                    .warn { color: #FFA726; text-shadow: 0 0 8px rgba(255,167,38,0.2); }
                    .bad { color: var(--accent-red); text-shadow: 0 0 8px rgba(255,82,82,0.2); }

                    .toast-notification {
                        position: fixed;
                        top: 20px;
                        left: 50%;
                        transform: translateX(-50%);
                        background: rgba(255, 82, 82, 0.9);
                        color: white;
                        padding: 12px 24px;
                        border-radius: 50px;
                        font-family: 'Jura', sans-serif;
                        font-weight: bold;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
                        display: none;
                        z-index: 10000;
                        border: 2px solid white;
                        animation: slideDown 0.3s ease;
                    }

                    @keyframes slideDown {
                        from { top: -50px; opacity: 0; }
                        to { top: 20px; opacity: 1; }
                    }

                    /* Footer */
                    .footer {
                        margin-top: 40px;
                        color: #444;
                        font-size: 11px;
                        letter-spacing: 1px;
                    }
                </style>
                <script>
                    function updateMetrics() {
                        fetch('/metrics')
                            .then(response => response.json())
                            .then(data => {
                                if (!data.emotion) return;
                                
                                // Check for Alert
                                const toast = document.getElementById('toast');
                                const statusDot = document.getElementById('status-dot');

                                let alertMsg = '';
                                if (data.emotion === 'MULTIPLE FACES') alertMsg = '⚠️ MULTIPLE FACES';
                                else if (data.emotion === 'NO FACE') alertMsg = '⚠️ FACE NOT DETECTED';
                                else if (data.mode === 'Strict' && (data.attention === 'DISTRACTED' || data.cognitive === 'DISTRACTED')) {
                                    alertMsg = '⚠️ DISTRACTION DETECTED';
                                }

                                if (alertMsg) {
                                    toast.innerText = alertMsg;
                                    toast.style.display = 'block';
                                    statusDot.classList.remove('active');
                                } else {
                                    toast.style.display = 'none';
                                    statusDot.classList.add('active');
                                }

                                updateCard('emotion', data.emotion.replace('Emotion: ', ''));
                                updateCard('gaze', data.gaze);
                                updateCard('attention', data.attention);
                                updateCard('cognitive', data.cognitive);
                            });
                    }

                    function updateCard(id, value) {
                        const el = document.getElementById(id);
                        el.innerText = value;
                        el.className = 'value ' + getColorClass(value);
                    }

                    function getColorClass(value) {
                        if (value === 'NO FACE' || value === 'ALERT' || value === 'MULTIPLE FACES') return 'bad';
                        if (value === 'ATTENTIVE' || value === 'FOCUSED' || value === 'Center' || value.includes('Happy')) return 'good';
                        if (value === 'DISTRACTED' || value.includes('Surprised')) return 'warn';
                        return 'bad'; // Default or bad states
                    }

                    setInterval(updateMetrics, 500);
                </script>
            </head>
            <body>
                <div id="toast" class="toast-notification"></div>

                <h1>VIGILUS MONITOR</h1>
                <h2>CONNECTED: {{ hostname }}</h2>
                
                <div class="video-container">
                    <div id="status-dot" class="status-dot active"></div>
                    <img src="{{ url_for('video_feed') }}" alt="Live Feed">
                </div>
                
                <div class="metrics-container">
                    <div class="card">
                        <div class="label">Emotion</div>
                        <div class="value" id="emotion">--</div>
                    </div>
                    <div class="card">
                        <div class="label">Gaze</div>
                        <div class="value" id="gaze">--</div>
                    </div>
                    <div class="card">
                        <div class="label">Attention</div>
                        <div class="value" id="attention">--</div>
                    </div>
                    <div class="card">
                        <div class="label">Cognitive</div>
                        <div class="value" id="cognitive">--</div>
                    </div>
                </div>

                <div class="footer">VIGILUS ATTENTION SYSTEM v2.0</div>
            </body>
            </html>
        """, hostname=self.hostname)

    def run(self):
        # Run Flask in a separate thread to avoid blocking the main UI
        kwargs = {'host': '0.0.0.0', 'port': self.port, 'debug': False, 'use_reloader': False}
        t = threading.Thread(target=self.app.run, kwargs=kwargs)
        t.daemon = True
        t.start()
        return f"https://{self.ip_address}:{self.port}"
