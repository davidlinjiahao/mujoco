"""Web-based control panel for recording using Flask"""

import threading

# Flask for web-based control panel
try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class RecordingControlPanel:
    """Web-based control panel for recording using Flask"""
    
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recording Controls</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                max-width: 400px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #333;
                margin: 0 0 20px 0;
                font-size: 24px;
            }
            .status {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .status.ready { background: #e0e0e0; color: #666; }
            .status.recording { background: #ffebee; color: #c62828; }
            .status.replaying { background: #e3f2fd; color: #1565c0; }
            .stats {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
                padding: 15px;
                background: #f8f8f8;
                border-radius: 8px;
            }
            .stat { text-align: center; }
            .stat-label { font-size: 12px; color: #666; }
            .stat-value { font-size: 24px; font-weight: bold; color: #333; }
            button {
                width: 100%;
                padding: 15px;
                margin: 10px 0;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: opacity 0.2s;
            }
            button:hover { opacity: 0.8; }
            button:active { opacity: 0.6; }
            .btn-start { background: #4CAF50; color: white; }
            .btn-stop { background: #f44336; color: white; }
            .btn-reset { background: #2196F3; color: white; }
            .btn-save { background: #FF9800; color: white; }
            .btn-discard { background: #9E9E9E; color: white; }
            hr { border: none; border-top: 1px solid #e0e0e0; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎮 Recording Controls</h1>
            
            <div id="status" class="status ready">⚪ Ready</div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Episodes</div>
                    <div class="stat-value" id="episodes">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Frames</div>
                    <div class="stat-value" id="frames">0</div>
                </div>
            </div>
            
            <hr>
            
            <button id="recordBtn" class="btn-start" onclick="toggleRecording()">▶️ Start Recording</button>
            <button class="btn-reset" onclick="resetCube()">🔄 Reset Cube</button>
            <button class="btn-reset" onclick="randomizeCube()">🎲 Randomize Cube Position</button>
            
            <hr>
            
            <button class="btn-save" onclick="saveEpisode()">💾 Save Episode</button>
            <button class="btn-discard" onclick="discardEpisode()">🗑️ Discard Episode</button>
            
            <hr>
            
            <button class="btn-reset" onclick="replayEpisode()">🎬 Replay Last Episode</button>
        </div>
        
        <script>
            function updateStatus() {
                fetch('/status')
                    .then(r => r.json())
                    .then(data => {
                        // Update status
                        const statusEl = document.getElementById('status');
                        const recordBtn = document.getElementById('recordBtn');
                        if (data.replaying) {
                            statusEl.textContent = '🎬 REPLAYING';
                            statusEl.className = 'status replaying';
                        } else if (data.recording) {
                            statusEl.textContent = '🔴 RECORDING';
                            statusEl.className = 'status recording';
                            recordBtn.textContent = '⏹️ Stop Recording';
                            recordBtn.className = 'btn-stop';
                        } else {
                            statusEl.textContent = '⚪ Ready';
                            statusEl.className = 'status ready';
                            recordBtn.textContent = '▶️ Start Recording';
                            recordBtn.className = 'btn-start';
                        }
                        
                        // Update counters
                        document.getElementById('episodes').textContent = data.episodes;
                        document.getElementById('frames').textContent = data.frames;
                    });
            }
            
            function toggleRecording() {
                fetch('/toggle_recording', {method: 'POST'});
            }
            
            function resetCube() {
                fetch('/reset_cube', {method: 'POST'});
            }
            
            function randomizeCube() {
                fetch('/randomize_cube', {method: 'POST'});
            }
            
            function saveEpisode() {
                fetch('/save_episode', {method: 'POST'});
            }
            
            function discardEpisode() {
                fetch('/discard_episode', {method: 'POST'});
            }
            
            function replayEpisode() {
                fetch('/replay_episode', {method: 'POST'});
            }
            
            // Update every 100ms
            setInterval(updateStatus, 100);
            updateStatus();
        </script>
    </body>
    </html>
    """
    
    def __init__(self, recorder, controls, model, data, port=5001):
        self.recorder = recorder
        self.controls = controls
        self.model = model
        self.data = data
        self.port = port
        self.app = None
        self.running = False
        
    def start(self):
        """Start the Flask web server in a separate thread"""
        if not FLASK_AVAILABLE:
            print("⚠️  Flask not installed - control panel disabled")
            print("   Install with: pip install flask")
            return
        
        self.running = True
        self.app = Flask(__name__)
        
        # Disable Flask logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.logger.disabled = True
        
        # Set up routes
        @self.app.route('/')
        def index():
            return render_template_string(self.HTML_TEMPLATE)
        
        @self.app.route('/status')
        def status():
            return jsonify({
                'recording': self.recorder.recording,
                'replaying': self.controls.replaying,
                'episodes': self.recorder.episode_count,
                'frames': len(self.recorder.trajectory)
            })
        
        @self.app.route('/toggle_recording', methods=['POST'])
        def toggle_recording():
            if self.recorder.recording:
                self.controls.stop_recording = True
            else:
                self.controls.start_recording = True
            return jsonify({'success': True})
        
        @self.app.route('/reset_cube', methods=['POST'])
        def reset_cube():
            self.controls.reset_cube = True
            return jsonify({'success': True})
        
        @self.app.route('/randomize_cube', methods=['POST'])
        def randomize_cube_route():
            self.controls.randomize_cube = True
            return jsonify({'success': True})
        
        @self.app.route('/save_episode', methods=['POST'])
        def save_episode():
            self.controls.save_recording = True
            return jsonify({'success': True})
        
        @self.app.route('/discard_episode', methods=['POST'])
        def discard_episode():
            self.controls.discard_recording = True
            return jsonify({'success': True})
        
        @self.app.route('/replay_episode', methods=['POST'])
        def replay_episode():
            self.controls.replay_episode = True
            return jsonify({'success': True})
        
        # Start server in thread
        thread = threading.Thread(
            target=lambda: self.app.run(host='127.0.0.1', port=self.port, debug=False, use_reloader=False),
            daemon=True
        )
        thread.start()
        
        print(f"\n🌐 Control Panel: http://127.0.0.1:{self.port}")
        print("   Open this URL in your web browser!\n")
    
    def stop(self):
        """Stop the web server"""
        self.running = False

