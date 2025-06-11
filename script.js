document.addEventListener('DOMContentLoaded', function () {
    // --- App Elements ---
    const startScreen = document.getElementById('start-screen');
    const launchButton = document.getElementById('launch-button');
    const appContainer = document.getElementById('app-container');

    const navButtons = document.querySelectorAll('.nav-button');
    const pages = document.querySelectorAll('.page');
    const themeCombo = document.getElementById('theme-combo');
    const consoleToggle = document.getElementById('console-toggle');
    const logText = document.getElementById('console-log-text');
    const startWebcamButton = document.getElementById('start-webcam-button');
    const webcamFeed = document.getElementById('webcam-feed');
    const outputCanvas = document.getElementById('output-canvas');
    const canvasContext = outputCanvas.getContext('2d');
    let model, videoStream;
    let isDetecting = false;
    let lastFrameTime = 0;
    const FPS = 30; // Target FPS for detection

    // COCO class names for YOLO
    const classNames = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ];

    // --- Launch Logic ---
    launchButton.addEventListener('click', async () => {
        try {
            await document.documentElement.requestFullscreen();
            logMessage("Entered full screen mode.");
        } catch (err) {
            logMessage(`Error attempting to enable full-screen mode: ${err.message}`, 'log-error');
        }
        
        startScreen.style.display = 'none';
        appContainer.classList.add('visible');
    });

    // --- Page Switching Logic ---
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const pageId = button.dataset.page;
            pages.forEach(page => page.classList.remove('active'));
            navButtons.forEach(btn => btn.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');
            button.classList.add('active');
            logMessage(`Switched to ${button.title} page.`);
        });
    });

    // --- Theme Switching Logic ---
    themeCombo.addEventListener('change', () => {
        const selectedTheme = themeCombo.value;
        document.body.className = selectedTheme;
        logMessage(`Theme changed to ${selectedTheme === 'dark-theme' ? 'Dark' : 'Light'}`);
    });

    // --- Console Toggle Logic ---
    consoleToggle.addEventListener('change', () => {
        logText.classList.toggle('visible');
    });
    
    // --- Webcam & Object Detection Logic ---
    if(startWebcamButton) {
        startWebcamButton.addEventListener('click', async () => {
            if (videoStream && videoStream.active) {
                videoStream.getTracks().forEach(track => track.stop());
                webcamFeed.srcObject = null;
                startWebcamButton.innerHTML = '<i class="fas fa-camera"></i> Start Webcam';
                logMessage("Webcam stopped.");
                canvasContext.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
                videoStream = null;
                isDetecting = false;
            } else {
                try {
                    startWebcamButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
                    logMessage("Starting webcam...");
                    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    webcamFeed.srcObject = videoStream;
                    webcamFeed.onloadedmetadata = async () => {
                        outputCanvas.width = webcamFeed.videoWidth;
                        outputCanvas.height = webcamFeed.videoHeight;
                        startWebcamButton.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Webcam';
                        logMessage("Webcam started successfully.");
                        
                        // Load YOLO model if not already loaded
                        if (!model) {
                            try {
                                logMessage("Loading YOLO model...");
                                model = await ort.InferenceSession.create('./yolov8n.onnx');
                                logMessage("Model loaded successfully.");
                            } catch (error) {
                                logMessage(`Error loading model: ${error.message}`, "log-error");
                                return;
                            }
                        }
                        
                        isDetecting = true;
                        runObjectDetection();
                    };
                } catch (error) {
                    logMessage(`Error starting webcam: ${error.message}`, "log-error");
                    startWebcamButton.innerHTML = '<i class="fas fa-camera"></i> Start Webcam';
                }
            }
        });
    }

    async function runObjectDetection() {
        if (!isDetecting) return;

        const currentTime = performance.now();
        const elapsed = currentTime - lastFrameTime;
        
        if (elapsed > (1000 / FPS)) {
            lastFrameTime = currentTime;
            
            try {
                const inputTensor = preprocess(webcamFeed);
                const feeds = { images: inputTensor };
                const results = await model.run(feeds);
                
                const detections = postprocess(results.output, outputCanvas.width, outputCanvas.height);
                drawBoundingBoxes(detections);
                
                // Update FPS counter
                document.getElementById('fps-label').innerText = `FPS: ${Math.round(1000 / elapsed)}`;
            } catch (error) {
                logMessage(`Detection error: ${error.message}`, "log-error");
            }
        }
        
        requestAnimationFrame(runObjectDetection);
    }
    
    function preprocess(video) {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 640, 640);
        const imageData = ctx.getImageData(0, 0, 640, 640);
        const { data } = imageData;
        const float32Data = new Float32Array(3 * 640 * 640);
        
        for (let i = 0; i < 640 * 640; i++) {
            float32Data[i] = data[i * 4] / 255.0; // R
            float32Data[i + 640 * 640] = data[i * 4 + 1] / 255.0; // G
            float32Data[i + 2 * 640 * 640] = data[i * 4 + 2] / 255.0; // B
        }
        
        return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
    }

    function postprocess(tensor, width, height) {
        const data = tensor.data;
        const detections = [];
        const numClasses = 80;
        const numBoxes = 8400;
        
        for (let i = 0; i < numBoxes; i++) {
            const confidence = data[4 * numBoxes + i];
            if (confidence > 0.5) { // Confidence threshold
                let maxClassScore = 0;
                let maxClassIndex = 0;
                
                // Find class with highest score
                for (let j = 0; j < numClasses; j++) {
                    const score = data[(5 + j) * numBoxes + i];
                    if (score > maxClassScore) {
                        maxClassScore = score;
                        maxClassIndex = j;
                    }
                }
                
                const finalScore = confidence * maxClassScore;
                if (finalScore > 0.5) { // Final confidence threshold
                    const x = data[i] * width / 640;
                    const y = data[numBoxes + i] * height / 640;
                    const w = data[2 * numBoxes + i] * width / 640;
                    const h = data[3 * numBoxes + i] * height / 640;
                    
                    detections.push({
                        x: x - w/2,
                        y: y - h/2,
                        width: w,
                        height: h,
                        class: classNames[maxClassIndex],
                        confidence: finalScore
                    });
                }
            }
        }
        return detections;
    }

    function drawBoundingBoxes(detections) {
        canvasContext.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        
        detections.forEach(detection => {
            // Draw bounding box
            canvasContext.strokeStyle = '#00ff00';
            canvasContext.lineWidth = 2;
            canvasContext.strokeRect(detection.x, detection.y, detection.width, detection.height);
            
            // Draw label background
            const label = `${detection.class} ${Math.round(detection.confidence * 100)}%`;
            canvasContext.fillStyle = 'rgba(0, 255, 0, 0.5)';
            canvasContext.font = '16px Arial';
            const textWidth = canvasContext.measureText(label).width;
            canvasContext.fillRect(detection.x, detection.y - 25, textWidth + 10, 25);
            
            // Draw label text
            canvasContext.fillStyle = '#000000';
            canvasContext.fillText(label, detection.x + 5, detection.y - 7);
        });

        document.getElementById('objects-label').innerText = `Objects: ${detections.length}`;
    }

    function logMessage(message, className = '') {
        const currentTime = new Date().toLocaleTimeString('en-GB');
        const p = document.createElement('p');
        const timestampSpan = document.createElement('span');
        timestampSpan.className = 'timestamp';
        timestampSpan.textContent = `[${currentTime}]`;
        const messageSpan = document.createElement('span');
        messageSpan.textContent = message;
        if(className) messageSpan.classList.add(className);
        p.appendChild(timestampSpan);
        p.appendChild(messageSpan);
        logText.appendChild(p);
        logText.scrollTop = logText.scrollHeight;
    }
    
    // --- Initialize ---
    if(consoleToggle.checked) {
        logText.classList.add('visible');
    }
});