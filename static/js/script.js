document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const webcamTab = document.getElementById('webcam-tab');
    const uploadTab = document.getElementById('upload-tab');
    const webcamContent = document.getElementById('webcam-content');
    const uploadContent = document.getElementById('upload-content');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const imageOptionBtn = document.getElementById('image-option-btn');
    const videoOptionBtn = document.getElementById('video-option-btn');
    const imageUploadContainer = document.getElementById('image-upload-container');
    const videoUploadContainer = document.getElementById('video-upload-container');
    const imageDropArea = document.getElementById('image-drop-area');
    const videoDropArea = document.getElementById('drop-area');
    const imageUpload = document.getElementById('image-upload');
    const videoUpload = document.getElementById('video-upload');
    const previewImage = document.getElementById('preview-image');
    const previewContainer = document.getElementById('preview-container');
    const previewCanvas = document.getElementById('preview-canvas');
    const analyzeBtn = document.getElementById('analyze-btn');
    const emotionsList = document.getElementById('emotions-list');
    const uploadedVideo = document.getElementById('uploaded-video');
    const videoCanvas = document.getElementById('video-canvas');
    const videoPreviewContainer = document.getElementById('video-preview-container');
    const playAnalyzeBtn = document.getElementById('play-analyze-btn');
    const stopAnalyzeBtn = document.getElementById('stop-analyze-btn');
    const analyzeProfileBtn = document.getElementById('analyze-profile-btn');
    const profileContent = document.getElementById('profile-content');
    const totalDetections = document.getElementById('total-detections');
    const dominantEmotion = document.getElementById('dominant-emotion');
    const emotionDiversity = document.getElementById('emotion-diversity');
    const sessionDuration = document.getElementById('session-duration');

    // Canvas contexts
    const ctx = canvas.getContext('2d');
    const previewCtx = previewCanvas.getContext('2d');
    const videoCtx = videoCanvas.getContext('2d');

    // Stream reference
    let stream = null;
    let isAnalyzingUploadedVideo = false;
    let sessionStartTime = null;
    let sessionTimer = null;

    // Emotion data storage
    let emotionHistory = [];
    const maxHistorySize = 100;
    let emotionCounts = {};
    let totalDetectionCount = 0;

    // Emotion colors
    const emotionColors = {
        'Angry': '#e74c3c',
        'Disgust': '#8e44ad',
        'Fear': '#9b59b6',
        'Happy': '#f1c40f',
        'Sad': '#3498db',
        'Surprise': '#e67e22',
        'Neutral': '#95a5a6',
        'Anxiety': '#1abc9c',
        'Embarrassed': '#e84393'
    };

    // Initialize charts
    const timelineCtx = document.getElementById('emotion-timeline').getContext('2d');
    const distributionCtx = document.getElementById('emotion-distribution').getContext('2d');
    
    // Timeline chart
    const timelineChart = new Chart(timelineCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Emotion Confidence',
                data: [],
                borderColor: '#3498db',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Confidence (%)'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
    
    // Distribution chart
    const distributionChart = new Chart(distributionCtx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                }
            }
        }
    });

    // Update date display
    const profileDate = document.querySelector('.profile-date');
    if (profileDate) {
        profileDate.textContent = `Last updated: ${new Date().toLocaleString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: 'numeric',
            minute: 'numeric',
            hour12: true
        })}`;
    }

    // Tab switching
    webcamTab.addEventListener('click', function() {
        webcamTab.classList.add('active');
        uploadTab.classList.remove('active');
        webcamContent.classList.add('active');
        uploadContent.classList.remove('active');
    });

    uploadTab.addEventListener('click', function() {
        uploadTab.classList.add('active');
        webcamTab.classList.remove('active');
        uploadContent.classList.add('active');
        webcamContent.classList.remove('active');
    });

    // Toggle between image and video upload options
    imageOptionBtn.addEventListener('click', function() {
        imageOptionBtn.classList.add('active');
        videoOptionBtn.classList.remove('active');
        imageUploadContainer.style.display = 'block';
        videoUploadContainer.style.display = 'none';
    });

    videoOptionBtn.addEventListener('click', function() {
        videoOptionBtn.classList.add('active');
        imageOptionBtn.classList.remove('active');
        videoUploadContainer.style.display = 'block';
        imageUploadContainer.style.display = 'none';
    });

    // Start webcam
    startBtn.addEventListener('click', async function() {
        try {
            // If there's an existing stream, clean it up first
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            
            // Set canvas dimensions once video metadata is loaded
            video.onloadedmetadata = function() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            };
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            // Start emotion detection
            detectEmotionFromVideo();
            
            // Start session timer
            startSessionTimer();
            
        } catch (err) {
            console.error('Error accessing webcam:', err);
            alert('Could not access webcam. Please check permissions.');
        }
    });

    // Stop webcam
    stopBtn.addEventListener('click', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            
            // Stop session timer
            stopSessionTimer();
        }
    });

    // Image drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        imageDropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        imageDropArea.addEventListener(eventName, () => highlight(imageDropArea), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        imageDropArea.addEventListener(eventName, () => unhighlight(imageDropArea), false);
    });

    // Video drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        videoDropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        videoDropArea.addEventListener(eventName, () => highlight(videoDropArea), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        videoDropArea.addEventListener(eventName, () => unhighlight(videoDropArea), false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(element) {
        element.classList.add('highlight');
    }

    function unhighlight(element) {
        element.classList.remove('highlight');
    }

    // Handle image drop
    imageDropArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            handleImageFile(files[0]);
        } else {
            alert('Please drop a valid image file.');
        }
    });

    // Handle video drop
    videoDropArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            handleVideoFile(files[0]);
        } else {
            alert('Please drop a valid video file.');
        }
    });

    // Handle click on image drop area
    imageDropArea.addEventListener('click', function() {
        imageUpload.click();
    });

    // Handle click on video drop area
    videoDropArea.addEventListener('click', function() {
        videoUpload.click();
    });

    // Image upload preview
    imageUpload.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleImageFile(this.files[0]);
        }
    });

    // Video upload preview
    videoUpload.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleVideoFile(this.files[0]);
        }
    });

    // Process the image file
    function handleImageFile(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            analyzeBtn.disabled = false;
            
            // Set canvas dimensions once image is loaded
            previewImage.onload = function() {
                previewCanvas.width = previewImage.width;
                previewCanvas.height = previewImage.height;
            };
        };
        
        reader.readAsDataURL(file);
    }

    // Process the video file
    function handleVideoFile(file) {
        const videoURL = URL.createObjectURL(file);
        uploadedVideo.src = videoURL;
        videoPreviewContainer.style.display = 'block';
        playAnalyzeBtn.disabled = false;
        stopAnalyzeBtn.disabled = true;
        
        // Set canvas dimensions when video metadata is loaded
        uploadedVideo.onloadedmetadata = function() {
            videoCanvas.width = uploadedVideo.videoWidth;
            videoCanvas.height = uploadedVideo.videoHeight;
        };
    }

    // Analyze uploaded image
    analyzeBtn.addEventListener('click', function() {
        const file = imageUpload.files[0];
        
        if (file) {
            const formData = new FormData();
            formData.append('image', file);
            
            fetch('/detect_emotion', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data, previewCtx, previewImage);
                updateStatistics(data);
            })
            .catch(error => {
                console.error('Error detecting emotions:', error);
                alert('Error detecting emotions. Please try again.');
            });
        }
    });

    // Play and analyze uploaded video
    playAnalyzeBtn.addEventListener('click', function() {
        if (uploadedVideo.src) {
            uploadedVideo.play();
            isAnalyzingUploadedVideo = true;
            playAnalyzeBtn.disabled = true;
            stopAnalyzeBtn.disabled = false;
            
            // Start session timer
            startSessionTimer();
            
            analyzeUploadedVideo();
        }
    });

    // Stop analyzing uploaded video
    stopAnalyzeBtn.addEventListener('click', function() {
        uploadedVideo.pause();
        isAnalyzingUploadedVideo = false;
        playAnalyzeBtn.disabled = false;
        stopAnalyzeBtn.disabled = true;
        
        // Stop session timer
        stopSessionTimer();
    });

    // Function to detect emotions from video frames
    async function detectEmotionFromVideo() {
        if (!video.srcObject) return;
        
        // Capture frame from video
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob
        canvas.toBlob(function(blob) {
            const formData = new FormData();
            formData.append('frame', blob);
            
            fetch('/video_feed', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data, ctx, video);
                updateStatistics(data);
                
                // Continue detection if webcam is still active
                if (video.srcObject) {
                    requestAnimationFrame(detectEmotionFromVideo);
                }
            })
            .catch(error => {
                console.error('Error detecting emotions:', error);
                
                // Continue detection despite error
                if (video.srcObject) {
                    requestAnimationFrame(detectEmotionFromVideo);
                }
            });
        }, 'image/jpeg');
    }

    // Function to analyze uploaded video frames
    function analyzeUploadedVideo() {
        if (!isAnalyzingUploadedVideo || uploadedVideo.paused || uploadedVideo.ended) {
            return;
        }
        
        // Draw current video frame to canvas
        videoCtx.drawImage(uploadedVideo, 0, 0, videoCanvas.width, videoCanvas.height);
        
        // Convert canvas to blob
        videoCanvas.toBlob(function(blob) {
            const formData = new FormData();
            formData.append('frame', blob);
            
            fetch('/video_feed', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data, videoCtx, uploadedVideo);
                updateStatistics(data);
                
                // Continue analyzing if video is still playing
                if (isAnalyzingUploadedVideo && !uploadedVideo.paused && !uploadedVideo.ended) {
                    requestAnimationFrame(analyzeUploadedVideo);
                }
            })
            .catch(error => {
                console.error('Error analyzing video frame:', error);
                
                // Continue despite error
                if (isAnalyzingUploadedVideo && !uploadedVideo.paused && !uploadedVideo.ended) {
                    requestAnimationFrame(analyzeUploadedVideo);
                }
            });
        }, 'image/jpeg');
    }

    // Add event listener for video end
    uploadedVideo.addEventListener('ended', function() {
        isAnalyzingUploadedVideo = false;
        playAnalyzeBtn.disabled = false;
        stopAnalyzeBtn.disabled = true;
        
        // Stop session timer
        stopSessionTimer();
    });

    // Function to display emotion detection results
    function displayResults(data, context, sourceElement) {
        // Clear previous drawings
        context.clearRect(0, 0, context.canvas.width, context.canvas.height);
        
        // Update emotions list
        if (data.count === 0) {
            return;
        }
        
        // Draw bounding boxes and display results
        data.results.forEach((result) => {
            const [x, y, w, h] = result.box;
            const emotion = result.emotion;
            const confidence = (result.confidence * 100).toFixed(2);
            
            // Draw bounding box
            context.lineWidth = 3;
            context.strokeStyle = emotionColors[emotion] || '#ffffff';
            context.strokeRect(x, y, w, h);
            
            // Create emotion label box
            context.fillStyle = emotionColors[emotion] || '#ffffff';
            context.font = '16px Arial';
            const labelWidth = context.measureText(`${emotion} (${confidence}%)`).width + 20;
            context.fillRect(x, y - 30, labelWidth, 30);
            
            // Add text
            context.fillStyle = '#ffffff';
            context.font = 'bold 16px Arial';
            context.fillText(`${emotion} (${confidence}%)`, x + 10, y - 10);
            
            // Add to emotions list
            const emotionCard = document.createElement('div');
            emotionCard.className = 'emotion-card';
            emotionCard.style.borderLeft = `4px solid ${emotionColors[emotion] || '#333333'}`;
            emotionCard.innerHTML = `
                <span class="emotion-name">${emotion}</span>
                <span class="emotion-confidence">${confidence}% confidence</span>
            `;
            
            emotionsList.prepend(emotionCard);
            
            // Keep emotions list manageable
            if (emotionsList.children.length > 10) {
                emotionsList.removeChild(emotionsList.lastChild);
            }
            
            // Store emotion data for profiling
            storeEmotionData(emotion, parseFloat(confidence) / 100);
        });
    }
    
    // Store emotion data for analysis
    function storeEmotionData(emotion, confidence) {
        const timestamp = new Date();
        
        emotionHistory.push({
            emotion: emotion,
            confidence: confidence,
            timestamp: timestamp
        });
        
        // Keep history size manageable
        if (emotionHistory.length > maxHistorySize) {
            emotionHistory.shift();
        }
        
        // Update emotion counts
        emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
        totalDetectionCount++;
    }
    
    // Update statistics and charts
    function updateStatistics(data) {
        if (data.count === 0) return;
        
        // Update total detections
        totalDetections.textContent = totalDetectionCount;
        
        // Update dominant emotion
        let maxCount = 0;
        let dominant = 'None';
        
        for (const emotion in emotionCounts) {
            if (emotionCounts[emotion] > maxCount) {
                maxCount = emotionCounts[emotion];
                dominant = emotion;
            }
        }
        
        dominantEmotion.textContent = dominant;
        dominantEmotion.style.color = emotionColors[dominant] || '#333333';
        
        // Update emotion diversity
        emotionDiversity.textContent = Object.keys(emotionCounts).length;
        
        // Update timeline chart
        timestamp = new Date().toLocaleTimeString();
        const highestConfidence = Math.max(...data.results.map(r => parseFloat(r.confidence) * 100));
    const dominantEmotion = data.results.reduce((prev, current) => 
        (parseFloat(prev.confidence) > parseFloat(current.confidence)) ? prev : current
    ).emotion;
    
    // Update timeline chart
    const timestamp = new Date().toLocaleTimeString();
    timelineChart.data.labels.push(timestamp);
    timelineChart.data.datasets[0].data.push(highestConfidence);
    timelineChart.data.datasets[0].borderColor = emotionColors[dominantEmotion] || '#3498db';
    
    // Keep only the last 10 data points for better visualization
    if (timelineChart.data.labels.length > 10) {
        timelineChart.data.labels.shift();
        timelineChart.data.datasets[0].data.shift();
    }
    
    timelineChart.update();
    
    // Update distribution chart
    updateDistributionChart();
}

// Update the emotion distribution chart
function updateDistributionChart() {
    // Convert emotion counts to array for sorting
    const emotionCountsArray = Object.entries(emotionCounts).map(([emotion, count]) => ({ emotion, count }));
    const sortedEmotions = emotionCountsArray.sort((a, b) => b.count - a.count);

    // Update chart data
    distributionChart.data.labels = sortedEmotions.map(e => e.emotion);
    distributionChart.data.datasets[0].data = sortedEmotions.map(e => e.count);
    distributionChart.data.datasets[0].backgroundColor = sortedEmotions.map(e => emotionColors[e.emotion] || '#333333');

    // Update chart
    distributionChart.update();
}

// Start session timer
function startSessionTimer() {
    sessionStartTime = new Date();
    
    // Clear any existing timer
    if (sessionTimer) {
        clearInterval(sessionTimer);
    }
    
    // Update timer every second
    sessionTimer = setInterval(() => {
        const duration = Math.floor((new Date() - sessionStartTime) / 1000);
        const minutes = Math.floor(duration / 60);
        const seconds = duration % 60;
        sessionDuration.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }, 1000);
}

// Stop session timer
function stopSessionTimer() {
    if (sessionTimer) {
        clearInterval(sessionTimer);
        sessionTimer = null;
    }
}

// Generate emotion profile using ChatGPT API
analyzeProfileBtn.addEventListener('click', function() {
    // Show loading state
    profileContent.innerHTML = `
        <p>Generating emotion profile...</p>
        <div class="loading"></div>
    `;
    
    // Update the date display
    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        hour12: true
    });
    
    // Check if we have enough data
    if (emotionHistory.length < 5) {
        profileContent.innerHTML = `
            <p>Not enough emotion data collected yet. Continue using the application to collect more data.</p>
            <p class="profile-date">Last updated: ${formattedDate}</p>
        `;
        return;
    }
    
    // Send emotion history to server for analysis
    fetch('/generate_profile', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ emotionHistory: emotionHistory })
    })
    .then(response => response.json())
    .then(data => {
        // Display the profile
        profileContent.innerHTML = `
            <p>${data.profile.replace(/\n/g, '<br>')}</p>
            <p class="profile-date">Last updated: ${formattedDate}</p>
        `;
    })
    .catch(error => {
        console.error('Error generating profile:', error);
        profileContent.innerHTML = `
            <p>Error generating emotion profile. Please try again.</p>
            <p class="profile-date">Last updated: ${formattedDate}</p>
        `;
    });
});

// Fetch statistics from server periodically
function fetchStatistics() {
    fetch('/get_statistics')
    .then(response => response.json())
    .then(data => {
        // If there's a timeline chart in the response, display it
        if (data.timeline_chart) {
            const timelineImg = document.createElement('img');
            timelineImg.src = `data:image/png;base64,${data.timeline_chart}`;
            timelineImg.alt = 'Emotion Timeline';
            timelineImg.style.width = '100%';
            
            // Replace the chart canvas with the image
            const chartContainer = document.querySelector('.chart-container');
            chartContainer.innerHTML = '';
            chartContainer.appendChild(timelineImg);
        }
    })
    .catch(error => {
        console.error('Error fetching statistics:', error);
    });
}

// Initialize the application
function initApp() {
    // Set current date in profile
    if (profileContent) {
        const currentDate = new Date();
        const formattedDate = currentDate.toLocaleString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: 'numeric',
            minute: 'numeric',
            hour12: true
        });
        
        const dateElement = profileContent.querySelector('.profile-date');
        if (dateElement) {
            dateElement.textContent = `Last updated: ${formattedDate}`;
        }
    }
    
    // Fetch statistics every 30 seconds
    setInterval(fetchStatistics, 30000);
}

// Initialize the app when DOM is loaded
initApp();
});
