// Function to toggle screens
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    document.getElementById(screenId).classList.add('active');
}

// Initialize the home screen as the active screen
showScreen('homeScreen');
// Start the webcam feed
function startWebcam() {
    const video = document.getElementById('webcam');
    // Access the webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(error => {
            console.error("Error accessing webcam: ", error);
            alert("Webcam access is required for mood detection.");
        });
}

// Stop the webcam feed
function stopWebcam() {
    const video = document.getElementById('webcam');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
}


// To show the live camera actions 

let stream; // To hold the webcam stream

// Start the webcam feed in the specified video element
function startWebcam() {
    const video = document.getElementById('webcam');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(mediaStream => {
            stream = mediaStream; // Save the stream so we can stop it later
            video.srcObject = stream;
        })
        .catch(error => {
            console.error("Error accessing webcam: ", error);
            alert("Webcam access is required for mood detection.");
        });
}

// Stop the webcam feed when no longer needed
function stopWebcam() {
    const video = document.getElementById('webcam');
    if (stream) {
        stream.getTracks().forEach(track => track.stop()); // Stop each track
        video.srcObject = null;
    }
}

// Show the specified screen and manage the webcam feed
function showScreen(screenId) {
    const screens = document.querySelectorAll('.screen');
    screens.forEach(screen => screen.classList.remove('active'));

    const activeScreen = document.getElementById(screenId);
    activeScreen.classList.add('active');

    if (screenId === 'moodDetectionScreen') {
        startWebcam();
    } else {
        stopWebcam();
    }
}

// Initialize the app by showing the home screen
document.addEventListener('DOMContentLoaded', () => {
    showScreen('homeScreen');
});
