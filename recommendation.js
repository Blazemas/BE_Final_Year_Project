// Access the webcam and start video feed
const video = document.getElementById('webcam');
const emotionDisplay = document.getElementById('detectedEmotion');
const recommendations = document.getElementById('musicRecommendations');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((error) => {
        console.error("Error accessing webcam:", error);
        alert("Camera access is required for emotion detection.");
    });

// Dummy function for emotion detection
document.getElementById('detectEmotionButton').addEventListener('click', () => {
    // Simulate emotion detection
    const emotions = ['Happy', 'Sad', 'Angry', 'Excited', 'Relaxed'];
    const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
    emotionDisplay.textContent = randomEmotion;
});

// Dummy function for music recommendation
document.getElementById('recommendMusicButton').addEventListener('click', () => {
    const emotion = emotionDisplay.textContent;

    let musicList = '';
    if (emotion === 'Happy') {
        musicList = '<p>1. Happy Song - Artist A</p><p>2. Joyful Tune - Artist B</p>';
    } else if (emotion === 'Sad') {
        musicList = '<p>1. Blue Mood - Artist C</p><p>2. Tears in Rain - Artist D</p>';
    } else {
        musicList = '<p>1. Chill Beats - Artist E</p><p>2. Ambient Vibes - Artist F</p>';
    }

    recommendations.innerHTML = musicList;
});
