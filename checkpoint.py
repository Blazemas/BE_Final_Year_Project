import requests
import time

# Function to fetch songs from Jamendo API
def fetch_songs(tag):
    url = "https://api.jamendo.com/v3.0/tracks"
    client_id = "1d5a6724"  # Your actual client ID
    params = {
        "client_id": client_id,
        "tags": tag,
        "limit": 5,  # Adjust this as needed
        "format": "json"
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to play a song
def play_song(song):
    # Implement your logic to play the song
    print(f"Now playing: {song['name']} by {song['artist_name']}")
    # Simulate playing the song
    time.sleep(1)  # Simulate time for playing the song (adjust as needed)

# Input mood
input_mood = "neutral"  # Example input (you can change this based on your project logic)

# Mood order: start with fear and gradually transition to happy
mood_order = ["fear", "angry", "sad", "disgust", "neutral", "surprise", "happy"]

# Find the index of the input mood in the mood_order list
start_index = mood_order.index(input_mood)

# Slice the mood order to only include moods starting from the input mood to "happy"
mood_sequence = mood_order[start_index:]

# Fetch and play songs based on the sliced mood sequence
for mood in mood_sequence:
    songs = fetch_songs(mood)
    if songs['results']:
        print(f"Playing songs for mood: {mood}")
        for song in songs['results']:
            play_song(song)  # Play each song in the current mood
