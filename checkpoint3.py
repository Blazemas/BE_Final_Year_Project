import requests
import webbrowser
import time

# Spotify Access Token
ACCESS_TOKEN = "BQDegIN7RqipSw_lt9pDSPipb-p8H4oAbsuBEeC7aYAExl9dQPwz-FhvjfdSCfLa2kniHYDa-MpUM5YB3JZWDDgeT8a0KjV63uA0eoN7dmw8-3yICfxCY-1g2JQXgmllDCqvTdypkNo"

# Mood-to-Playlist Mapping
mood_to_playlist = {
    1: "5XGCEyEv01OLSieGQb0L3r",  # Angry
    2: "2uEODdOqnVjn2I7hFyam6C",  # Disgust
    3: "4ZYnq0FmTGP5qRxm7dfN1i",  # Fear
    4: "2N43HIotuWDjlRB4BrUEQR",  # Neutral
    5: "7M9YyZeUW75LUUhAWgRf58",  # Sad
    6: "0flG11VjoQPAbseYlWvkBk",  # Happy
}

# Function to fetch tracks from a playlist
def fetch_playlist_tracks(playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        tracks = response.json().get("items", [])
        return [track["track"] for track in tracks if track.get("track") and track["track"].get("id")]
    else:
        print(f"Failed to fetch tracks for playlist {playlist_id}: {response.status_code}, {response.text}")
        return []

# Function to play songs and transition moods
def play_and_transition(initial_mood, songs_per_mood):
    mood_order = list(range(initial_mood, 7))  # Transition from initial_mood to Happy (6)

    print(f"Starting playlist transition from mood {initial_mood} to Happy (6):")

    for mood in mood_order:
        print(f"\nPlaying songs from mood {mood}:")
        current_playlist = fetch_playlist_tracks(mood_to_playlist[mood])

        if not current_playlist:
            print(f"Playlist for mood {mood} is empty. Skipping.")
            continue

        for _ in range(min(songs_per_mood, len(current_playlist))):
            track = current_playlist.pop(0)  # Play songs in order
            print(f"Mood: {mood} - Song: {track['name']}")
            
            # Open song in browser
            webbrowser.open(f"https://open.spotify.com/track/{track['id']}?autoplay=true")
            
            # Wait for user confirmation before playing next song
            input("Press Enter to play the next song...")

# Run the function
play_and_transition(initial_mood=1, songs_per_mood=3)
