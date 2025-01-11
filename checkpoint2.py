import requests
import random
import webbrowser

# Spotify Access Token
ACCESS_TOKEN = "BQBOu24dJvgEcztOXz1JuxOi7F7T3KQcdnqaPim-eN2_YV2Hqhp8pZy6DqvtnylgPMZWuYip1OFGJmGWACMRjLZ8ljT2YmKHh7iobJ3VnlzPCxTUvhs"

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
        # Skip tracks with no name
        return [track["track"] for track in tracks if track["track"] and track["track"]["name"]]
    else:
        print(f"Failed to fetch tracks for playlist {playlist_id}: {response.status_code}, {response.text}")
        return []

# Function to play tracks and transition between playlists
def play_and_transition(initial_mood, songs_per_mood):
    # List of all moods from initial to Happy (6)
    mood_order = list(range(initial_mood, 7))  # Starting from initial_mood to Happy (6)
    
    print(f"Starting playlist transition from mood {initial_mood} to Happy (6):")

    # Iterate through the moods and playlists
    for mood in mood_order:
        print(f"\nPlaying from mood {mood}:")
        current_playlist = fetch_playlist_tracks(mood_to_playlist[mood])
        
        if not current_playlist:
            print(f"Playlist for mood {mood} is empty. Skipping this mood.")
            continue

        # Play fixed number of songs from the current mood playlist
        for _ in range(songs_per_mood):
            if current_playlist:
                track = random.choice(current_playlist)
                current_playlist.remove(track)  # Avoid repeats
                print(f"Mood: {mood} - Song: {track['name']}")
                # Open the song URL in the browser to play it
                webbrowser.open(f"https://open.spotify.com/track/{track['id']}")
            else:
                print(f"Playlist for mood {mood} is empty.")
                break

# Example: Start with Disgust (2), play 3 songs from that, and transition till Happy (6)
play_and_transition(initial_mood=1, songs_per_mood=3)
