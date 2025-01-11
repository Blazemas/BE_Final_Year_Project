import requests
import base64

# Spotify Client ID and Secret
client_id = "e84512d57081422a8fd6ce94b7ee2308"
client_secret = "7ac00f1197f542419056b48ccc2088df"

# Encode Client ID and Secret
auth_string = f"{client_id}:{client_secret}"
auth_bytes = auth_string.encode("utf-8")
auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

# Request Access Token
url = "https://accounts.spotify.com/api/token"
headers = {
    "Authorization": f"Basic {auth_base64}",
    "Content-Type": "application/x-www-form-urlencoded"
}
data = {"grant_type": "client_credentials"}

response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    access_token = response.json()["access_token"]
    print("Access Token:", access_token)
else:
    print("Failed to obtain access token:", response.status_code, response.text)
