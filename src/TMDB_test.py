import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("TMDB_READ_TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "accept": "application/json"
}

tmdb_id = 550  # Fight Club

response = requests.get(
    f"https://api.themoviedb.org/3/movie/{tmdb_id}",
    headers=headers,
    params={
        "language": "en-US",
        "append_to_response": "credits,keywords"
    },
    timeout=20
)

response.raise_for_status()
movie = response.json()

print(movie["title"])
print(movie["overview"])
print([keyword["name"] for keyword in movie["keywords"]["keywords"]])
print([person["name"] for person in movie["credits"]["cast"][:5]])

print()
print()