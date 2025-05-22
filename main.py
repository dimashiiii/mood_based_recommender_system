import os
from dotenv import load_dotenv
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit.components.v1 as components
import gdown
from safetensors.torch import load_file

load_dotenv()

SPOTIPY_CLIENT_ID     = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

st.set_page_config(page_title="Mood-Based Music Recommender", page_icon="🎵", layout="centered")

# MODEL_PATH = "distilbert-finetuned-emotion/checkpoint-500"
MODEL_PATH = "model"
CLASSES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

SPOTIFY_PLAYLISTS = {
    'sadness': '37i9dQZF1DX7qK8ma5wgG1',
    'joy':     '37i9dQZF1DXdPec7aLTmlC',
    'love':    '37i9dQZF1DX50QitC6Oqtn',
    'anger':   '37i9dQZF1DX1dCsSMSXSsK',
    'fear':    '37i9dQZF1DX4fpCWaHOned',
    'surprise':'37i9dQZF1DX4fpCWaHOned',
}

@st.cache_resource
def load_model_and_tokenizer(path):
    import gdown

    safetensor_path = os.path.join(path, "model.safetensors")

    # Егер safetensors файл жоқ болса — жүктеу
    if not os.path.exists(safetensor_path):
        st.warning("Model file not found. Downloading from Google Drive...")
        os.makedirs(path, exist_ok=True)
        gdown.download(
            url="https://drive.google.com/uc?id=1m0ByQMvmZswCwajlVdfRBWDBVY9ISJo_",
            output=safetensor_path,
            quiet=False
        )

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_emotion(text: str) -> str:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return CLASSES[torch.argmax(logits, dim=1).item()]

credentials = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
)
sp = Spotify(client_credentials_manager=credentials)

st.title("🎵 Mood-Based Music Recommender")
user_input = st.text_input("💬 How are you feeling today ?")

if st.button("🎧 Recommend"):
    if not user_input.strip():
        st.warning("Please, describe your feelings.")
    else:
        emotion = predict_emotion(user_input)
        st.success(f"i think you feel **{emotion}**")

        playlist_id = SPOTIFY_PLAYLISTS.get(emotion)
        if playlist_id:
            embed = f"""
            <iframe src="https://open.spotify.com/embed/playlist/{playlist_id}"
                    width="300" height="380" frameborder="0"
                    allowtransparency="true" allow="encrypted-media"></iframe>
            """
            st.markdown("**Recommended playlist in Spotify:**")
            components.html(embed, height=400)
        else:
            st.info("For this emotion there is not playlist in spotify.")
