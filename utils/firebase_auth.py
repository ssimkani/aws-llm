import requests
from utils.firebase_config import FIREBASE_API_KEY

BASE_URL = "https://identitytoolkit.googleapis.com/v1/accounts"


def firebase_signup(email, password):
    url = f"{BASE_URL}:signUp?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    res = requests.post(url, json=payload)
    return res.json()


def firebase_login(email, password):
    url = f"{BASE_URL}:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    res = requests.post(url, json=payload)
    return res.json()
