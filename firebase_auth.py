import firebase_admin
from firebase_admin import auth, credentials

# Load Firebase credentials
cred = credentials.Certificate("firebase_private_key.json")  # Download this file (Step 6 below)
firebase_admin.initialize_app(cred)

def verify_token(id_token):
    """Verify Firebase ID Token"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        return None
