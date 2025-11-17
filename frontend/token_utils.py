import jwt
from datetime import timezone, datetime, timedelta
from config import Config

def generate_token(email: str):
    """Generate JWT token with user information"""
    user_data = Config.USERS_DB.get(email)
    if not user_data:
        return None

    payload = {
        "email": email,
        "name": user_data["name"],
        "role": user_data["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=5)
    }
    return jwt.encode(payload, Config.SECRET_KEY, algorithm="HS256")  
def verify_token(token: str):
    """Validate JWT token"""
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=["HS256"])  
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None