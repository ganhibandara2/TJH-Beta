import logging
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt

from settings import settings
from models.auth import UserInDB
from services.supabase_service import supabase_service

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password
    
    Args:
        plain_password: Plain text password
        hashed_password: BCrypt hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using BCrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Authenticate a user with username and password
    
    Args:
        username: Username to authenticate
        password: Plain text password
        
    Returns:
        UserInDB if authentication successful, None otherwise
    """
    try:
        # Fetch user from Supabase
        user_data = supabase_service.get_user_by_username(username)
        if not user_data:
            logger.warning(f"Authentication failed: user '{username}' not found")
            return None
        
        # Verify password
        if not verify_password(password, user_data.get("hashed_password", "")):
            logger.warning(f"Authentication failed: invalid password for user '{username}'")
            return None
        
        # Convert to UserInDB model
        user = UserInDB(
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data.get("full_name"),
            disabled=user_data.get("disabled", False),
            hashed_password=user_data["hashed_password"],
            created_at=user_data.get("created_at")
        )
        
        logger.info(f"User '{username}' authenticated successfully")
        return user
        
    except Exception as e:
        logger.error(f"Error authenticating user '{username}': {e}")
        return None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Data to encode in the token (typically {"sub": username})
        expires_delta: Optional expiration time delta
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    return encoded_jwt
