from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class Token(BaseModel):
    """OAuth2 token response"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Data extracted from JWT token"""
    username: Optional[str] = None


class UserBase(BaseModel):
    """Base user model with common fields"""
    username: str
    email: EmailStr
    full_name: Optional[str] = None


class User(UserBase):
    """User model for responses (without sensitive data)"""
    disabled: bool = False
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserInDB(User):
    """User model including hashed password (for internal use)"""
    hashed_password: str


class LoginRequest(BaseModel):
    """Login credentials"""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")


class UserCreate(UserBase):
    """Model for creating a new user"""
    password: str = Field(..., min_length=6, description="User password")
