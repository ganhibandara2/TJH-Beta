import logging
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from models.auth import Token, User
from services.auth_service import authenticate_user, create_access_token
from services import session_service
from dependencies.auth import get_current_active_user
from settings import settings

logger = logging.getLogger(__name__)

# Create router for authentication endpoints
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible login endpoint
    
    Authenticate user with username and password, return JWT access token.
    Only one user can be logged in at a time. If a session is already active,
    the login will be rejected.
    
    Args:
        form_data: OAuth2 form data with username and password
        
    Returns:
        Token object with access_token and token_type
        
    Raises:
        HTTPException: If credentials are invalid or a user is already logged in
    """
    # Check if a session is already active
    if session_service.session_manager is None:
        logger.error("Session manager not initialized - session enforcement disabled")
        # Allow login to proceed without session check if manager not initialized
    elif session_service.session_manager.is_session_active():
        logger.warning(f"Login attempt blocked: a user is already logged in")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user is currently logged in"
        )
    
    # Authenticate user
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is disabled
    if user.disabled:
        logger.warning(f"Disabled user '{user.username}' attempted login")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is disabled"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, 
        expires_delta=access_token_expires
    )
    
    # Set the active session
    if session_service.session_manager:
        session_service.session_manager.set_active_session(access_token, user.username)
    
    logger.info(f"User '{user.username}' logged in successfully")
    
    return Token(access_token=access_token, token_type="bearer")


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout endpoint - clears the active session
    
    Requires valid authentication to prevent unauthorized session clearing.
    
    Args:
        current_user: Current user from authentication dependency
        
    Returns:
        Success message
    """
    if session_service.session_manager:
        session_service.session_manager.clear_active_session(current_user.username)
    logger.info(f"User '{current_user.username}' logged out successfully")
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user information
    
    Args:
        current_user: Current user from authentication dependency
        
    Returns:
        User object with current user details
    """
    return current_user
