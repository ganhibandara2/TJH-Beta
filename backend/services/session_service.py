"""
Session Service - Manages single active session tracking with Supabase persistence

This service implements persistent session tracking to ensure only one user
can be logged in at a time. Sessions automatically expire when JWT tokens expire.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from supabase import Client

from settings import settings

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages active session state for single-session login enforcement.
    
    Sessions are stored in Supabase for persistence across server restarts.
    Only one session can be active at a time. Sessions automatically expire
    based on JWT token expiration time.
    """
    
    def __init__(self, supabase_client: Client):
        """
        Initialize SessionManager with Supabase client.
        
        Args:
            supabase_client: Supabase client instance
        """
        self.client = supabase_client
    
    def is_session_active(self) -> bool:
        """
        Check if there is currently an active (non-expired) session.
        
        Returns:
            True if a session is active, False otherwise
        """
        try:
            # Query for non-expired sessions
            response = self.client.table("sessions").select("id").gt(
                "expires_at", datetime.utcnow().isoformat()
            ).execute()
            
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Error checking session status: {e}")
            return False
    
    def set_active_session(self, token: str, username: str) -> bool:
        """
        Set the active session with the given token.
        
        Automatically calculates expiration based on ACCESS_TOKEN_EXPIRE_MINUTES.
        
        Args:
            token: JWT access token for the session
            username: Username of the logged-in user
            
        Returns:
            True if session was created successfully
        """
        try:
            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )
            
            # Insert new session
            session_data = {
                "username": username,
                "token": token,
                "expires_at": expires_at.isoformat()
            }
            
            response = self.client.table("sessions").insert(session_data).execute()
            
            if response.data:
                logger.info(f"Session activated for user '{username}', expires at {expires_at}")
                return True
            
            logger.error("Failed to create session")
            return False
            
        except Exception as e:
            logger.error(f"Error setting active session: {e}")
            return False
    
    def clear_active_session(self, username: Optional[str] = None) -> bool:
        """
        Clear the currently active session.
        
        Args:
            username: Optional username to clear specific user's session
            
        Returns:
            True if session was cleared successfully
        """
        try:
            if username:
                # Clear specific user's session
                response = self.client.table("sessions").delete().eq(
                    "username", username
                ).execute()
                logger.info(f"Session cleared for user '{username}'")
            else:
                # Clear all sessions (fallback)
                response = self.client.table("sessions").delete().neq(
                    "id", "00000000-0000-0000-0000-000000000000"  # Delete all
                ).execute()
                logger.info("All sessions cleared")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
    
    def get_active_session(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get the current active session for a user.
        
        Args:
            username: Username to get session for
            
        Returns:
            Session data dict or None if no active session
        """
        try:
            response = self.client.table("sessions").select("*").eq(
                "username", username
            ).gt(
                "expires_at", datetime.utcnow().isoformat()
            ).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting active session: {e}")
            return None
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions from the database.
        
        Should be called periodically by a background task.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            # Delete sessions where expires_at < now
            response = self.client.table("sessions").delete().lt(
                "expires_at", datetime.utcnow().isoformat()
            ).execute()
            
            count = len(response.data) if response.data else 0
            if count > 0:
                logger.info(f"Cleaned up {count} expired session(s)")
            
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def get_session_info(self, username: str) -> dict:
        """
        Get information about the current session.
        
        Args:
            username: Username to get session info for
            
        Returns:
            Dict with session info or empty dict if no active session
        """
        session = self.get_active_session(username)
        
        if not session:
            return {}
        
        return {
            "username": session.get("username"),
            "created_at": session.get("created_at"),
            "expires_at": session.get("expires_at"),
            "is_active": True
        }


# Note: session_manager will be initialized in main.py after Supabase client is available
session_manager: Optional[SessionManager] = None


def initialize_session_manager(supabase_client: Client) -> None:
    """
    Initialize the global session manager instance.
    
    Should be called during application startup.
    
    Args:
        supabase_client: Supabase client instance
    """
    global session_manager
    session_manager = SessionManager(supabase_client)
    logger.info("Session manager initialized with Supabase")
