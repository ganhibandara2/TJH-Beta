import logging
from typing import Optional, Dict, Any
from supabase import create_client, Client
from settings import settings

logger = logging.getLogger(__name__)


class SupabaseService:
    """Service for interacting with Supabase database"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        self._ensure_users_table()
    
    def _ensure_users_table(self):
        """Ensure the users table exists in Supabase"""
        try:
            # Try to query the table - if it doesn't exist, Supabase will create it
            # Note: For production, you should use Supabase migrations
            logger.info("Checking users table in Supabase")
            # Just verify we can access the table
            self.client.table("users").select("id").limit(1).execute()
            logger.info("Users table exists")
        except Exception as e:
            logger.warning(f"Users table may not exist or is inaccessible: {e}")
            logger.info("Please create the users table in Supabase dashboard with the schema from implementation_plan.md")
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user by username from Supabase
        
        Args:
            username: Username to search for
            
        Returns:
            User data dict or None if not found
        """
        try:
            response = self.client.table("users").select("*").eq("username", username).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching user by username '{username}': {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user by email from Supabase
        
        Args:
            email: Email to search for
            
        Returns:
            User data dict or None if not found
        """
        try:
            response = self.client.table("users").select("*").eq("email", email).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching user by email '{email}': {e}")
            return None
    
    def create_user(self, username: str, email: str, hashed_password: str, 
                   full_name: Optional[str] = None, disabled: bool = False) -> Optional[Dict[str, Any]]:
        """
        Create a new user in Supabase
        
        Args:
            username: Unique username
            email: Unique email address
            hashed_password: BCrypt hashed password
            full_name: Optional full name
            disabled: Whether the user is disabled
            
        Returns:
            Created user data dict or None if creation failed
        """
        try:
            user_data = {
                "username": username,
                "email": email,
                "hashed_password": hashed_password,
                "full_name": full_name,
                "disabled": disabled
            }
            response = self.client.table("users").insert(user_data).execute()
            if response.data and len(response.data) > 0:
                logger.info(f"Created user: {username}")
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error creating user '{username}': {e}")
            return None
    
    def update_user(self, username: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update user information
        
        Args:
            username: Username of user to update
            update_data: Dict of fields to update
            
        Returns:
            Updated user data dict or None if update failed
        """
        try:
            response = self.client.table("users").update(update_data).eq("username", username).execute()
            if response.data and len(response.data) > 0:
                logger.info(f"Updated user: {username}")
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error updating user '{username}': {e}")
            return None
    
    def initialize_default_user(self, username: str, password: str, email: str, 
                               full_name: str, hashed_password: str) -> bool:
        """
        Initialize default user if it doesn't exist
        
        Args:
            username: Default username
            password: Plain password (not used, kept for signature compatibility)
            email: Default email
            full_name: Default full name
            hashed_password: Pre-hashed password
            
        Returns:
            True if user was created or already exists, False on error
        """
        try:
            # Check if user already exists
            existing_user = self.get_user_by_username(username)
            if existing_user:
                logger.info(f"Default user '{username}' already exists")
                return True
            
            # Create default user
            user = self.create_user(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                disabled=False
            )
            
            if user:
                logger.info(f"Successfully created default user '{username}'")
                return True
            else:
                logger.error(f"Failed to create default user '{username}'")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing default user: {e}")
            return False


# Global instance
supabase_service = SupabaseService()
