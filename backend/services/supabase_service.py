import logging
from typing import Optional, Dict, Any, List
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
    
    # ========== Job Methods ==========
    
    def save_jobs(self, jobs: List[Dict[str, Any]], search_query: str = "", 
                  search_location: str = "") -> int:
        """
        Save scraped jobs to the jobs table with upsert (insert or update).
        
        Args:
            jobs: List of job dictionaries
            search_query: Original search query
            search_location: Original search location
            
        Returns:
            Number of jobs saved/updated
        """
        if not jobs:
            return 0
        
        saved_count = 0
        for job in jobs:
            try:
                job_data = {
                    "job_id": job.get("job_id", ""),
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "salary": job.get("salary", ""),
                    "job_type": job.get("job_type", ""),
                    "description": job.get("description", "")[:1000] if job.get("description") else "",
                    "apply_url": job.get("apply_url", ""),
                    "source": job.get("source", "indeed"),
                    "scraped_at": job.get("scraped_at"),
                    "search_query": search_query,
                    "search_location": search_location,
                }
                
                # Upsert: insert or update on conflict
                response = self.client.table("jobs").upsert(
                    job_data,
                    on_conflict="job_id"
                ).execute()
                
                if response.data:
                    saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving job '{job.get('title', 'unknown')}': {e}")
                continue
        
        logger.info(f"Saved {saved_count} of {len(jobs)} jobs to database")
        return saved_count
    
    def get_jobs(self, query: str = "", location: str = "", 
                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve jobs from the database with optional filtering.
        
        Args:
            query: Filter by search query (optional)
            location: Filter by search location (optional)
            limit: Maximum number of jobs to return
            
        Returns:
            List of job dictionaries
        """
        try:
            db_query = self.client.table("jobs").select("*")
            
            if query:
                db_query = db_query.ilike("search_query", f"%{query}%")
            if location:
                db_query = db_query.ilike("search_location", f"%{location}%")
            
            db_query = db_query.order("scraped_at", desc=True).limit(limit)
            
            response = db_query.execute()
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error retrieving jobs: {e}")
            return []


# Global instance
supabase_service = SupabaseService()

