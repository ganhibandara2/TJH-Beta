"""
Database Migration - Create sessions table for persistent session storage

Run this migration on application startup to ensure sessions table exists.
"""

import logging
from supabase import Client

logger = logging.getLogger(__name__)


def run_migration(supabase_client: Client) -> bool:
    """
    Create sessions table if it doesn't exist.
    
    Args:
        supabase_client: Supabase client instance
        
    Returns:
        True if migration successful or table already exists
    """
    try:
        # Check if table exists by trying to query it
        try:
            supabase_client.table("sessions").select("id").limit(1).execute()
            logger.info("Sessions table already exists")
            return True
        except Exception:
            # Table doesn't exist, need to create it
            logger.info("Sessions table does not exist, creating...")
            pass
        
        # Note: Supabase doesn't support CREATE TABLE via the Python client
        # You need to run this SQL in the Supabase SQL Editor:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            username VARCHAR(255) NOT NULL,
            token TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username);
        CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
        CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
        """
        
        logger.warning(
            "Sessions table needs to be created manually in Supabase SQL Editor. "
            "Please run the following SQL:\n\n" + create_table_sql
        )
        
        # Try to verify table exists after warning
        try:
            supabase_client.table("sessions").select("id").limit(1).execute()
            logger.info("Sessions table verified successfully")
            return True
        except Exception as e:
            logger.error(f"Sessions table not found: {e}")
            logger.error("Please create the sessions table in Supabase SQL Editor")
            # Don't fail the application, just warn
            return False
            
    except Exception as e:
        logger.error(f"Migration error: {e}")
        return False
