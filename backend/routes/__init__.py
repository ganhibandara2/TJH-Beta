"""API Routes"""
from .auth import router as auth_router
from .scrape import router as scrape_router

__all__ = ["auth_router", "scrape_router"]

