"""
Scrape API routes for deep job scraping using Bright Data Scraping Browser.
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.brightdata_scraper_service import scrape_indeed_jobs, test_connection
from services.supabase_service import supabase_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scrape", tags=["Scraping"])


class DeepScrapeRequest(BaseModel):
    """Request model for deep scraping endpoint."""
    query: str = Field(..., min_length=1, max_length=200, description="Job title/keywords to search")
    location: str = Field(default="", max_length=200, description="Location to search in")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum jobs to scrape (1-100)")
    country_code: str = Field(default="US", max_length=2, description="ISO 2-letter country code")
    save_to_db: bool = Field(default=True, description="Whether to save results to database")


class JobResult(BaseModel):
    """Individual job result."""
    job_id: str
    title: str
    company: str
    location: str
    salary: str
    job_type: str
    description: str
    apply_url: str
    source: str
    scraped_at: str


class DeepScrapeResponse(BaseModel):
    """Response model for deep scraping endpoint."""
    success: bool
    message: str
    job_count: int
    saved_count: int
    jobs: list[JobResult]


@router.post("/deep", response_model=DeepScrapeResponse)
async def deep_scrape_indeed(request: DeepScrapeRequest):
    """
    Deep scrape Indeed jobs using Bright Data Scraping Browser.
    
    This endpoint connects to Bright Data's remote browser via CDP to scrape
    job listings from Indeed. Results can optionally be saved to the database.
    
    **Note**: This uses Bright Data's infrastructure - no local browser execution.
    
    Args:
        request: DeepScrapeRequest with query, location, and options
        
    Returns:
        DeepScrapeResponse with scraped jobs and counts
    """
    logger.info(f"Deep scrape request: query='{request.query}', location='{request.location}', max={request.max_results}")
    
    try:
        # Scrape jobs using Bright Data Scraping Browser
        jobs = await scrape_indeed_jobs(
            query=request.query,
            location=request.location,
            max_results=request.max_results,
            country_code=request.country_code
        )
        
        saved_count = 0
        if request.save_to_db and jobs:
            # Save to Supabase
            saved_count = supabase_service.save_jobs(
                jobs=jobs,
                search_query=request.query,
                search_location=request.location
            )
        
        # Convert to response format
        job_results = [
            JobResult(
                job_id=job.get("job_id", ""),
                title=job.get("title", ""),
                company=job.get("company", ""),
                location=job.get("location", ""),
                salary=job.get("salary", ""),
                job_type=job.get("job_type", ""),
                description=job.get("description", ""),
                apply_url=job.get("apply_url", ""),
                source=job.get("source", "indeed"),
                scraped_at=job.get("scraped_at", "")
            )
            for job in jobs
        ]
        
        return DeepScrapeResponse(
            success=True,
            message=f"Successfully scraped {len(jobs)} jobs",
            job_count=len(jobs),
            saved_count=saved_count,
            jobs=job_results
        )
        
    except ValueError as e:
        # Configuration errors (missing credentials)
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except RuntimeError as e:
        # Scraping errors
        logger.error(f"Scraping error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Scraping failed: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error during deep scrape: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.get("/test-connection")
async def test_brightdata_connection():
    """
    Test the connection to Bright Data Scraping Browser.
    
    Use this endpoint to verify that credentials are configured correctly
    and the remote browser is accessible.
    
    Returns:
        Connection status and browser version if successful
    """
    try:
        result = await test_connection()
        
        if result["status"] == "success":
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result["message"]
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Connection test error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connection test failed: {str(e)}"
        )
