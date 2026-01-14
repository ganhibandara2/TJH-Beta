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
    time_limit_seconds: int = Field(default=60, ge=10, le=180, description="Time limit for scraping in seconds (10-180)")
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
    
    **TIME-BASED APPROACH**: Scrapes as many jobs as possible within the time limit
    (default: 60 seconds). This is more practical than requesting a fixed job count.
    
    **Note**: This uses Bright Data's infrastructure - no local browser execution.
    
    Args:
        request: DeepScrapeRequest with query, location, and options
        
    Returns:
        DeepScrapeResponse with scraped jobs and counts
    """
    logger.info(f"Deep scrape request: query='{request.query}', location='{request.location}', time_limit={request.time_limit_seconds}s")
    
    try:
        # Scrape jobs using Bright Data Scraping Browser with time limit
        jobs = await scrape_indeed_jobs(
            query=request.query,
            location=request.location,
            time_limit_seconds=request.time_limit_seconds,
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


# Import for streaming
import asyncio
import json
import time
from fastapi.responses import StreamingResponse
from services.brightdata_scraper_service import scrape_indeed_jobs_with_progress


@router.post("/deep/stream")
async def deep_scrape_stream(request: DeepScrapeRequest):
    """
    Deep scrape Indeed jobs with STREAMING progress updates.
    
    Returns Server-Sent Events (SSE) stream with:
    - Progress updates: {type: "progress", time_remaining: X, jobs_collected: Y}
    - Final result: {type: "complete", jobs: [...], saved_count: N}
    - Error: {type: "error", message: "..."}
    """
    logger.info(f"[STREAM] Deep scrape request: query='{request.query}', location='{request.location}'")
    
    async def generate_stream():
        try:
            start_time = time.time()
            time_limit = request.time_limit_seconds
            
            # Send initial progress
            yield f"data: {json.dumps({'type': 'progress', 'time_remaining': time_limit, 'jobs_collected': 0, 'status': 'Connecting to Bright Data...'})}\n\n"
            
            # Start scraping with progress callback
            jobs = []
            async for progress in scrape_indeed_jobs_with_progress(
                query=request.query,
                location=request.location,
                time_limit_seconds=time_limit,
                country_code=request.country_code
            ):
                if progress["type"] == "progress":
                    yield f"data: {json.dumps(progress)}\n\n"
                elif progress["type"] == "jobs":
                    jobs = progress["jobs"]
            
            # Save to database
            saved_count = 0
            if request.save_to_db and jobs:
                yield f"data: {json.dumps({'type': 'progress', 'time_remaining': 0, 'jobs_collected': len(jobs), 'status': 'Saving to database...'})}\n\n"
                saved_count = supabase_service.save_jobs(
                    jobs=jobs,
                    search_query=request.query,
                    search_location=request.location
                )
            
            # Send final result
            job_results = [
                {
                    "job_id": job.get("job_id", ""),
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "salary": job.get("salary", ""),
                    "job_type": job.get("job_type", ""),
                    "description": job.get("description", ""),
                    "apply_url": job.get("apply_url", ""),
                    "source": job.get("source", "indeed"),
                    "scraped_at": job.get("scraped_at", "")
                }
                for job in jobs
            ]
            
            yield f"data: {json.dumps({'type': 'complete', 'jobs': job_results, 'job_count': len(jobs), 'saved_count': saved_count})}\n\n"
            
        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


class SearchJobsRequest(BaseModel):
    """Request model for searching cached jobs."""
    query: str = Field(..., min_length=1, max_length=200, description="Job title to search for")
    location: str = Field(default="", max_length=200, description="Location to filter by")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum jobs to return")


class SearchJobsResponse(BaseModel):
    """Response model for job search endpoint."""
    success: bool
    message: str
    job_count: int
    jobs: list[JobResult]


@router.post("/search", response_model=SearchJobsResponse)
async def search_cached_jobs(request: SearchJobsRequest):
    """
    Search for jobs from the Supabase cache (previously scraped jobs).
    
    This is the "normal" search endpoint - it queries the database for jobs
    that were previously collected via deep scraping. This is FAST (no scraping).
    
    Use the /deep endpoint to populate the cache with fresh jobs first.
    
    Args:
        request: SearchJobsRequest with query, location, and limit
        
    Returns:
        SearchJobsResponse with matching jobs from the database
    """
    logger.info(f"Cache search request: query='{request.query}', location='{request.location}', limit={request.limit}")
    
    try:
        # Get jobs from Supabase cache
        jobs = supabase_service.get_jobs(
            query=request.query,
            location=request.location,
            limit=request.limit
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
        
        return SearchJobsResponse(
            success=True,
            message=f"Found {len(job_results)} matching jobs in cache",
            job_count=len(job_results),
            jobs=job_results
        )
        
    except Exception as e:
        logger.exception(f"Error searching cached jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search jobs"
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
