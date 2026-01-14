"""
Bright Data Scraping Browser Service for deep Indeed job scraping.

Uses Playwright connect_over_cdp() to connect to Bright Data's remote browser.
IMPORTANT: This does NOT use launch() - all execution happens on Bright Data's infrastructure.

Pricing: Per-session based on Bright Data Scraping Browser pricing.

NOTE: Uses SYNCHRONOUS Playwright API in a ThreadPoolExecutor to avoid Windows asyncio 
subprocess issues with uvicorn --reload. This is the recommended approach for running 
Playwright on Windows with FastAPI/uvicorn.

RELIABILITY: Implements retry with exponential backoff to handle transient network failures.
"""
import asyncio
import logging
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial, wraps
from typing import List, Dict, Any, Optional, Callable, TypeVar

from urllib.parse import quote

# Use SYNC Playwright API to avoid Windows asyncio issues
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeout

from settings import settings

logger = logging.getLogger(__name__)

# ============================================================================
# RETRY CONFIGURATION
# ============================================================================
MAX_RETRIES = 3  # Maximum number of retry attempts
INITIAL_BACKOFF_SECONDS = 2  # Initial backoff delay (doubles each retry)
MAX_BACKOFF_SECONDS = 30  # Maximum backoff delay cap

# Thread pool for running sync Playwright operations
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="playwright_worker")

# Type variable for generic return type
T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
    max_backoff: float = MAX_BACKOFF_SECONDS,
    retryable_exceptions: tuple = (PlaywrightTimeout, RuntimeError, ConnectionError, TimeoutError)
) -> Callable:
    """
    Decorator that implements retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial delay in seconds before first retry
        max_backoff: Maximum delay cap in seconds
        retryable_exceptions: Tuple of exception types that should trigger a retry
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        backoff = min(initial_backoff * (2 ** (attempt - 1)), max_backoff)
                        logger.info(f"Retry attempt {attempt}/{max_retries} after {backoff:.1f}s backoff...")
                        time.sleep(backoff)
                    
                    return func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    error_type = type(e).__name__
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed with {error_type}: {e}. "
                            f"Will retry..."
                        )
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed. Last error ({error_type}): {e}"
                        )
                        
                except Exception as e:
                    # Non-retryable exception - fail immediately
                    logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
                    raise
            
            # All retries exhausted
            raise last_exception  # type: ignore
        
        return wrapper
    return decorator


def _generate_job_id(job_url: str, title: str, company: str) -> str:
    """Generate a unique job ID from job details."""
    # Try to extract Indeed's job key from URL
    jk_match = re.search(r'jk=([a-f0-9]+)', job_url)
    if jk_match:
        return jk_match.group(1)
    
    # Fallback to hash-based ID
    unique_str = f"{job_url}_{title}_{company}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]


def _build_cdp_url() -> str:
    """
    Build the CDP WebSocket URL for Bright Data Scraping Browser.
    
    Format: wss://username:password@host:port
    """
    username = settings.BRIGHT_DATA_USERNAME
    password = settings.BRIGHT_DATA_PASSWORD
    host = settings.BRIGHT_DATA_HOST
    port = settings.BRIGHT_DATA_PORT
    
    logger.info(f"Bright Data config - Host: {host}, Port: {port}, Username set: {bool(username)}, Password set: {bool(password)}")
    
    if not username or not password:
        logger.error("BRIGHT_DATA_USERNAME or BRIGHT_DATA_PASSWORD is not configured!")
        raise ValueError(
            "BRIGHT_DATA_USERNAME and BRIGHT_DATA_PASSWORD must be configured. "
            "Set these in your .env file."
        )
    
    # URL-encode credentials in case of special characters
    auth = f"{quote(username)}:{quote(password)}"
    cdp_url = f"wss://{auth}@{host}:{port}"
    # Log URL without password for security
    safe_url = f"wss://{quote(username)}:****@{host}:{port}"
    logger.info(f"Built CDP URL: {safe_url}")
    return cdp_url


@retry_with_backoff(max_retries=MAX_RETRIES)
def _scrape_indeed_jobs_sync(
    query: str,
    location: str = "",
    max_results: int = 20,
    country_code: str = "US"
) -> List[Dict[str, Any]]:
    """
    SYNCHRONOUS scraping function that runs in a thread pool.
    
    This avoids Windows asyncio subprocess issues with uvicorn --reload.
    Decorated with @retry_with_backoff for automatic retry on transient failures.
    """
    max_results = min(max_results, 100)  # Cap at 100
    
    # Map country code to Indeed domain
    domain_map = {
        "US": "www.indeed.com",
        "GB": "uk.indeed.com",
        "CA": "ca.indeed.com",
        "AU": "au.indeed.com",
        "IN": "in.indeed.com",
        "DE": "de.indeed.com",
        "FR": "fr.indeed.com",
        "NL": "nl.indeed.com",
        "SG": "sg.indeed.com",
    }
    domain = domain_map.get(country_code.upper(), "www.indeed.com")
    
    # Build Indeed search URL
    search_url = f"https://{domain}/jobs?q={quote(query)}"
    if location:
        search_url += f"&l={quote(location)}"
    
    logger.info(f"Starting Bright Data deep scrape: query='{query}', location='{location}', max={max_results}")
    logger.info(f"Target URL: {search_url}")
    
    try:
        cdp_url = _build_cdp_url()
    except ValueError as e:
        logger.error(f"Failed to build CDP URL: {e}")
        raise
    
    jobs: List[Dict[str, Any]] = []
    browser: Optional[Browser] = None
    
    logger.info("Initializing Playwright sync API...")
    with sync_playwright() as playwright:
        logger.info("Playwright initialized successfully")
        try:
            # Connect to Bright Data Scraping Browser via CDP
            logger.info("Attempting to connect to Bright Data Scraping Browser via CDP...")
            logger.info("Connection timeout set to 120 seconds...")
            
            import time
            connect_start = time.time()
            browser = playwright.chromium.connect_over_cdp(
                cdp_url,
                timeout=120000  # 120 second connection timeout for Bright Data
            )
            connect_time = time.time() - connect_start
            logger.info(f"Successfully connected to remote browser in {connect_time:.2f}s")
            logger.info(f"Browser version: {browser.version}")
            logger.info(f"Browser is connected: {browser.is_connected()}")
            
            # Get the default context and page
            logger.info(f"Getting browser context (existing contexts: {len(browser.contexts)})")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            logger.info(f"Got context, getting page (existing pages: {len(context.pages)})")
            page = context.pages[0] if context.pages else context.new_page()
            logger.info("Got page object, ready to navigate")
            
            # Navigate to Indeed search - use 'load' not 'networkidle' as Indeed never reaches idle
            logger.info(f"Navigating to {search_url} (timeout: 90s)...")
            nav_start = time.time()
            page.goto(search_url, wait_until="load", timeout=90000)  # Increased to 90s
            nav_time = time.time() - nav_start
            logger.info(f"Navigation completed in {nav_time:.2f}s")
            logger.info(f"Current page URL: {page.url}")
            logger.info(f"Page title: {page.title()}")
            
            # Give page time to render dynamic content
            page.wait_for_timeout(3000)
            
            # Wait for job cards to load - try multiple selectors
            try:
                page.wait_for_selector('[data-testid="jobsearch-ResultsList"]', timeout=20000)
            except PlaywrightTimeout:
                # Fallback selectors for Indeed's job list
                try:
                    page.wait_for_selector('.jobsearch-ResultsList', timeout=10000)
                except PlaywrightTimeout:
                    page.wait_for_selector('#mosaic-jobResults, .job_seen_beacon', timeout=10000)
            
            # Scrape jobs from current page
            page_jobs = _extract_jobs_from_page_sync(page, domain)
            jobs.extend(page_jobs)
            logger.info(f"Scraped {len(page_jobs)} jobs from page 1")
            
            # Paginate if needed
            page_num = 2
            while len(jobs) < max_results:
                # Check for next page button
                next_button = page.query_selector('[data-testid="pagination-page-next"]')
                if not next_button:
                    logger.info("No more pages available")
                    break
                
                # Click next page
                next_button.click()
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(2000)  # Brief delay for content load
                
                # Extract jobs from new page
                page_jobs = _extract_jobs_from_page_sync(page, domain)
                if not page_jobs:
                    logger.info("No jobs found on page, stopping pagination")
                    break
                    
                jobs.extend(page_jobs)
                logger.info(f"Scraped {len(page_jobs)} jobs from page {page_num} (total: {len(jobs)})")
                page_num += 1
                
                # Safety limit on pages
                if page_num > 10:
                    logger.info("Reached maximum page limit (10)")
                    break
            
        except PlaywrightTimeout as e:
            logger.error(f"Playwright Timeout Error: {e}")
            logger.error(f"This usually means the remote browser couldn't load the page in time.")
            logger.error(f"Possible causes: slow Bright Data connection, Indeed blocking, or network issues.")
            raise RuntimeError(f"Scraping timed out: {e}")
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Error during Bright Data scraping ({error_type}): {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Scraping failed: {e}")
        finally:
            if browser:
                try:
                    logger.info("Closing browser connection...")
                    browser.close()
                    logger.info("Browser connection closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")
            else:
                logger.warning("No browser connection to close")
    
    # Trim to max_results
    jobs = jobs[:max_results]
    logger.info(f"Deep scrape complete: {len(jobs)} jobs retrieved")
    
    return jobs


def _extract_jobs_from_page_sync(page: Page, domain: str) -> List[Dict[str, Any]]:
    """
    Extract job listings from the current Indeed search results page (sync version).
    """
    jobs = []
    
    # Get all job cards
    job_cards = page.query_selector_all('[data-testid="slider_item"]')
    
    for card in job_cards:
        try:
            job = _parse_job_card_sync(card, domain)
            if job and job.get("title"):
                jobs.append(job)
        except Exception as e:
            logger.debug(f"Error parsing job card: {e}")
            continue
    
    return jobs


def _parse_job_card_sync(card, domain: str) -> Dict[str, Any]:
    """
    Parse a single job card element into a job dictionary (sync version).
    """
    job = {
        "title": "",
        "company": "",
        "location": "",
        "salary": "",
        "job_type": "",
        "description": "",
        "apply_url": "",
        "source": "indeed",
        "scraped_at": datetime.utcnow().isoformat(),
    }
    
    # Extract job title
    title_el = card.query_selector('h2.jobTitle span[title], h2.jobTitle a')
    if title_el:
        job["title"] = title_el.inner_text().strip()
    
    # Extract company name
    company_el = card.query_selector('[data-testid="company-name"], .companyName')
    if company_el:
        job["company"] = company_el.inner_text().strip()
    
    # Extract location
    location_el = card.query_selector('[data-testid="text-location"], .companyLocation')
    if location_el:
        job["location"] = location_el.inner_text().strip()
    
    # Extract salary if available
    salary_el = card.query_selector('[data-testid="attribute_snippet_testid"], .salary-snippet-container')
    if salary_el:
        job["salary"] = salary_el.inner_text().strip()
    
    # Extract job type/metadata
    metadata_el = card.query_selector('.metadata')
    if metadata_el:
        job["job_type"] = metadata_el.inner_text().strip()
    
    # Extract job snippet/description
    snippet_el = card.query_selector('.job-snippet, [data-testid="jobDescriptionText"]')
    if snippet_el:
        job["description"] = snippet_el.inner_text().strip()[:500]
    
    # Extract job URL
    link_el = card.query_selector('h2.jobTitle a, a.jcs-JobTitle')
    if link_el:
        href = link_el.get_attribute("href")
        if href:
            if href.startswith("/"):
                job["apply_url"] = f"https://{domain}{href}"
            else:
                job["apply_url"] = href
    
    # Generate job ID
    job["job_id"] = _generate_job_id(
        job["apply_url"], 
        job["title"], 
        job["company"]
    )
    
    return job


async def scrape_indeed_jobs(
    query: str,
    location: str = "",
    max_results: int = 20,
    country_code: str = "US"
) -> List[Dict[str, Any]]:
    """
    Scrape Indeed jobs using Bright Data Scraping Browser.
    
    IMPORTANT: Uses connect_over_cdp() to connect to Bright Data's remote browser.
    NO local browser execution.
    
    This async function runs the sync Playwright code in a thread pool to avoid
    Windows asyncio subprocess issues with uvicorn --reload.
    
    RELIABILITY: Automatically retries up to 3 times with exponential backoff 
    (2s, 4s, 8s delays) on transient failures like timeouts or connection errors.
    
    Args:
        query: Job title/keywords to search for
        location: Location to search in
        max_results: Maximum number of jobs to scrape (default: 20, max: 100)
        country_code: ISO 2-letter country code (default: "US")
    
    Returns:
        List of job dictionaries with title, company, location, salary, url, etc.
    
    Raises:
        ValueError: If credentials are not configured
        RuntimeError: If connection or scraping fails after all retries
    """
    # Run sync Playwright code in thread pool to avoid Windows asyncio issues
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        partial(
            _scrape_indeed_jobs_sync,
            query=query,
            location=location,
            max_results=max_results,
            country_code=country_code
        )
    )


def _test_connection_sync() -> Dict[str, Any]:
    """
    Synchronous connection test function.
    """
    cdp_url = _build_cdp_url()
    
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.connect_over_cdp(
                cdp_url,
                timeout=30000
            )
            
            # Get browser version info
            version = browser.version
            
            browser.close()
            
            return {
                "status": "success",
                "message": "Successfully connected to Bright Data Scraping Browser",
                "browser_version": version
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}",
                "browser_version": None
            }


async def test_connection() -> Dict[str, Any]:
    """
    Test the connection to Bright Data Scraping Browser.
    
    Returns:
        Dict with connection status and details
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _test_connection_sync)
