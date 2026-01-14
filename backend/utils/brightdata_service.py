"""
Bright Data Indeed Job Scraper Service.
Replacement for Apify-based Indeed scraping with cost-effective Bright Data API.
Pricing: ~$0.001 per record vs Apify's actor-based pricing.
"""
import json
import logging
import time
import re
import requests

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger(__name__)

# Bright Data API endpoints
BRIGHTDATA_API_BASE = "https://api.brightdata.com/datasets/v3"


def _parse_date(date_str: Optional[str]) -> str:
    """
    Parse date strings from Bright Data response.
    Handles ISO format and relative dates.
    
    Args:
        date_str: Date string (e.g., "2024-01-15", "2 days ago")
    
    Returns:
        Normalized date string in ISO format (YYYY-MM-DD)
    """
    if not date_str:
        return ""
    
    date_str = str(date_str).strip()
    
    # Try ISO format first
    try:
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('T')[0])
            return dt.strftime('%Y-%m-%d')
    except (ValueError, AttributeError):
        pass
    
    # Parse relative dates
    now = datetime.now()
    
    day_match = re.search(r'(\d+)\s+day', date_str.lower())
    if day_match:
        days = int(day_match.group(1))
        return (now - timedelta(days=days)).strftime('%Y-%m-%d')
    
    week_match = re.search(r'(\d+)\s+week', date_str.lower())
    if week_match:
        weeks = int(week_match.group(1))
        return (now - timedelta(weeks=weeks)).strftime('%Y-%m-%d')
    
    month_match = re.search(r'(\d+)\s+month', date_str.lower())
    if month_match:
        months = int(month_match.group(1))
        return (now - timedelta(days=months * 30)).strftime('%Y-%m-%d')
    
    hour_match = re.search(r'(\d+)\s+hour', date_str.lower())
    if hour_match:
        return now.strftime('%Y-%m-%d')
    
    if "just now" in date_str.lower() or "today" in date_str.lower():
        return now.strftime('%Y-%m-%d')
    
    if "yesterday" in date_str.lower():
        return (now - timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.debug(f"Could not parse date: {date_str}")
    return date_str


def _filter_jobs_by_date(jobs: List[Dict[str, Any]], date_posted: Optional[str]) -> List[Dict[str, Any]]:
    """
    Filter jobs based on date_posted criteria.
    
    Args:
        jobs: List of job dictionaries
        date_posted: Filter criteria ("24h", "day", "today", "week", "anytime", "all")
    
    Returns:
        Filtered list of jobs
    """
    if not date_posted or date_posted.lower() in ["anytime", "all", ""]:
        return jobs
    
    date_lower = date_posted.lower()
    now = datetime.now()
    filtered = []
    
    for job in jobs:
        job_date_str = job.get("date_posted", "")
        if not job_date_str:
            filtered.append(job)
            continue
        
        try:
            if re.match(r'^\d{4}-\d{2}-\d{2}', job_date_str):
                job_date = datetime.strptime(job_date_str, '%Y-%m-%d')
                diff = (now - job_date).days
                
                if "day" in date_lower or "today" in date_lower or "24h" in date_lower:
                    if diff <= 1:
                        filtered.append(job)
                elif "week" in date_lower:
                    if diff <= 7:
                        filtered.append(job)
                elif "month" in date_lower:
                    if diff <= 30:
                        filtered.append(job)
                else:
                    filtered.append(job)
            else:
                filtered.append(job)
        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse job date: {e}")
            filtered.append(job)
    
    return filtered


def search_indeed_jobs_brightdata(
    job_title: str,
    location: str = "",
    max_results: int = 20,
    date_posted: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str, Optional[str]], None]] = None,
    country_code: Optional[str] = None,
    api_key: Optional[str] = None,
    dataset_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for jobs on Indeed using Bright Data API.
    
    Args:
        job_title: Job title to search for
        location: Location to search in
        max_results: Maximum number of jobs to return (default: 20)
        date_posted: Filter by date posted
        progress_callback: Optional callback for progress updates
        country_code: ISO 2-letter country code (e.g., "US", "GB")
        api_key: Bright Data API key (from settings if not provided)
        dataset_id: Bright Data Indeed dataset ID
    
    Returns:
        List of job dictionaries
    
    Raises:
        ValueError: If API key or dataset ID is not configured
        RuntimeError: If API request fails
    """
    # Import settings here to avoid circular imports
    from settings import settings
    
    api_key = api_key or getattr(settings, 'BRIGHTDATA_API_KEY', '')
    dataset_id = dataset_id or getattr(settings, 'BRIGHTDATA_DATASET_ID', '')
    
    if not api_key:
        raise ValueError("BRIGHTDATA_API_KEY is not configured")
    if not dataset_id:
        raise ValueError("BRIGHTDATA_DATASET_ID is not configured")
    
    if not job_title:
        raise ValueError("job_title is required")
    
    logger.info(f"Starting Bright Data SYNCHRONOUS search for '{job_title}' in '{location}' (country: {country_code or 'US'})")
    
    # Use SYNCHRONOUS /scrape endpoint for immediate results (max 60s wait)
    scrape_url = f"{BRIGHTDATA_API_BASE}/scrape?dataset_id={dataset_id}&include_errors=true&type=discover_new&discover_by=keyword&limit_per_input={max_results}&format=json"
    
    # Map country code to Indeed domain
    country = country_code.upper() if country_code else "US"
    domain_map = {
        "US": "indeed.com",
        "GB": "uk.indeed.com",
        "CA": "ca.indeed.com",
        "AU": "au.indeed.com",
        "IN": "in.indeed.com",
        "DE": "de.indeed.com",
        "FR": "fr.indeed.com",
        "NL": "nl.indeed.com",
        "SG": "sg.indeed.com",
    }
    domain = domain_map.get(country, "indeed.com")
    
    # Map date_posted to Bright Data format
    date_filter = ""
    if date_posted:
        date_lower = date_posted.lower()
        if "day" in date_lower or "24h" in date_lower or "today" in date_lower:
            date_filter = "Last 24 hours"
        elif "week" in date_lower:
            date_filter = "Last 7 days"
        elif "month" in date_lower:
            date_filter = "Last 30 days"
    
    payload = [
        {
            "country": country,
            "domain": domain,
            "keyword_search": job_title,
            "location": location if location else "",
            "date_posted": date_filter,
            "posted_by": "",
            "location_radius": ""
        }
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    logger.info(f"Bright Data SYNC request: keyword='{job_title}', location='{location}', country='{country}'")
    
    if progress_callback:
        progress_callback(0, max_results, "REQUESTING", None)
    
    jobs_data = []
    
    try:
        # Synchronous request - returns results directly (up to 60s timeout on their end)
        # Use longer client timeout to accommodate
        response = requests.post(scrape_url, json=payload, headers=headers, timeout=90)
        
        if response.status_code == 202:
            # Still processing - get snapshot_id and poll
            snapshot_id = response.json().get("snapshot_id")
            logger.info(f"Request still processing, got snapshot_id: {snapshot_id}")
            
            if progress_callback:
                progress_callback(0, max_results, "PROCESSING", snapshot_id)
            
            # Poll for results until ready
            results_url = f"{BRIGHTDATA_API_BASE}/snapshot/{snapshot_id}?format=json"
            for attempt in range(4):  # Try for ~1 minute (4 x 15s)

                time.sleep(15)
                logger.info(f"Polling snapshot {snapshot_id}... attempt {attempt + 1}")
                
                poll_response = requests.get(results_url, headers=headers, timeout=30)
                response_text = poll_response.text.strip()
                
                # Check if still processing - look for ALL possible status values
                is_still_processing = any(status_phrase in response_text for status_phrase in [
                    '"status":"running"', '"status": "running"',
                    '"status":"starting"', '"status": "starting"',
                    '"status":"pending"', '"status": "pending"',
                    '"status":"processing"', '"status": "processing"',
                    "not ready yet"
                ])
                
                if not is_still_processing:
                    response = poll_response
                    logger.info("Snapshot ready - data available")
                    break
                else:
                    logger.info(f"Snapshot still processing...")
                    if progress_callback:
                        progress_callback(0, max_results, f"SCRAPING ({(attempt + 1) * 15}s)", snapshot_id)
        
        response.raise_for_status()
        response_text = response.text.strip()
        
        logger.info(f"Response length: {len(response_text)} chars")
        logger.info(f"Response preview: {response_text[:500]}..." if len(response_text) > 500 else f"Response: {response_text}")
        
        # Check if this is a status-only response (not actual job data)
        if len(response_text) < 200:
            try:
                maybe_status = json.loads(response_text)
                if isinstance(maybe_status, dict) and maybe_status.get("status"):
                    logger.warning(f"Got status response instead of jobs: {maybe_status}")
                    # Return empty - snapshot not ready
                    jobs_data = []
                    logger.info(f"Parsed {len(jobs_data)} raw items from response")
                    raise RuntimeError(f"Bright Data snapshot not ready: {maybe_status.get('message', 'still processing')}")
            except json.JSONDecodeError:
                pass
        
        # Parse response - should be JSON array of jobs
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                # Filter out any status objects
                jobs_data = [item for item in parsed if not (isinstance(item, dict) and item.get("status"))]
                logger.info(f"Parsed JSON array with {len(jobs_data)} job items")
            elif isinstance(parsed, dict):
                if parsed.get("status"):
                    logger.warning(f"Got status object: {parsed}")
                    jobs_data = []
                elif "data" in parsed:
                    jobs_data = parsed["data"]
                else:
                    jobs_data = [parsed]
        except json.JSONDecodeError:
            # Try NDJSON format
            for line in response_text.split('\n'):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and obj.get("status"):
                            continue  # Skip status objects
                        if isinstance(obj, list):
                            jobs_data.extend(obj)
                        elif isinstance(obj, dict):
                            jobs_data.append(obj)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Parsed {len(jobs_data)} raw items from response")


        
    except requests.HTTPError as e:
        logger.error(f"Bright Data HTTP error: {e}")
        logger.error(f"Response: {e.response.text[:500] if e.response else 'No response'}")
        if e.response and e.response.status_code == 401:
            raise ValueError("Invalid Bright Data API key")
        raise RuntimeError(f"Bright Data API error: {e}")
    except requests.Timeout:
        logger.error("Bright Data request timed out")
        raise RuntimeError("Bright Data request timed out - try again")
    except requests.RequestException as e:
        logger.error(f"Network error: {e}")
        raise RuntimeError(f"Network error: {e}")
    
    logger.info(f"Final result: {len(jobs_data)} jobs retrieved")



    # Normalize job data to match existing format
    cleaned_jobs = []
    for idx, job in enumerate(jobs_data[:max_results]):
        # Skip items that are status/metadata objects
        if job.get("status") in ["running", "ready", "failed", "processing"]:
            continue
            
        # Bright Data may nest job data - check common locations
        # The actual job fields might be at top level or nested
        def get_field(field_names, default=""):
            """Try to get field from job object or nested 'input' object"""
            for name in field_names:
                # Check top level
                if job.get(name):
                    return job.get(name)
                # Check inside 'input' key
                if job.get("input") and isinstance(job.get("input"), dict):
                    if job["input"].get(name):
                        return job["input"].get(name)
            return default
        
        # Extract job title
        job_title_val = get_field(["job_title", "title", "positionName", "position_name", "name"])
        
        # Extract company
        company = get_field(["company_name", "company", "employer", "employer_name"])
        
        # Extract location
        location_str = get_field(["location", "job_location", "city", "address"])
        
        # Get job URL - check multiple possible field names
        job_url = get_field(["apply_link", "url", "job_url", "link", "externalApplyLink"])
        
        # If no job_url found, try to extract from input.url
        if not job_url and job.get("input") and isinstance(job.get("input"), dict):
            job_url = job["input"].get("url", "")
        
        # Get description
        description = get_field(["description_text", "description", "snippet", "job_description", "full_description"])
        
        # Get salary
        salary = get_field(["salary_formatted", "salary", "salary_range", "compensation", "salary_text"])
        
        # Get date posted
        raw_date = get_field(["date_posted", "posted_at", "post_date", "timestamp", "posted_date"])
        normalized_date = _parse_date(raw_date)
        
        # Skip if no title found - probably metadata object
        if not job_title_val:
            logger.debug(f"Skipping item {idx} - no job title found. Keys: {list(job.keys())[:10]}")
            continue
        
        # Parse location into components
        city, state, country_val = "", "", ""
        if location_str:
            parts = [p.strip() for p in location_str.split(",")]
            if len(parts) >= 1:
                city = parts[0]
            if len(parts) >= 2:
                state = parts[1]
            if len(parts) >= 3:
                country_val = parts[2]
        
        # Determine employment type and remote status
        job_type = get_field(["job_type", "employment_type", "type"])
        remote = False
        if isinstance(job_type, str):
            job_type_lower = job_type.lower()
            if "remote" in job_type_lower or "hybrid" in job_type_lower:
                remote = True

        
        cleaned_jobs.append({
            "title": job_title_val,
            "company": company,
            "location": location_str,
            "city": city,
            "state": state,
            "country": country_val,
            "url": job_url,
            "date_posted": normalized_date,
            "description": description[:500] if description else "",
            "employment_type": job_type if isinstance(job_type, str) else "",
            "remote": remote,
            "salary": salary if isinstance(salary, str) else str(salary) if salary else "",
            "rating": job.get("company_rating"),
            "reviews_count": job.get("reviews_count"),
        })
    
    logger.info(f"Cleaned and normalized {len(cleaned_jobs)} jobs")
    
    # Apply date filter
    if date_posted:
        original_count = len(cleaned_jobs)
        cleaned_jobs = _filter_jobs_by_date(cleaned_jobs, date_posted)
        logger.info(f"Filtered {original_count} to {len(cleaned_jobs)} jobs by date: {date_posted}")
    
    if progress_callback:
        progress_callback(len(cleaned_jobs), max_results, "SUCCEEDED", snapshot_id)
    
    return cleaned_jobs


def normalize_brightdata_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Bright Data job to match the expected response format.
    
    Args:
        job: Job dictionary from Bright Data
    
    Returns:
        Normalized job dictionary with consistent field names
    """
    import hashlib
    
    job_url = job.get("url", "")
    
    # Generate job ID
    if job_url:
        jk_match = re.search(r'jk=([a-f0-9]+)', job_url)
        if jk_match:
            job_id = jk_match.group(1)
        else:
            unique_str = f"{job_url}_{job.get('title', '')}_{job.get('company', '')}"
            job_id = hashlib.md5(unique_str.encode()).hexdigest()[:12]
    else:
        unique_str = f"{job.get('title', '')}_{job.get('company', '')}"
        job_id = hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    return {
        "job_id": job_id,
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "location": job.get("location", ""),
        "city": job.get("city", ""),
        "state": job.get("state", ""),
        "country": job.get("country", ""),
        "link": job.get("url", ""),
        "date_posted": job.get("date_posted", ""),
        "description": job.get("description", ""),
        "employment_type": job.get("employment_type", ""),
        "remote": job.get("remote", False),
        "salary": job.get("salary", ""),
    }
