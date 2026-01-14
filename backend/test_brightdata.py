"""
Test script for Bright Data Indeed scraping.
Tests the synchronous API with the specified parameters.
"""
import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("BRIGHTDATA_API_KEY")
DATASET_ID = os.getenv("BRIGHTDATA_DATASET_ID", "gd_l4dx9j9sscpvs7no2")

if not API_KEY:
    print("ERROR: BRIGHTDATA_API_KEY not set in .env file")
    exit(1)

print(f"API Key: {API_KEY[:10]}...")
print(f"Dataset ID: {DATASET_ID}")

# Test parameters based on user's screenshot
payload = [
    {
        "country": "US",
        "domain": "indeed.com",
        "keyword_search": "Software Engineer",
        "location": "San Francisco",
        "date_posted": "Last 7 days",
        "posted_by": "",
        "location_radius": ""
    }
]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Use synchronous /scrape endpoint with limit of 10
scrape_url = f"https://api.brightdata.com/datasets/v3/scrape?dataset_id={DATASET_ID}&include_errors=true&type=discover_new&discover_by=keyword&limit_per_input=10&format=json"

print(f"\n{'='*60}")
print("SENDING REQUEST TO BRIGHT DATA")
print(f"{'='*60}")
print(f"URL: {scrape_url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

start_time = time.time()

try:
    print("\nSending request...")
    response = requests.post(scrape_url, json=payload, headers=headers, timeout=90)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 202:
        # Need to poll
        snapshot_id = response.json().get("snapshot_id")
        print(f"Got snapshot_id: {snapshot_id}")
        print("Polling for results...")
        
        results_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"
        
        for attempt in range(8):  # ~2 minutes
            time.sleep(15)
            print(f"  Poll attempt {attempt + 1}...")
            
            poll_response = requests.get(results_url, headers=headers, timeout=30)
            response_text = poll_response.text
            
            # Check if still processing
            if "not ready" in response_text or '"status":"running"' in response_text or '"status":"starting"' in response_text:
                print(f"  Still processing...")
                continue
            
            response = poll_response
            break
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"RESPONSE RECEIVED (after {elapsed:.1f}s)")
    print(f"{'='*60}")
    
    response_text = response.text
    print(f"Response length: {len(response_text)} chars")
    
    # Parse JSON
    try:
        data = json.loads(response_text)
        
        if isinstance(data, list):
            print(f"Got JSON array with {len(data)} items")
            
            # Filter out status objects and error objects
            jobs = [item for item in data if isinstance(item, dict) and not item.get("status") and not item.get("error")]
            error_items = [item for item in data if isinstance(item, dict) and item.get("error")]
            
            print(f"Filtered to {len(jobs)} job items (excluding {len(error_items)} errors)")
            
            if error_items:
                print(f"\n{'='*60}")
                print(f"ERRORS FOUND ({len(error_items)}):")
                print(f"{'='*60}")
                for err in error_items[:3]:  # Show first 3 errors
                    print(f"  - {err.get('error')}: {err.get('error_code')}")
            
            if jobs:
                print(f"\n{'='*60}")
                print("FIRST VALID JOB FOUND:")
                print(f"{'='*60}")
                first_job = jobs[0]
                print(json.dumps(first_job, indent=2)[:3000])
                
                print(f"\n{'='*60}")
                print("ALL JOB FIELDS IN FIRST RESULT:")
                print(f"{'='*60}")
                for key in first_job.keys():
                    value = first_job[key]
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"  {key}: {value}")
                
                # Show summary of all valid jobs
                print(f"\n{'='*60}")
                print(f"ALL {len(jobs)} VALID JOBS:")
                print(f"{'='*60}")
                for i, job in enumerate(jobs[:10]):  # Show first 10
                    title = job.get("job_title") or job.get("title") or job.get("name") or "Unknown"
                    company = job.get("company_name") or job.get("company") or "Unknown"
                    print(f"  {i+1}. {title} at {company}")
                    
            else:
                print("\nNo valid jobs found in response!")
                print("All items are error objects.")
                print(f"First 1000 chars of response: {response_text[:1000]}")

        
        elif isinstance(data, dict):
            print("Got single JSON object")
            print(json.dumps(data, indent=2)[:1000])
            
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"First 500 chars: {response_text[:500]}")

except requests.Timeout:
    print("Request timed out!")
except requests.RequestException as e:
    print(f"Request error: {e}")
except Exception as e:
    print(f"Error: {e}")

print(f"\n{'='*60}")
print(f"TEST COMPLETE")
print(f"{'='*60}")
