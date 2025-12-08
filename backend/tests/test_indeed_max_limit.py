"""
Test script to verify Indeed scraper respects max_results limit.
This script tests the max limit functionality without manual intervention.

Usage:
    python test_indeed_max_limit.py
"""
import os
import sys
import time
import logging
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.indeed_service import search_indeed_jobs
from settings import APIFY_API_KEY, APIFY_ACTOR_ID

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_max_limit(max_results: int = 20, test_name: str = "software engineer"):
    """
    Test that the Indeed scraper respects the max_results limit.
    
    Args:
        max_results: Maximum number of results to collect
        test_name: Job title to search for
    """
    logger.info("=" * 80)
    logger.info(f"TEST: Max Results Limit (target: {max_results})")
    logger.info("=" * 80)
    
    if not APIFY_API_KEY:
        logger.error("APIFY_API_KEY not set. Please set it in your .env file.")
        return False
    
    start_time = time.time()
    
    try:
        # Track progress
        progress_updates = []
        
        def progress_callback(count: int, max_count: int, status: str, run_id: str = None):
            """Track progress updates"""
            progress_updates.append({
                'count': count,
                'max_count': max_count,
                'status': status,
                'run_id': run_id,
                'timestamp': time.time()
            })
            logger.info(f"Progress: {count}/{max_count} results (status: {status})")
        
        # Run the search
        logger.info(f"Starting search for '{test_name}' with max_results={max_results}")
        jobs = search_indeed_jobs(
            job_title=test_name,
            location="United States",
            max_results=max_results,
            date_posted=None,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        
        # Analyze results
        actual_count = len(jobs)
        logger.info("=" * 80)
        logger.info("TEST RESULTS:")
        logger.info("=" * 80)
        logger.info(f"Requested max_results: {max_results}")
        logger.info(f"Actual results returned: {actual_count}")
        logger.info(f"Time elapsed: {elapsed:.2f} seconds")
        logger.info(f"Progress updates received: {len(progress_updates)}")
        
        # Check if limit was respected
        if actual_count > max_results:
            logger.error(f"❌ FAIL: Returned {actual_count} results, exceeded limit of {max_results}")
            logger.error(f"   Difference: {actual_count - max_results} extra results")
            return False
        elif actual_count == max_results:
            logger.info(f"✅ PASS: Returned exactly {max_results} results (perfect match)")
        else:
            logger.warning(f"⚠️  WARNING: Returned {actual_count} results, less than limit of {max_results}")
            logger.warning(f"   This is acceptable if there weren't enough jobs available")
        
        # Check progress updates
        if progress_updates:
            max_progress_count = max(p['count'] for p in progress_updates)
            logger.info(f"Maximum count seen during progress: {max_progress_count}")
            
            if max_progress_count > max_results:
                logger.warning(f"⚠️  WARNING: Actor collected {max_progress_count} results before abort")
                logger.warning(f"   But final result was correctly limited to {actual_count}")
            else:
                logger.info(f"✅ Progress tracking showed max of {max_progress_count} (within limit)")
        
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST FAILED with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    load_dotenv()
    
    # Run test with max_results = 20
    success = test_max_limit(max_results=20, test_name="software engineer")
    
    if success:
        logger.info("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Tests failed!")
        sys.exit(1)

