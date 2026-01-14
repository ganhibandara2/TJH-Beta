-- Supabase Jobs Table Schema
-- Run this SQL in your Supabase SQL Editor to create the jobs table

CREATE TABLE IF NOT EXISTS jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id TEXT UNIQUE NOT NULL,           -- External job ID (from Indeed or hash)
  title TEXT NOT NULL,
  company TEXT,
  location TEXT,
  salary TEXT,
  job_type TEXT,
  description TEXT,
  apply_url TEXT,
  source TEXT DEFAULT 'indeed',
  scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  search_query TEXT,                      -- Original search query
  search_location TEXT,                   -- Original search location
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_jobs_scraped_at ON jobs(scraped_at);
CREATE INDEX IF NOT EXISTS idx_jobs_search_query ON jobs(search_query);

-- Create a trigger to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_jobs_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_jobs_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;

-- Create policy to allow service role full access
CREATE POLICY "Service role can manage jobs" ON jobs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Create policy for authenticated users to read jobs
CREATE POLICY "Authenticated users can read jobs" ON jobs
    FOR SELECT
    TO authenticated
    USING (true);
