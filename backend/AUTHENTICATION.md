# Authentication Setup Guide

## Prerequisites

You need to have a Supabase project set up with the credentials in your `.env` file:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service role key (or anon key)

## Database Setup

### 1. Create the Users Table

Go to your Supabase project dashboard:

1. Navigate to **SQL Editor**
2. Run the SQL script from `db/users_table.sql`
3. This will create the `users` table with proper indexes and triggers

Alternatively, you can run:

```sql
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  full_name TEXT,
  hashed_password TEXT NOT NULL,
  disabled BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
```

### 2. Start the Server

The default user will be automatically created on startup:

```bash
uv run uvicorn main:app --reload
```

Watch the logs for:

- "Checking users table in Supabase"
- "Default user initialization completed"

## Testing Authentication

### Option 1: Using FastAPI Docs (Recommended)

1. Go to http://localhost:8000/docs
2. Click on **Authorize** button (top right)
3. Test the login:
   - Click on `POST /auth/login`
   - Click "Try it out"
   - Enter credentials:
     - Username: `sahan`
     - Password: `sahan`
   - Click "Execute"
4. Copy the `access_token` from the response
5. Click **Authorize** button again
6. Paste the token in the format: `<paste_token_here>` (no "Bearer" prefix needed)
7. Click "Authorize"
8. Now test protected endpoints:
   - `GET /auth/me` - Get current user info

### Option 2: Using cURL

1. **Login to get token:**

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=sahan&password=sahan"
```

Response:

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}
```

2. **Use token to access protected endpoint:**

```bash
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer <your_token_here>"
```

Response:

```json
{
  "username": "sahan",
  "email": "sahanpallage19@gmail.com",
  "full_name": "Sahan Pallage",
  "disabled": false,
  "created_at": "2024-12-11T18:52:17.123456+00:00"
}
```

### Option 3: Using Postman/Insomnia

1. **Create a POST request:**

   - URL: `http://localhost:8000/auth/login`
   - Body: `x-www-form-urlencoded`
   - Add fields:
     - `username`: `sahan`
     - `password`: `sahan`

2. **Use the token in subsequent requests:**
   - Create a GET request to `http://localhost:8000/auth/me`
   - Add header: `Authorization: Bearer <your_token>`

## Default User Credentials

- **Username:** `sahan`
- **Password:** `sahan`
- **Email:** `sahanpallage19@gmail.com`
- **Full Name:** Sahan Pallage

## API Endpoints

### Authentication

| Method | Endpoint      | Description                                 | Auth Required |
| ------ | ------------- | ------------------------------------------- | ------------- |
| POST   | `/auth/login` | Login with username/password, get JWT token | No            |
| GET    | `/auth/me`    | Get current user information                | Yes           |

### Legacy Endpoints (Deprecated)

The following endpoints have been replaced:

- `/token` → `/auth/login`
- `/users/me/` → `/auth/me`
- `/users/items` → Removed

## Token Expiration

- Access tokens expire after **30 minutes** by default
- Configure via `ACCESS_TOKEN_EXPIRE_MINUTES` in `settings.py`
- When a token expires, you'll get a 401 Unauthorized response
- Simply login again to get a new token

## JWT Secret Key

The JWT secret key is configured in `settings.py`:

- **Development:** Uses a default hardcoded key
- **Production:** Should be set via `JWT_SECRET_KEY` environment variable in `.env`

To generate a secure key for production:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Add to `.env`:

```
JWT_SECRET_KEY=your_generated_secure_key_here
```

## Troubleshooting

### "Users table may not exist or is inaccessible"

- Run the SQL script in Supabase SQL Editor
- Check `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Verify you're using the service role key for full access

### "Default user initialization failed"

- Check Supabase connection
- Verify users table exists
- Check server logs for detailed error messages

### "Could not validate credentials"

- Token may be expired (30 min default)
- Token format should be: `Authorization: Bearer <token>`
- Login again to get a new token

### Import errors

- Make sure `supabase` package is installed: `uv pip install supabase`
- Restart the server after installing dependencies
