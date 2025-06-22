# 🌐 Environment Setup Guide

This guide explains how to configure the frontend to connect to the backend using environment variables, including tunnel support.

## 📁 Frontend Environment Configuration

The frontend uses environment variables to determine which backend URL to connect to. This allows easy switching between local development and tunnel URLs.

### Environment Variables

Create or edit `frontend/.env`:

```bash
# Primary backend URL - the main URL to use
EXPO_PUBLIC_API_BASE_URL=https://your-tunnel-url.ngrok.io

# Local development fallback
EXPO_PUBLIC_LOCAL_URL=http://localhost:8000

# Tunnel URL (for reference)
TUNNEL_URL=https://your-tunnel-url.ngrok.io

# Development settings
EXPO_PUBLIC_DEBUG_MODE=true
EXPO_PUBLIC_AUTO_CONNECT=true

# Fallback URLs (comma-separated, tried in order)
EXPO_PUBLIC_FALLBACK_URLS=https://your-tunnel-url.ngrok.io,http://localhost:8000,http://127.0.0.1:8000,http://192.168.156.157:8000
```

## 🚀 Quick Setup Scripts

### 1. Auto Setup with Tunnel

```bash
./setup_tunnel_auto.sh
```

This script:
- ✅ Checks if backend is running (starts if needed)
- ✅ Verifies ngrok installation
- ✅ Starts ngrok tunnel
- ✅ Auto-detects tunnel URL
- ✅ Updates frontend `.env` file
- ✅ Tests the connection

### 2. Manual URL Update

```bash
./update_tunnel_url.sh https://abc123.ngrok.io
```

Updates the frontend configuration with a specific tunnel URL.

### 3. Test Configuration

```bash
node test_frontend_env.js
```

Validates environment configuration and tests URLs.

## 🔧 Manual Setup

### Step 1: Start Backend

```bash
# Start backend server
source .env
source .venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Create Tunnel (if needed)

```bash
# Install ngrok (if not installed)
# Visit: https://ngrok.com/download

# Start tunnel
ngrok http 8000
```

Copy the `https://` URL from ngrok output.

### Step 3: Update Frontend Config

```bash
# Method 1: Use script
./update_tunnel_url.sh https://abc123.ngrok.io

# Method 2: Edit manually
nano frontend/.env
```

### Step 4: Start Frontend

```bash
cd frontend
npm start
# Press 'w' for web browser
```

## 🧪 Testing Connection

### 1. Manual URL Test

```bash
# Test backend directly
curl https://your-tunnel-url.ngrok.io/health

# Expected response:
# {"status":"healthy","components":{"memory_graph":"healthy","embedding_search":"healthy"}}
```

### 2. Frontend Debug Panel

The app includes a **ConnectionDebug** component with a collapsible interface:

**Features:**
- ✅ Current environment configuration display
- ✅ Real-time connection status
- ✅ "Run Full Test" for comprehensive API testing
- ✅ Dynamic URL input and testing
- ✅ Floating button in top-left (debug mode only)
- ✅ Keyboard shortcut: `Ctrl+D` or `Cmd+D` (web)

**Access:**
- Click the floating debug button (top-left corner)
- Keyboard shortcut: `Ctrl+D` / `Cmd+D` (web only)
- Only visible when `EXPO_PUBLIC_DEBUG_MODE=true`

### 3. Browser Console

Check browser developer console for detailed logs:
```
🔧 Environment URLs: {...}
🌐 API Configuration: {...}
🔍 Health check: https://...
```

## 🔄 Environment Priority

The frontend tries URLs in this order:

1. **EXPO_PUBLIC_API_BASE_URL** (primary)
2. **EXPO_PUBLIC_LOCAL_URL** (fallback)
3. **EXPO_PUBLIC_FALLBACK_URLS** (additional options)
4. Default: `http://localhost:8000`

## ⚠️ Troubleshooting

### Frontend Shows "Using Mock Data"

**Problem**: Frontend can't connect to backend

**Solutions**:
1. Check backend is running: `curl http://localhost:8000/health`
2. Verify tunnel URL: `curl https://your-tunnel.ngrok.io/health`
3. Update frontend config: `./update_tunnel_url.sh <new-url>`
4. Check browser console for errors
5. Use ConnectionDebug component to test

### Tunnel URL Not Working

**Problem**: Ngrok tunnel is down or expired

**Solutions**:
1. Restart tunnel: `./setup_tunnel_auto.sh`
2. Check ngrok dashboard: https://dashboard.ngrok.com/
3. Update URL: `./update_tunnel_url.sh <new-url>`

### CORS Errors

**Problem**: Cross-origin request blocked

**Solution**: Backend CORS is configured to allow all origins for development. If issues persist:
1. Check backend logs
2. Verify URL format (include https://)
3. Try different tunnel provider

## 📱 Platform Support

### Web Browser (Recommended)
- ✅ Full CORS support
- ✅ Easy debugging with dev tools
- ✅ Works with localhost and tunnel URLs

### Mobile Simulators
- ✅ iOS Simulator: Works with tunnel URLs
- ✅ Android Emulator: Works with tunnel URLs
- ⚠️ May need tunnel for network access

### Physical Devices
- ✅ Requires tunnel URL (ngrok)
- ✅ Both devices must have internet access

## 🔒 Security Notes

- 🚫 **Never commit real API keys to git**
- ✅ Tunnel URLs are temporary and safe for development
- ✅ Use environment variables for configuration
- ✅ Backend CORS is permissive for development only

## 📝 Example Configurations

### Local Development Only
```bash
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
EXPO_PUBLIC_DEBUG_MODE=true
```

### Tunnel for Mobile Testing
```bash
EXPO_PUBLIC_API_BASE_URL=https://abc123.ngrok.io
EXPO_PUBLIC_LOCAL_URL=http://localhost:8000
EXPO_PUBLIC_DEBUG_MODE=true
```

### Production-Ready
```bash
EXPO_PUBLIC_API_BASE_URL=https://your-api.domain.com
EXPO_PUBLIC_DEBUG_MODE=false
EXPO_PUBLIC_AUTO_CONNECT=true
```