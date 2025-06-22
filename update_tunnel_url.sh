#!/bin/bash

# Script to update frontend .env with new tunnel URL

if [ -z "$1" ]; then
    echo "🚇 Update Frontend Tunnel URL"
    echo "============================"
    echo ""
    echo "Usage: $0 <tunnel-url>"
    echo ""
    echo "Example:"
    echo "  $0 https://abc123.ngrok.io"
    echo ""
    echo "Current .env configuration:"
    if [ -f "frontend/.env" ]; then
        echo "---"
        cat frontend/.env | grep -E "(API_BASE_URL|TUNNEL_URL|FALLBACK_URLS)"
        echo "---"
    else
        echo "  No frontend/.env file found"
    fi
    echo ""
    exit 1
fi

TUNNEL_URL="$1"

# Validate URL format
if [[ ! $TUNNEL_URL =~ ^https?:// ]]; then
    echo "❌ Invalid URL format. Please include https:// or http://"
    exit 1
fi

echo "🔄 Updating frontend .env with tunnel URL: $TUNNEL_URL"

# Create backup
if [ -f "frontend/.env" ]; then
    cp frontend/.env frontend/.env.backup
    echo "📋 Backup created: frontend/.env.backup"
fi

# Update the .env file
cat > frontend/.env << EOF
# AI Memory System Frontend Configuration

# Backend API URL - Primary URL to use
EXPO_PUBLIC_API_BASE_URL=${TUNNEL_URL}

# Local development fallback
EXPO_PUBLIC_LOCAL_URL=http://localhost:8000

# Tunnel URL (ngrok)
TUNNEL_URL=${TUNNEL_URL}

# Development settings
EXPO_PUBLIC_DEBUG_MODE=true
EXPO_PUBLIC_AUTO_CONNECT=true

# Fallback URLs (comma-separated)
EXPO_PUBLIC_FALLBACK_URLS=${TUNNEL_URL},http://localhost:8000,http://127.0.0.1:8000,http://192.168.156.157:8000
EOF

echo "✅ Frontend .env updated successfully!"
echo ""
echo "📋 New configuration:"
echo "---"
cat frontend/.env
echo "---"
echo ""
echo "🔄 Next steps:"
echo "1. Restart your frontend development server"
echo "2. The app will automatically use the new URL"
echo "3. Check the ConnectionDebug component to verify connection"
echo ""
echo "💡 To test the connection:"
echo "   curl ${TUNNEL_URL}/health"