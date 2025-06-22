#!/bin/bash

echo "🚇 Setting up ngrok tunnel for AI Memory System"
echo "=============================================="

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Installing..."
    
    # Download and install ngrok
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "📥 Downloading ngrok for Linux..."
        curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
        echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
        sudo apt update && sudo apt install ngrok
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "📥 Installing ngrok via Homebrew..."
        brew install ngrok/ngrok/ngrok
    else
        echo "❌ Unsupported OS. Please install ngrok manually: https://ngrok.com/download"
        exit 1
    fi
fi

# Check if backend is running
echo "🩺 Checking if backend is running..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is running on localhost:8000"
else
    echo "❌ Backend not running. Start it first:"
    echo "   source .venv/bin/activate && python main.py"
    exit 1
fi

# Start ngrok tunnel
echo "🚀 Starting ngrok tunnel..."
echo "   This will create a public URL for your backend"
echo "   Use Ctrl+C to stop the tunnel"
echo ""

# Start ngrok in background and capture URL
ngrok http 8000 --log=stdout | while read line; do
    if [[ $line == *"url=https://"* ]]; then
        url=$(echo $line | grep -o 'url=https://[^[:space:]]*' | cut -d= -f2)
        echo "🌐 Tunnel URL: $url"
        echo ""
        echo "📝 Update your frontend config:"
        echo "   BASE_URL: '$url'"
        echo ""
        echo "   Or use the ConnectionStatus component to update the URL dynamically"
        echo ""
    fi
    echo "$line"
done