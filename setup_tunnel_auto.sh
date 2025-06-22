#!/bin/bash

echo "🚇 Auto-Setup Tunnel for AI Memory System"
echo "========================================"

# Check if backend is running
echo "1️⃣ Checking backend status..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Backend not running. Starting backend first..."
    source .env
    source .venv/bin/activate
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    
    echo "⏳ Waiting for backend to start..."
    for i in {1..15}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "✅ Backend started successfully"
            break
        fi
        sleep 2
        echo "   Attempt $i/15..."
    done
    
    if ! curl -s http://localhost:8000/health > /dev/null; then
        echo "❌ Failed to start backend. Please start it manually."
        exit 1
    fi
fi

echo "✅ Backend is running"

# Check if ngrok is installed
echo ""
echo "2️⃣ Checking ngrok installation..."
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Please install it first:"
    echo "   https://ngrok.com/download"
    echo ""
    echo "   Or run: ./setup_tunnel.sh"
    exit 1
fi

echo "✅ ngrok is installed"

# Start ngrok and capture URL
echo ""
echo "3️⃣ Starting ngrok tunnel..."
echo "   This will create a public HTTPS URL for your backend"

# Kill any existing ngrok processes
pkill -f "ngrok http" 2>/dev/null

# Start ngrok in background and capture output
ngrok http 8000 --log=stdout 2>&1 | while read line; do
    echo "$line"
    
    # Extract the HTTPS URL when ngrok starts
    if [[ $line == *"url=https://"* ]]; then
        tunnel_url=$(echo $line | grep -o 'url=https://[^[:space:]]*' | cut -d= -f2)
        
        if [[ -n "$tunnel_url" ]]; then
            echo ""
            echo "🎉 Tunnel URL detected: $tunnel_url"
            
            # Update frontend .env file
            echo "4️⃣ Updating frontend configuration..."
            ./update_tunnel_url.sh "$tunnel_url"
            
            echo ""
            echo "✅ Setup complete! Your tunnel is ready:"
            echo "   🌐 Tunnel URL: $tunnel_url"
            echo "   🔗 Health check: $tunnel_url/health"
            echo ""
            echo "📱 Next steps:"
            echo "   1. Start/restart your frontend: cd frontend && npm start"
            echo "   2. The app will automatically use the tunnel URL"
            echo "   3. Test the connection using the debug panel"
            echo ""
            echo "🛑 Press Ctrl+C to stop the tunnel"
            echo ""
            
            # Test the tunnel
            echo "🧪 Testing tunnel connection..."
            if curl -s "$tunnel_url/health" > /dev/null; then
                echo "✅ Tunnel is working!"
            else
                echo "⚠️  Tunnel may not be ready yet, wait a moment and try again"
            fi
        fi
    fi
done &

# Wait for ngrok to start
sleep 3

echo ""
echo "ℹ️  If the tunnel URL wasn't automatically detected:"
echo "   1. Look for the ngrok URL in the output above"
echo "   2. Run: ./update_tunnel_url.sh <your-ngrok-url>"
echo "   3. Restart your frontend development server"