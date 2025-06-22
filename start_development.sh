#!/bin/bash

echo "🚀 AI Memory System - Development Setup"
echo "======================================"

# Check if backend is running
echo "1️⃣ Checking backend status..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is running"
else
    echo "🔄 Starting backend..."
    source .env
    source .venv/bin/activate
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    
    # Wait for backend to start
    echo "⏳ Waiting for backend to start..."
    for i in {1..10}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "✅ Backend started successfully"
            break
        fi
        sleep 1
        echo "   Attempt $i/10..."
    done
fi

echo ""
echo "2️⃣ Backend URLs available:"
echo "   🔗 http://localhost:8000 (primary)"
echo "   🔗 http://127.0.0.1:8000 (alternative)"
echo "   🔗 http://192.168.156.157:8000 (WSL IP)"

echo ""
echo "3️⃣ Next steps:"
echo "   📱 Start frontend: cd frontend && npm start"
echo "   🌐 Press 'w' for web browser"
echo "   📲 Press 'a' for Android simulator"
echo "   🍎 Press 'i' for iOS simulator"

echo ""
echo "4️⃣ Frontend Environment Configuration:"
if [ -f "frontend/.env" ]; then
    echo "   ✅ frontend/.env configured:"
    cat frontend/.env | grep EXPO_PUBLIC_API_BASE_URL | head -1
else
    echo "   ⚠️  No frontend/.env found - using localhost"
fi

echo ""
echo "5️⃣ If frontend can't connect:"
echo "   🚇 Auto setup tunnel: ./setup_tunnel_auto.sh"
echo "   🔧 Manual tunnel: ./setup_tunnel.sh"
echo "   📝 Update URL: ./update_tunnel_url.sh <url>"
echo "   🔧 Use ConnectionDebug component to test"
echo "   📋 Check browser console for errors"

echo ""
echo "📊 Test backend manually:"
echo "   curl http://localhost:8000/health"
echo "   python test_integration.py"

echo ""
echo "🔧 Debug tools available:"
echo "   - ConnectionDebug component in ChatScreen"
echo "   - test_frontend_backend.html"
echo "   - Browser developer console"