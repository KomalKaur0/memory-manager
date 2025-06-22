#!/usr/bin/env python3
"""
Check network configuration to determine if tunnel is needed
"""
import socket
import subprocess
import json
import requests

def get_network_info():
    """Get network interface information"""
    print("🌐 Network Configuration Check")
    print("=" * 40)
    
    # Check localhost interfaces
    try:
        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"📍 Local IP: {local_ip}")
    except Exception as e:
        print(f"❌ Could not determine local IP: {e}")
        local_ip = "unknown"
    
    # Check if backend is accessible on different interfaces
    test_urls = [
        "http://localhost:8000/health",
        "http://127.0.0.1:8000/health",
        f"http://{local_ip}:8000/health",
        "http://0.0.0.0:8000/health",
    ]
    
    print("\n🧪 Testing Backend Accessibility:")
    working_urls = []
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {url} - WORKS")
                working_urls.append(url)
            else:
                print(f"❌ {url} - HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {url} - {str(e)[:50]}...")
    
    return {
        "local_ip": local_ip,
        "working_urls": working_urls,
    }

def check_docker_desktop():
    """Check if running in Docker Desktop environment"""
    try:
        # Check for Docker Desktop WSL integration
        result = subprocess.run(['uname', '-a'], capture_output=True, text=True)
        if 'microsoft' in result.stdout.lower() or 'wsl' in result.stdout.lower():
            print("🐳 WSL/Docker Desktop environment detected")
            return True
    except:
        pass
    return False

def recommend_solution(network_info):
    """Recommend solution based on network configuration"""
    print("\n💡 Recommendations:")
    
    if len(network_info["working_urls"]) > 0:
        print("✅ Backend is accessible locally")
        print("📱 Frontend should be able to connect if on same machine")
        
        if network_info["local_ip"] != "unknown":
            suggested_url = f"http://{network_info['local_ip']}:8000"
            print(f"🔧 Try using local IP in frontend: {suggested_url}")
        
        print("\n🎯 Next steps:")
        print("1. Check frontend logs for detailed error messages")
        print("2. Verify CORS configuration includes frontend origin")
        print("3. Ensure no firewall blocking requests")
        
    else:
        print("❌ Backend not accessible on any interface")
        print("🔧 Possible solutions:")
        print("1. Restart backend with: python -m uvicorn main:app --host 0.0.0.0 --port 8000")
        print("2. Check if backend is actually running")
        print("3. Check for port conflicts")
    
    # Check if tunnel might be needed
    is_docker = check_docker_desktop()
    
    if is_docker:
        print("\n🚇 Tunnel might be needed for cross-network access:")
        print("1. Install ngrok: https://ngrok.com/download")
        print("2. Run: ngrok http 8000")
        print("3. Use the ngrok URL in frontend config")
    
    print("\n📱 For Expo development:")
    print("- Expo web runs on localhost:19006 by default")
    print("- Expo mobile simulators may need local IP or tunnel")

if __name__ == "__main__":
    network_info = get_network_info()
    recommend_solution(network_info)
    
    print("\n🔍 Debug Information:")
    print(json.dumps(network_info, indent=2))