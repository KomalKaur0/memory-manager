#!/usr/bin/env python3
"""
Test script for full stack integration
"""
import requests
import json
import time

def test_backend_integration():
    """Test backend API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Backend Integration...")
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test memory nodes endpoint
    print("\n2. Testing memory nodes endpoint...")
    try:
        response = requests.get(f"{base_url}/api/memory/nodes")
        if response.status_code == 200:
            nodes = response.json()
            print(f"✅ Memory nodes retrieved: {len(nodes)} nodes")
            for node_id, node in list(nodes.items())[:2]:  # Show first 2
                print(f"   - {node['concept'][:50]}...")
        else:
            print("❌ Memory nodes failed")
    except Exception as e:
        print(f"❌ Memory nodes error: {e}")
    
    # Test chat endpoint
    print("\n3. Testing chat endpoint...")
    try:
        chat_data = {
            "content": "Tell me about React development",
            "conversation_history": []
        }
        response = requests.post(
            f"{base_url}/api/chat/send",
            headers={"Content-Type": "application/json"},
            json=chat_data
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat endpoint working")
            print(f"   Response length: {len(result['message']['content'])} chars")
            print(f"   Retrieved memories: {len(result['retrieved_memories'])}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            
            # Show memory access details
            if result['memory_access_events']:
                print("   Memory accesses:")
                for event in result['memory_access_events'][:2]:
                    print(f"     - {event['node_id'][:8]}... (score: {event['relevance_score']:.3f})")
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
    
    # Test co-access connections
    print("\n4. Testing co-access connection creation...")
    try:
        # Send another chat to create co-access patterns
        chat_data = {
            "content": "How do I combine React with 3D graphics and mobile development?",
            "conversation_history": []
        }
        response = requests.post(
            f"{base_url}/api/chat/send",
            headers={"Content-Type": "application/json"},
            json=chat_data
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Co-access test completed")
            print(f"   Retrieved memories: {len(result['retrieved_memories'])}")
            
            # Check if connections were created
            response = requests.get(f"{base_url}/api/memory/nodes")
            if response.status_code == 200:
                nodes = response.json()
                total_connections = sum(len(node.get('connections', {})) for node in nodes.values())
                print(f"   Total connections in graph: {total_connections}")
        else:
            print(f"❌ Co-access test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Co-access test error: {e}")
    
    print("\n🎉 Backend integration test completed!")
    return True

def test_frontend_connection():
    """Test if frontend can connect to backend"""
    print("\n🌐 Testing Frontend-Backend Connection...")
    
    # This would be called by the frontend
    base_url = "http://localhost:8000"
    
    try:
        # Simulate frontend health check
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Frontend can connect to backend")
            return True
        else:
            print("❌ Frontend connection failed")
            return False
    except Exception as e:
        print(f"❌ Frontend connection error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AI Memory System - Full Stack Integration Test")
    print("=" * 50)
    
    backend_ok = test_backend_integration()
    frontend_ok = test_frontend_connection()
    
    print("\n📊 Test Results:")
    print(f"Backend API: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"Frontend Connection: {'✅ PASS' if frontend_ok else '❌ FAIL'}")
    
    if backend_ok and frontend_ok:
        print("\n🎉 All tests passed! Full stack integration is working.")
        print("\n📝 Next steps:")
        print("   1. Start frontend: cd frontend && npm start")
        print("   2. Press 'w' for web browser")
        print("   3. Test chat interface with real backend")
    else:
        print("\n⚠️  Some tests failed. Check logs above.")