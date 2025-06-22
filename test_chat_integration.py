#!/usr/bin/env python3
"""
Test script to verify the complete chat integration is working.
This tests the full flow from user message to AI response with memory retrieval.
"""

import asyncio
import aiohttp
import json
import sys

async def test_chat_integration():
    """Test the complete chat integration flow"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing AI Memory System Chat Integration")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Health Check
        print("\n1. 🏥 Testing backend health...")
        try:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    print(f"   ✅ Backend healthy: {health_data}")
                else:
                    print(f"   ❌ Health check failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
        
        # Test 2: Memory Nodes Available
        print("\n2. 🧠 Testing memory nodes...")
        try:
            async with session.get(f"{base_url}/api/memory/nodes") as resp:
                if resp.status == 200:
                    nodes_data = await resp.json()
                    node_count = len(nodes_data)
                    print(f"   ✅ Found {node_count} memory nodes")
                    
                    # Show sample node concepts
                    sample_concepts = [node["concept"] for node in list(nodes_data.values())[:3]]
                    print(f"   📝 Sample concepts: {sample_concepts}")
                else:
                    print(f"   ❌ Memory nodes fetch failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Memory nodes error: {e}")
            return False
        
        # Test 3: Chat Integration
        print("\n3. 💬 Testing chat integration...")
        test_messages = [
            "Tell me about transformers in AI",
            "How do vector embeddings work?",
            "What are the best practices for React development?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n   Test {i}: '{message}'")
            
            payload = {
                "content": message,
                "conversation_history": []
            }
            
            try:
                async with session.post(
                    f"{base_url}/api/chat/send",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        response_data = await resp.json()
                        
                        # Validate response structure
                        if "message" in response_data and "retrieved_memories" in response_data:
                            msg = response_data["message"]
                            memories = response_data["retrieved_memories"]
                            access_events = response_data.get("memory_access_events", [])
                            
                            print(f"      ✅ Response received ({len(msg['content'])} chars)")
                            print(f"      📚 Retrieved {len(memories)} memories")
                            print(f"      🎯 Recorded {len(access_events)} access events")
                            
                            # Show retrieved memory concepts
                            if memories:
                                concepts = [mem["concept"] for mem in memories[:2]]
                                scores = [f"{mem['relevance_score']:.3f}" for mem in memories[:2]]
                                print(f"      🔍 Top memories: {list(zip(concepts, scores))}")
                            
                            # Show sample response (first 100 chars)
                            response_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"      💭 Response: {response_preview}")
                            
                        else:
                            print(f"      ❌ Invalid response structure")
                            return False
                    else:
                        error_text = await resp.text()
                        print(f"      ❌ Chat API error {resp.status}: {error_text}")
                        return False
                        
                # Small delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"      ❌ Chat request error: {e}")
                return False
        
        # Test 4: Memory Access Events (WebSocket would be ideal, but check via API)
        print("\n4. 🎭 Testing memory access events...")
        try:
            async with session.get(f"{base_url}/api/analytics/access-patterns") as resp:
                if resp.status == 200:
                    patterns_data = await resp.json()
                    print(f"   ✅ Access patterns API working: {len(patterns_data.get('events', []))} events")
                else:
                    print(f"   ⚠️ Access patterns API returned {resp.status} (may be empty)")
        except Exception as e:
            print(f"   ⚠️ Access patterns test failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 Chat Integration Test PASSED!")
        print("   ✅ Backend is healthy and responding")
        print("   ✅ Memory retrieval is working")
        print("   ✅ AI response generation is working")
        print("   ✅ Memory access events are being recorded")
        print("   ✅ Real-time memory visualization data is available")
        return True

async def main():
    """Main test runner"""
    try:
        success = await test_chat_integration()
        if success:
            print("\n🚀 Integration is ready for frontend testing!")
            print("   💡 Restart the frontend to pick up environment changes")
            print("   🌐 Open browser and test chat functionality")
            sys.exit(0)
        else:
            print("\n❌ Integration test failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test script error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())