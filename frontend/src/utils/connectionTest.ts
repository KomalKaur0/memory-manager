/**
 * Connection test utilities for debugging frontend-backend integration
 */

export interface ConnectionTestResult {
  success: boolean;
  error?: string;
  details?: any;
  duration?: number;
}

export class ConnectionTester {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async testConnection(): Promise<ConnectionTestResult> {
    const startTime = Date.now();
    
    try {
      console.log(`🔧 Testing connection to ${this.baseUrl}`);
      
      // Test basic connectivity
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      });
      
      clearTimeout(timeoutId);
      const duration = Date.now() - startTime;
      
      if (response.ok) {
        const data = await response.json();
        console.log(`✅ Connection successful in ${duration}ms:`, data);
        
        return {
          success: true,
          details: data,
          duration,
        };
      } else {
        const error = `HTTP ${response.status}: ${response.statusText}`;
        console.error(`❌ Connection failed: ${error}`);
        
        return {
          success: false,
          error,
          duration,
        };
      }
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      console.error(`❌ Connection error after ${duration}ms:`, error);
      
      // Provide specific error guidance
      let guidance = '';
      if (errorMessage.includes('Failed to fetch')) {
        guidance = 'Backend may not be running or CORS issue. Check if backend is on localhost:8000';
      } else if (errorMessage.includes('AbortError')) {
        guidance = 'Connection timed out. Backend may be slow or unreachable';
      } else if (errorMessage.includes('NetworkError')) {
        guidance = 'Network error. Check if backend is accessible';
      }
      
      return {
        success: false,
        error: `${errorMessage}${guidance ? ` - ${guidance}` : ''}`,
        duration,
      };
    }
  }

  async testFullAPI(): Promise<{
    health: ConnectionTestResult;
    memoryNodes: ConnectionTestResult;
    chat: ConnectionTestResult;
  }> {
    console.log('🧪 Running full API test suite...');
    
    const results = {
      health: await this.testConnection(),
      memoryNodes: await this.testMemoryNodes(),
      chat: await this.testChat(),
    };
    
    const allPassed = Object.values(results).every(r => r.success);
    console.log(`📊 API test results: ${allPassed ? '✅ ALL PASSED' : '❌ SOME FAILED'}`);
    
    return results;
  }

  private async testMemoryNodes(): Promise<ConnectionTestResult> {
    const startTime = Date.now();
    
    try {
      console.log('📦 Testing memory nodes endpoint...');
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const response = await fetch(`${this.baseUrl}/api/memory/nodes`, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      });
      
      clearTimeout(timeoutId);
      const duration = Date.now() - startTime;
      
      if (response.ok) {
        const data = await response.json();
        const nodeCount = Object.keys(data).length;
        console.log(`✅ Memory nodes: ${nodeCount} nodes found`);
        
        return {
          success: true,
          details: { nodeCount, sample: Object.keys(data).slice(0, 3) },
          duration,
        };
      } else {
        const error = `HTTP ${response.status}: ${response.statusText}`;
        console.error(`❌ Memory nodes failed: ${error}`);
        
        return {
          success: false,
          error,
          duration,
        };
      }
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`❌ Memory nodes error:`, error);
      
      return {
        success: false,
        error: errorMessage,
        duration,
      };
    }
  }

  private async testChat(): Promise<ConnectionTestResult> {
    const startTime = Date.now();
    
    try {
      console.log('💬 Testing chat endpoint...');
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);
      
      const response = await fetch(`${this.baseUrl}/api/chat/send`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: 'Connection test message',
          conversation_history: [],
        }),
      });
      
      clearTimeout(timeoutId);
      const duration = Date.now() - startTime;
      
      if (response.ok) {
        const data = await response.json();
        console.log(`✅ Chat: Response received (${data.retrieved_memories.length} memories)`);
        
        return {
          success: true,
          details: {
            responseLength: data.message.content.length,
            memoryCount: data.retrieved_memories.length,
            processingTime: data.processing_time,
          },
          duration,
        };
      } else {
        const error = `HTTP ${response.status}: ${response.statusText}`;
        console.error(`❌ Chat failed: ${error}`);
        
        return {
          success: false,
          error,
          duration,
        };
      }
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`❌ Chat error:`, error);
      
      return {
        success: false,
        error: errorMessage,
        duration,
      };
    }
  }

  // Test different base URLs to find working connection
  async findWorkingConnection(): Promise<string | null> {
    // Import here to avoid circular dependency
    const { API_CONFIG } = await import('../config/api');
    
    const urlsToTest = [
      ...API_CONFIG.FALLBACK_URLS,
      'http://0.0.0.0:8000',
    ];
    
    console.log('🔍 Searching for working backend connection...');
    
    for (const url of urlsToTest) {
      console.log(`Testing: ${url}`);
      const tester = new ConnectionTester(url);
      const result = await tester.testConnection();
      
      if (result.success) {
        console.log(`✅ Found working connection: ${url}`);
        return url;
      }
    }
    
    console.log('❌ No working connection found');
    return null;
  }
}

// Export singleton instance
export const connectionTester = new ConnectionTester();