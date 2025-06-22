import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/apiService';
import { useMemoryStore } from '../stores/memoryStore';
import { useChatStore } from '../stores/chatStore';
import { connectionTester } from '../utils/connectionTest';

interface ConnectionStatus {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastConnected: Date | null;
}

export const useBackendConnection = () => {
  const [status, setStatus] = useState<ConnectionStatus>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastConnected: null,
  });

  const { nodes, addNode, addMemoryAccess, generateMockData } = useMemoryStore();
  const { addMessage } = useChatStore();

  // Check actual backend connection
  const checkConnection = useCallback(async () => {
    console.log('🔄 Starting enhanced backend connection check...');
    setStatus(prev => ({ ...prev, isConnecting: true, error: null }));
    
    try {
      // First, test the connection with detailed diagnostics
      console.log('🔧 Running connection diagnostics...');
      const connectionResult = await connectionTester.testConnection();
      
      if (connectionResult.success) {
        console.log('✅ Connection test passed, proceeding with health check...');
        
        // Double-check with our API service
        const isHealthy = await apiService.healthCheck();
        
        if (isHealthy) {
          console.log('✅ Backend is healthy, loading initial data...');
          setStatus({
            isConnected: true,
            isConnecting: false,
            error: null,
            lastConnected: new Date(),
          });
          
          // Load initial data from backend
          try {
            await loadInitialData();
            console.log('🎉 Successfully connected to backend and loaded data');
          } catch (dataError) {
            console.warn('⚠️ Connected but failed to load data:', dataError);
            // Still consider connected since health check passed
          }
        } else {
          throw new Error('API service health check failed after successful connection test');
        }
      } else {
        // Connection test failed, try to find working connection
        console.log('❌ Initial connection failed, searching for alternatives...');
        const workingUrl = await connectionTester.findWorkingConnection();
        
        if (workingUrl) {
          console.log(`🔧 Found working URL: ${workingUrl}, updating API service...`);
          apiService.updateBaseUrl(workingUrl);
          
          // Retry with new URL
          const retryResult = await apiService.healthCheck();
          if (retryResult) {
            setStatus({
              isConnected: true,
              isConnecting: false,
              error: null,
              lastConnected: new Date(),
            });
            await loadInitialData();
            console.log('🎉 Successfully connected with alternative URL');
            return;
          }
        }
        
        throw new Error(connectionResult.error || 'All connection attempts failed');
      }
    } catch (error) {
      console.warn('❌ Backend connection failed, using mock data:', error);
      setStatus({
        isConnected: false,
        isConnecting: false,
        error: error instanceof Error ? error.message : 'Connection failed',
        lastConnected: null,
      });
      
      // Fallback to mock data
      if (Object.keys(nodes).length === 0) {
        console.log('📦 Loading mock data as fallback');
        generateMockData();
      }
    }
  }, [nodes, generateMockData, loadInitialData]);

  // Load initial memory data from backend
  const loadInitialData = useCallback(async () => {
    try {
      console.log('📥 Loading initial memory data from backend...');
      const response = await apiService.getMemoryNodes();
      console.log('📊 Memory nodes response:', response);
      
      if (response.success && response.data) {
        const nodeCount = Object.keys(response.data).length;
        console.log(`📦 Adding ${nodeCount} nodes to store`);
        
        // Add nodes to store
        Object.values(response.data).forEach(node => {
          addNode(node);
        });
        
        console.log('✅ Initial data loaded successfully');
      } else {
        console.warn('⚠️ No data received or request failed:', response);
      }
    } catch (error) {
      console.error('❌ Failed to load initial data:', error);
      throw error; // Re-throw to handle in checkConnection
    }
  }, [addNode]);

  // Simplified - no WebSocket for now
  const setupWebSocket = useCallback(() => {
    // WebSocket setup removed for simplicity
    return () => {};
  }, []);

  // Update API base URL (useful for switching ngrok tunnels)
  const updateApiUrl = useCallback((newUrl: string) => {
    apiService.updateBaseUrl(newUrl);
    checkConnection();
  }, [checkConnection]);

  // Retry connection
  const retryConnection = useCallback(() => {
    checkConnection();
  }, [checkConnection]);

  // Send message to backend
  const sendMessage = useCallback(async (content: string, conversationHistory: any[] = []) => {
    if (!status.isConnected) {
      throw new Error('Not connected to backend');
    }

    try {
      const response = await apiService.sendMessage(content, conversationHistory);
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.message || 'Failed to send message');
      }
    } catch (error) {
      throw error;
    }
  }, [status.isConnected]);

  // Stream chat response
  const streamChatResponse = useCallback(async function* (content: string) {
    if (!status.isConnected) {
      throw new Error('Not connected to backend');
    }

    yield* apiService.streamChatResponse(content);
  }, [status.isConnected]);

  // Initial connection check
  useEffect(() => {
    checkConnection();
    
    // Set up periodic health checks
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  // Set up WebSocket when connected
  useEffect(() => {
    let cleanup: (() => void) | undefined;
    
    if (status.isConnected) {
      cleanup = setupWebSocket();
    }
    
    return cleanup;
  }, [setupWebSocket]);

  // Test connection with detailed diagnostics
  const testConnection = useCallback(async () => {
    console.log('🧪 Running full connection test...');
    const results = await connectionTester.testFullAPI();
    console.log('📊 Test results:', results);
    return results;
  }, []);

  return {
    status,
    checkConnection,
    retryConnection,
    updateApiUrl,
    sendMessage,
    streamChatResponse,
    testConnection,
  };
};