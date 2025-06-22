import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/apiService';
import { useMemoryStore } from '../stores/memoryStore';
import { useChatStore } from '../stores/chatStore';

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

  // Simplified - always use mock data for now
  const checkConnection = useCallback(async () => {
    setStatus({
      isConnected: false,
      isConnecting: false,
      error: 'Using mock data for local development',
      lastConnected: null,
    });
    
    // Always use mock data
    if (Object.keys(nodes).length === 0) {
      console.log('Loading mock data for development');
      generateMockData();
    }
  }, [nodes, generateMockData]);

  // Load initial memory data from backend
  const loadInitialData = useCallback(async () => {
    try {
      const response = await apiService.getMemoryNodes();
      if (response.success && response.data) {
        // Add nodes to store
        Object.values(response.data).forEach(node => {
          addNode(node);
        });
      }
    } catch (error) {
      console.error('Failed to load initial data:', error);
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
  const sendMessage = useCallback(async (content: string) => {
    if (!status.isConnected) {
      throw new Error('Not connected to backend');
    }

    try {
      const response = await apiService.sendMessage(content);
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

  return {
    status,
    checkConnection,
    retryConnection,
    updateApiUrl,
    sendMessage,
    streamChatResponse,
  };
};