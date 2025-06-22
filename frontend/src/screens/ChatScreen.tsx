import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, FlatList, SafeAreaView, StatusBar, Platform } from 'react-native';
import { ChatBubble } from '../components/Chat/ChatBubble';
import { ChatInput } from '../components/Chat/ChatInput';
import { TypingIndicator } from '../components/Chat/TypingIndicator';
import { MemoryVisualizationOverlay } from '../components/MemoryOverlay/MemoryVisualizationOverlay';
import { ConnectionStatus } from '../components/Connection/ConnectionStatus';
import { ConnectionDebug } from '../components/Debug/ConnectionDebug';
import { DebugFloatingButton } from '../components/Debug/DebugFloatingButton';
import { useChatStore } from '../stores/chatStore';
import { useMemoryStore } from '../stores/memoryStore';
import { useBackendConnection } from '../hooks/useBackendConnection';

export const ChatScreen: React.FC = () => {
  const flatListRef = useRef<FlatList>(null);
  const [debugVisible, setDebugVisible] = React.useState(false);
  
  const {
    messages,
    isTyping,
    currentInput,
    memoryVisualizationVisible,
    setCurrentInput,
    setMemoryVisualizationVisible,
    addMessage,
    simulateThinkingWithMemoryAccess,
    setTyping,
  } = useChatStore();

  const { 
    selectNode, 
    simulateMemoryAccess, 
    loadMemoryNodes, 
    generateMockData, 
    nodes, 
    isUsingBackend,
    initialized 
  } = useMemoryStore();
  const { status: connectionStatus, sendMessage } = useBackendConnection();

  // Initialize memory data on component mount
  useEffect(() => {
    const initializeMemoryData = async () => {
      const nodeCount = Object.keys(nodes).length;
      console.log(`🔄 Chat: Initialized: ${initialized}, Node count: ${nodeCount}`);
      
      // Only initialize if not already initialized
      if (!initialized) {
        console.log('🔄 Chat: Initializing memory data...');
        
        // Try to load from backend first
        const backendSuccess = await loadMemoryNodes();
        
        if (!backendSuccess) {
          console.log('📦 Chat: Backend unavailable, using mock data');
          generateMockData();
        }
      }
    };
    
    initializeMemoryData();
  }, []); // Remove dependencies to prevent infinite loop

  useEffect(() => {
    // Auto-scroll to bottom when new message is added
    if (messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages.length, isTyping]);

  // Debug panel keyboard shortcut (for web only)
  useEffect(() => {
    // Only run on web platform and if debug mode is enabled
    if (Platform.OS !== 'web' || process.env.EXPO_PUBLIC_DEBUG_MODE !== 'true') {
      return;
    }

    // Double-check window exists and has addEventListener
    if (typeof window === 'undefined' || typeof window.addEventListener !== 'function') {
      return;
    }

    const handleKeyPress = (event: KeyboardEvent) => {
      // Ctrl+D or Cmd+D to toggle debug panel
      if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
        event.preventDefault();
        setDebugVisible(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => {
      if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
        window.removeEventListener('keydown', handleKeyPress);
      }
    };
  }, []);

  const handleSend = async () => {
    if (!currentInput.trim()) return;
    
    const userMessage = currentInput.trim();
    setCurrentInput('');
    
    // Add user message
    addMessage({
      content: userMessage,
      role: 'user',
    });

    console.log(`🔄 Chat: Sending message "${userMessage.slice(0, 50)}..."`);
    console.log(`🔌 Connection status: Connected=${connectionStatus.isConnected}, Error=${connectionStatus.error}`);

    // Show thinking indicator
    setTyping(true);
    
    let backendFailed = false;
    let backendError: string | null = null;

    try {
      // First try: Real backend
      if (connectionStatus.isConnected) {
        console.log('🤖 Attempting real backend for AI response...');
        
        try {
          const response = await sendMessage(userMessage, messages);
          console.log('✅ Backend response received:', response);
          
          setTyping(false);
          
          // Add AI response with memory accesses
          addMessage({
            content: response.message.content,
            role: 'assistant',
            memory_accesses: response.memory_access_events || [],
          });
          
          // Update memory store with retrieved memories if any
          if (response.retrieved_memories?.length > 0) {
            console.log('📚 Retrieved memories:', response.retrieved_memories.length);
            // Process retrieved memories and update memory visualization
            response.retrieved_memories.forEach((memory: any) => {
              console.log(`  - ${memory.concept}: ${memory.relevance_score}`);
            });
          }
          
          // Trigger memory visualization if there were memory accesses
          if (response.memory_access_events?.length > 0) {
            setMemoryVisualizationVisible(true);
            setTimeout(() => setMemoryVisualizationVisible(false), 3000);
          }
          
          // Success! Return early
          return;
        } catch (backendErr) {
          console.log('⚠️ Backend failed, will try fallback...');
          backendFailed = true;
          backendError = backendErr instanceof Error ? backendErr.message : 'Backend communication failed';
        }
      } else {
        console.log('🔌 Backend not connected, using fallback...');
        backendFailed = true;
        backendError = connectionStatus.error || 'Backend not connected';
      }
      
      // Second try: Simulation fallback
      if (backendFailed) {
        console.log('📦 Using simulation fallback...');
        setTyping(false); // Stop thinking indicator before simulation starts its own
        await simulateThinkingWithMemoryAccess();
        return;
      }
      
    } catch (simulationError) {
      // All fallbacks failed - now we show an error
      console.error('❌ All fallbacks failed:', simulationError);
      setTyping(false);
      
      addMessage({
        content: `I'm having trouble responding right now. Backend error: ${backendError || 'Unknown'}. Simulation error: ${simulationError instanceof Error ? simulationError.message : 'Unknown'}. Please try again.`,
        role: 'assistant',
      });
    }
  };

  const handleMemoryAccessPress = (nodeId: string) => {
    selectNode(nodeId);
    simulateMemoryAccess(nodeId, 'read');
    // Here you would typically navigate to the memory visualization
    // or show a modal with memory details
  };

  const renderMessage = ({ item }: { item: typeof messages[0] }) => (
    <ChatBubble
      message={item}
      onMemoryAccessPress={handleMemoryAccessPress}
    />
  );

  const renderTypingIndicator = () => (
    <TypingIndicator visible={isTyping} />
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
      
      {/* Connection Status */}
      {!connectionStatus.isConnected && (
        <ConnectionStatus showDetails={false} />
      )}
      
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(item) => item.id}
        style={styles.messagesList}
        contentContainerStyle={styles.messagesContent}
        showsVerticalScrollIndicator={false}
        ListFooterComponent={renderTypingIndicator}
      />
      
      <ChatInput
        value={currentInput}
        onChangeText={setCurrentInput}
        onSend={handleSend}
        disabled={isTyping}
      />

      {/* Memory Visualization Overlay */}
      <MemoryVisualizationOverlay
        visible={memoryVisualizationVisible}
        onClose={() => setMemoryVisualizationVisible(false)}
      />

      {/* Debug Panel */}
      {debugVisible && (
        <View style={styles.debugOverlay}>
          <ConnectionDebug
            isVisible={debugVisible}
            onToggle={() => setDebugVisible(false)}
          />
        </View>
      )}

      {/* Floating Debug Button - Only show in debug mode */}
      {process.env.EXPO_PUBLIC_DEBUG_MODE === 'true' && (
        <DebugFloatingButton
          onPress={() => setDebugVisible(!debugVisible)}
          isDebugVisible={debugVisible}
        />
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  messagesList: {
    flex: 1,
  },
  messagesContent: {
    paddingTop: 16,
    paddingBottom: 8,
  },
  debugOverlay: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    bottom: 100,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    zIndex: 999,
    justifyContent: 'flex-start',
    padding: 16,
  },
});