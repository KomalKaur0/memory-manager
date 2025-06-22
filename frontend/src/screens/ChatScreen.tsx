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
  } = useChatStore();

  const { selectNode, simulateMemoryAccess } = useMemoryStore();
  const { status: connectionStatus, sendMessage } = useBackendConnection();

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

    try {
      if (connectionStatus.isConnected) {
        // Use real backend
        const response = await sendMessage(userMessage, messages);
        
        // Add AI response with memory accesses
        addMessage({
          content: response.message.content,
          role: 'assistant',
          memory_accesses: response.memory_access_events || [],
        });
        
        // Update memory store with retrieved memories if any
        if (response.retrieved_memories?.length > 0) {
          // Process retrieved memories and update memory visualization
          console.log('Retrieved memories:', response.retrieved_memories);
        }
      } else {
        // Fallback to simulation
        await simulateThinkingWithMemoryAccess();
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      // Fallback to simulation on error
      await simulateThinkingWithMemoryAccess();
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