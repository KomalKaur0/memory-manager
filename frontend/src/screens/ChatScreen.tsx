import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, FlatList, SafeAreaView, StatusBar } from 'react-native';
import { ChatBubble } from '../components/Chat/ChatBubble';
import { ChatInput } from '../components/Chat/ChatInput';
import { TypingIndicator } from '../components/Chat/TypingIndicator';
import { MemoryVisualizationOverlay } from '../components/MemoryOverlay/MemoryVisualizationOverlay';
import { ConnectionStatus } from '../components/Connection/ConnectionStatus';
import { useChatStore } from '../stores/chatStore';
import { useMemoryStore } from '../stores/memoryStore';
import { useBackendConnection } from '../hooks/useBackendConnection';

export const ChatScreen: React.FC = () => {
  const flatListRef = useRef<FlatList>(null);
  
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
  const { status: connectionStatus } = useBackendConnection();

  useEffect(() => {
    // Auto-scroll to bottom when new message is added
    if (messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages.length, isTyping]);

  const handleSend = async () => {
    if (!currentInput.trim()) return;
    
    const userMessage = currentInput.trim();
    setCurrentInput('');
    
    // Add user message
    addMessage({
      content: userMessage,
      role: 'user',
    });

    // Simulate AI response with memory access
    await simulateThinkingWithMemoryAccess();
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
});