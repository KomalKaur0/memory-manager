import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { ChatMessage } from '../../types/memory';

interface ChatBubbleProps {
  message: ChatMessage;
  onMemoryAccessPress?: (nodeId: string) => void;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({ message, onMemoryAccessPress }) => {
  const isUser = message.role === 'user';
  const hasMemoryAccess = message.memory_accesses && message.memory_accesses.length > 0;

  return (
    <View style={[styles.container, isUser ? styles.userContainer : styles.assistantContainer]}>
      <View style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble]}>
        <Text style={[styles.text, isUser ? styles.userText : styles.assistantText]}>
          {message.content}
        </Text>
        
        {hasMemoryAccess && (
          <View style={styles.memoryAccessContainer}>
            <Text style={styles.memoryAccessLabel}>Memory accessed:</Text>
            <View style={styles.memoryNodes}>
              {message.memory_accesses!.map((access, index) => (
                <Pressable
                  key={`${access.node_id}-${index}`}
                  style={[
                    styles.memoryNode,
                    { 
                      backgroundColor: getAccessTypeColor(access.access_type),
                      shadowColor: getAccessTypeColor(access.access_type),
                      shadowOpacity: 0.6,
                      shadowRadius: 6,
                      shadowOffset: { width: 0, height: 0 },
                    }
                  ]}
                  onPress={() => onMemoryAccessPress?.(access.node_id)}
                >
                  <Text style={styles.memoryNodeText}>
                    {access.node_id}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        )}
      </View>
      
      <Text style={styles.timestamp}>
        {new Date(message.timestamp).toLocaleTimeString([], { 
          hour: '2-digit', 
          minute: '2-digit' 
        })}
      </Text>
    </View>
  );
};

const getAccessTypeColor = (accessType: string): string => {
  switch (accessType) {
    case 'read': return '#87CEEB';     // Sky blue glow - matching node colors
    case 'write': return '#98FB98';    // Pale green glow
    case 'strengthen': return '#FFD700'; // Gold glow
    case 'traverse': return '#DDA0DD';   // Plum glow
    default: return '#E6E6FA';           // Lavender
  }
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
    marginHorizontal: 16,
  },
  userContainer: {
    alignItems: 'flex-end',
  },
  assistantContainer: {
    alignItems: 'flex-start',
  },
  bubble: {
    maxWidth: '80%',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  userBubble: {
    backgroundColor: '#007AFF',
    borderBottomRightRadius: 6,
  },
  assistantBubble: {
    backgroundColor: '#F2F2F2',
    borderBottomLeftRadius: 6,
  },
  text: {
    fontSize: 16,
    lineHeight: 22,
  },
  userText: {
    color: '#FFFFFF',
  },
  assistantText: {
    color: '#000000',
  },
  timestamp: {
    fontSize: 12,
    color: '#8E8E93',
    marginTop: 4,
    marginHorizontal: 8,
  },
  memoryAccessContainer: {
    marginTop: 12,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.1)',
  },
  memoryAccessLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 6,
  },
  memoryNodes: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
  },
  memoryNode: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    marginRight: 4,
    marginBottom: 4,
  },
  memoryNodeText: {
    fontSize: 11,
    color: '#000011',
    fontWeight: '600',
  },
});