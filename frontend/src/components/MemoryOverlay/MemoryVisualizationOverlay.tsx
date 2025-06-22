import React, { useEffect, useState } from 'react';
import { View, StyleSheet, Modal, Pressable, Text, Dimensions } from 'react-native';
import { BlurView } from 'expo-blur';
import { Feather } from '@expo/vector-icons';
import { MemoryGraphOverlay } from '../Memory3D/MemoryGraphOverlay';
import { useChatStore } from '../../stores/chatStore';
import { useMemoryStore } from '../../stores/memoryStore';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

interface MemoryVisualizationOverlayProps {
  visible: boolean;
  onClose: () => void;
}

export const MemoryVisualizationOverlay: React.FC<MemoryVisualizationOverlayProps> = ({
  visible,
  onClose,
}) => {
  const { isTyping } = useChatStore();
  const { isThinking, recentAccesses } = useMemoryStore();
  const [pulseAnimation, setPulseAnimation] = useState(false);

  // Pulsing animation for thinking indicator
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isThinking || isTyping) {
      interval = setInterval(() => {
        setPulseAnimation(prev => !prev);
      }, 800);
    } else {
      setPulseAnimation(false);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isThinking, isTyping]);

  // Get recent access count for display
  const recentAccessCount = recentAccesses.filter(
    access => Date.now() - access.timestamp < 10000 // Last 10 seconds
  ).length;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <BlurView intensity={20} style={styles.overlay}>
        <View style={styles.container}>
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <View style={styles.thinkingIndicator}>
                <View style={[
                  styles.thinkingDot, 
                  { 
                    opacity: (isTyping || isThinking) ? (pulseAnimation ? 1 : 0.5) : 0.3,
                    backgroundColor: (isTyping || isThinking) ? '#FFD700' : '#007AFF',
                    transform: [{ scale: (isTyping || isThinking) && pulseAnimation ? 1.2 : 1 }]
                  }
                ]} />
                <View style={styles.thinkingTextContainer}>
                  <Text style={styles.thinkingText}>
                    {(isTyping || isThinking) ? 'AI is thinking...' : 'Memory Access'}
                  </Text>
                  {recentAccessCount > 0 && (
                    <Text style={styles.accessCountText}>
                      {recentAccessCount} recent access{recentAccessCount !== 1 ? 'es' : ''}
                    </Text>
                  )}
                </View>
              </View>
            </View>
            
            <Pressable style={styles.closeButton} onPress={onClose}>
              <Feather name="x" size={24} color="#FFFFFF" />
            </Pressable>
          </View>

          {/* 3D Visualization */}
          <View style={styles.visualizationContainer}>
            <MemoryGraphOverlay
              width={screenWidth - 40}
              height={screenHeight * 0.7}
            />
          </View>

          {/* Info Panel */}
          <View style={styles.infoPanel}>
            <Text style={styles.infoText}>
              {(isTyping || isThinking)
                ? 'Watch as the AI accesses different memories to formulate a response. Nodes glow brighter when accessed.'
                : 'Memory access visualization - nodes light up when accessed. Send a message to see the AI think!'
              }
            </Text>
          </View>
        </View>
      </BlurView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  container: {
    width: screenWidth - 20,
    height: screenHeight * 0.85,
    backgroundColor: 'rgba(0, 0, 17, 0.95)',
    borderRadius: 20,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  headerLeft: {
    flex: 1,
  },
  thinkingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  thinkingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#007AFF',
    marginRight: 8,
  },
  thinkingTextContainer: {
    flex: 1,
  },
  thinkingText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
  },
  accessCountText: {
    color: '#FFFFFF',
    fontSize: 12,
    opacity: 0.7,
    marginTop: 2,
  },
  closeButton: {
    padding: 8,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  visualizationContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  infoPanel: {
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  infoText: {
    color: '#FFFFFF',
    fontSize: 14,
    textAlign: 'center',
    opacity: 0.8,
    lineHeight: 20,
  },
});