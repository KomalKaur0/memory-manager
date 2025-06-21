import React from 'react';
import { View, StyleSheet, Modal, Pressable, Text, Dimensions } from 'react-native';
import { BlurView } from 'expo-blur';
import { Feather } from '@expo/vector-icons';
import { MemoryGraph3D } from '../Memory3D/MemoryGraph3D';
import { useChatStore } from '../../stores/chatStore';

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
                <View style={[styles.thinkingDot, { opacity: isTyping ? 1 : 0.3 }]} />
                <Text style={styles.thinkingText}>
                  {isTyping ? 'AI is thinking...' : 'Memory Access'}
                </Text>
              </View>
            </View>
            
            <Pressable style={styles.closeButton} onPress={onClose}>
              <Feather name="x" size={24} color="#FFFFFF" />
            </Pressable>
          </View>

          {/* 3D Visualization */}
          <View style={styles.visualizationContainer}>
            <MemoryGraph3D
              width={screenWidth - 40}
              height={screenHeight * 0.7}
            />
          </View>

          {/* Info Panel */}
          <View style={styles.infoPanel}>
            <Text style={styles.infoText}>
              {isTyping 
                ? 'Watch as the AI accesses different memories to formulate a response'
                : 'Memory access visualization - nodes light up when accessed'
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
  thinkingText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
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