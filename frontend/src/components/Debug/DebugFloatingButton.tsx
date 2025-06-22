import React from 'react';
import { View, StyleSheet, Pressable, Text } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { useBackendConnection } from '../../hooks/useBackendConnection';

interface DebugFloatingButtonProps {
  onPress: () => void;
  isDebugVisible: boolean;
}

export const DebugFloatingButton: React.FC<DebugFloatingButtonProps> = ({
  onPress,
  isDebugVisible,
}) => {
  const { status } = useBackendConnection();

  const getStatusColor = () => {
    if (status.isConnecting) return '#FFA500';
    if (status.isConnected) return '#4CAF50';
    return '#F44336';
  };

  const getStatusIcon = () => {
    if (status.isConnecting) return 'loader';
    if (status.isConnected) return 'wifi';
    return 'wifi-off';
  };

  return (
    <View style={styles.container}>
      <Pressable
        style={[styles.button, { borderColor: getStatusColor() }]}
        onPress={onPress}
      >
        <View style={styles.buttonContent}>
          <Feather 
            name={isDebugVisible ? 'x' : getStatusIcon()} 
            size={14} 
            color={getStatusColor()} 
          />
          <View style={[styles.statusDot, { backgroundColor: getStatusColor() }]} />
        </View>
        
        {!status.isConnected && !isDebugVisible && (
          <View style={styles.badge}>
            <Text style={styles.badgeText}>!</Text>
          </View>
        )}
      </Pressable>
      
      {!isDebugVisible && (
        <Text style={styles.hint}>Debug</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 60,
    left: 20,
    alignItems: 'center',
    zIndex: 1000,
  },
  button: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#FFFFFF',
    borderWidth: 2,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  buttonContent: {
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    position: 'absolute',
    top: -8,
    right: -8,
  },
  badge: {
    position: 'absolute',
    top: -4,
    right: -4,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#F44336',
    justifyContent: 'center',
    alignItems: 'center',
  },
  badgeText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '600',
  },
  hint: {
    marginTop: 4,
    fontSize: 10,
    color: '#666',
    fontWeight: '500',
  },
});