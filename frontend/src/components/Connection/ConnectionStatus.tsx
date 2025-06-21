import React, { useState } from 'react';
import { View, Text, StyleSheet, Pressable, Modal, TextInput, Alert } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { useBackendConnection } from '../../hooks/useBackendConnection';

interface ConnectionStatusProps {
  showDetails?: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ showDetails = false }) => {
  const { status, retryConnection, updateApiUrl } = useBackendConnection();
  const [showModal, setShowModal] = useState(false);
  const [newUrl, setNewUrl] = useState('');

  const getStatusColor = () => {
    if (status.isConnecting) return '#FFA500';
    if (status.isConnected) return '#4CAF50';
    return '#F44336';
  };

  const getStatusText = () => {
    if (status.isConnecting) return 'Connecting...';
    if (status.isConnected) return 'Connected';
    return 'Disconnected';
  };

  const getStatusIcon = () => {
    if (status.isConnecting) return 'loader';
    if (status.isConnected) return 'wifi';
    return 'wifi-off';
  };

  const handleUpdateUrl = () => {
    if (newUrl.trim()) {
      // Validate URL format
      try {
        new URL(newUrl);
        updateApiUrl(newUrl.trim());
        setShowModal(false);
        setNewUrl('');
      } catch {
        Alert.alert('Invalid URL', 'Please enter a valid URL (include https://)');
      }
    }
  };

  const StatusIndicator = () => (
    <View style={styles.statusContainer}>
      <View style={[styles.statusDot, { backgroundColor: getStatusColor() }]} />
      {showDetails && (
        <Text style={[styles.statusText, { color: getStatusColor() }]}>
          {getStatusText()}
        </Text>
      )}
    </View>
  );

  if (!showDetails) {
    return <StatusIndicator />;
  }

  return (
    <>
      <Pressable 
        style={styles.connectionCard}
        onPress={() => setShowModal(true)}
      >
        <View style={styles.cardHeader}>
          <Feather name={getStatusIcon()} size={20} color={getStatusColor()} />
          <Text style={styles.cardTitle}>Backend Connection</Text>
        </View>
        
        <Text style={[styles.statusText, { color: getStatusColor() }]}>
          {getStatusText()}
        </Text>
        
        {status.error && (
          <Text style={styles.errorText} numberOfLines={2}>
            {status.error}
          </Text>
        )}
        
        {status.lastConnected && (
          <Text style={styles.lastConnectedText}>
            Last connected: {status.lastConnected.toLocaleTimeString()}
          </Text>
        )}
        
        <View style={styles.cardActions}>
          <Pressable
            style={[styles.actionButton, styles.retryButton]}
            onPress={retryConnection}
            disabled={status.isConnecting}
          >
            <Feather name="refresh-cw" size={16} color="#FFFFFF" />
            <Text style={styles.actionButtonText}>Retry</Text>
          </Pressable>
          
          <Pressable
            style={[styles.actionButton, styles.configButton]}
            onPress={() => setShowModal(true)}
          >
            <Feather name="settings" size={16} color="#007AFF" />
            <Text style={[styles.actionButtonText, styles.configButtonText]}>Config</Text>
          </Pressable>
        </View>
      </Pressable>

      {/* Configuration Modal */}
      <Modal
        visible={showModal}
        transparent
        animationType="slide"
        onRequestClose={() => setShowModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Backend Configuration</Text>
              <Pressable onPress={() => setShowModal(false)}>
                <Feather name="x" size={24} color="#666" />
              </Pressable>
            </View>
            
            <View style={styles.modalBody}>
              <Text style={styles.instructionTitle}>Setup Instructions:</Text>
              <Text style={styles.instructionText}>
                1. Start your backend: python -m uvicorn src.api.main:app --port 8000{'\n'}
                2. Create ngrok tunnel: ngrok http 8000{'\n'}
                3. Copy the ngrok URL and paste below
              </Text>
              
              <Text style={styles.inputLabel}>Backend URL:</Text>
              <TextInput
                style={styles.urlInput}
                value={newUrl}
                onChangeText={setNewUrl}
                placeholder="https://abc123.ngrok.io"
                placeholderTextColor="#999"
                autoCapitalize="none"
                autoCorrect={false}
              />
              
              <View style={styles.modalActions}>
                <Pressable
                  style={[styles.modalButton, styles.cancelButton]}
                  onPress={() => {
                    setShowModal(false);
                    setNewUrl('');
                  }}
                >
                  <Text style={styles.cancelButtonText}>Cancel</Text>
                </Pressable>
                
                <Pressable
                  style={[styles.modalButton, styles.saveButton]}
                  onPress={handleUpdateUrl}
                >
                  <Text style={styles.saveButtonText}>Update & Connect</Text>
                </Pressable>
              </View>
            </View>
          </View>
        </View>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 14,
    fontWeight: '500',
  },
  connectionCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 16,
    marginVertical: 8,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
  },
  errorText: {
    fontSize: 12,
    color: '#F44336',
    marginTop: 4,
  },
  lastConnectedText: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  cardActions: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 12,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    flex: 1,
    justifyContent: 'center',
  },
  retryButton: {
    backgroundColor: '#007AFF',
  },
  configButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  actionButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#FFFFFF',
  },
  configButtonText: {
    color: '#007AFF',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    width: '90%',
    maxWidth: 400,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
  },
  modalBody: {
    padding: 20,
  },
  instructionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    marginBottom: 8,
  },
  instructionText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#000',
    marginBottom: 8,
  },
  urlInput: {
    borderWidth: 1,
    borderColor: '#E5E5EA',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
    marginBottom: 20,
  },
  modalActions: {
    flexDirection: 'row',
    gap: 12,
  },
  modalButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: '#F2F2F7',
  },
  saveButton: {
    backgroundColor: '#007AFF',
  },
  cancelButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#666',
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#FFFFFF',
  },
});