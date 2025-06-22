import React, { useState } from 'react';
import { View, Text, StyleSheet, Pressable, ScrollView, Alert, TextInput, Animated } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { useBackendConnection } from '../../hooks/useBackendConnection';
import { useMemoryStore } from '../../stores/memoryStore';
import { API_CONFIG } from '../../config/api';

interface ConnectionDebugProps {
  isVisible?: boolean;
  onToggle?: () => void;
}

export const ConnectionDebug: React.FC<ConnectionDebugProps> = ({ 
  isVisible = true, 
  onToggle 
}) => {
  const { status, testConnection, retryConnection, updateApiUrl } = useBackendConnection();
  const { generateMockData, generateDemoData, nodes } = useMemoryStore();
  const [testResults, setTestResults] = useState<any>(null);
  const [testing, setTesting] = useState(false);
  const [newUrl, setNewUrl] = useState('');

  const runFullTest = async () => {
    setTesting(true);
    try {
      const results = await testConnection();
      setTestResults(results);
      
      const allPassed = Object.values(results).every((r: any) => r.success);
      Alert.alert(
        'Connection Test Complete',
        allPassed ? 'All tests passed!' : 'Some tests failed - check logs'
      );
    } catch (error) {
      Alert.alert('Test Error', error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setTesting(false);
    }
  };

  const getStatusColor = () => {
    if (status.isConnecting) return '#FFA500';
    if (status.isConnected) return '#4CAF50';
    return '#F44336';
  };

  const updateUrl = () => {
    if (newUrl.trim()) {
      try {
        new URL(newUrl);
        updateApiUrl(newUrl.trim());
        setNewUrl('');
        Alert.alert('URL Updated', 'Testing connection with new URL...');
      } catch {
        Alert.alert('Invalid URL', 'Please enter a valid URL (include https://)');
      }
    }
  };

  const renderTestResult = (name: string, result: any) => {
    if (!result) return null;
    
    return (
      <View style={styles.testResult}>
        <Text style={[styles.testName, { color: result.success ? '#4CAF50' : '#F44336' }]}>
          {result.success ? '✅' : '❌'} {name}
        </Text>
        {result.duration && (
          <Text style={styles.testDetail}>Duration: {result.duration}ms</Text>
        )}
        {result.error && (
          <Text style={styles.errorText}>{result.error}</Text>
        )}
        {result.details && (
          <Text style={styles.detailText}>
            {JSON.stringify(result.details, null, 2).substring(0, 200)}...
          </Text>
        )}
      </View>
    );
  };

  if (!isVisible) {
    return null;
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🔧 Connection Debug</Text>
        <View style={styles.headerRight}>
          <View style={[styles.statusDot, { backgroundColor: getStatusColor() }]} />
          {onToggle && (
            <Pressable style={styles.closeButton} onPress={onToggle}>
              <Feather name="x" size={18} color="#666" />
            </Pressable>
          )}
        </View>
      </View>

      <View style={styles.statusSection}>
        <Text style={styles.sectionTitle}>Configuration</Text>
        <Text style={styles.detailText}>
          Current URL: {API_CONFIG.BASE_URL}
        </Text>
        <Text style={styles.detailText}>
          Fallbacks: {API_CONFIG.FALLBACK_URLS.length} URLs
        </Text>
        <Text style={styles.detailText}>
          Debug Mode: {API_CONFIG.DEBUG_MODE ? 'ON' : 'OFF'}
        </Text>
      </View>

      <View style={styles.statusSection}>
        <Text style={styles.sectionTitle}>Status</Text>
        <Text style={styles.statusText}>
          {status.isConnecting ? 'Connecting...' : 
           status.isConnected ? 'Connected' : 'Disconnected'}
        </Text>
        {status.error && (
          <Text style={styles.errorText}>{status.error}</Text>
        )}
        {status.lastConnected && (
          <Text style={styles.detailText}>
            Last connected: {status.lastConnected.toLocaleTimeString()}
          </Text>
        )}
      </View>

      <View style={styles.actionSection}>
        <Pressable
          style={[styles.button, styles.testButton]}
          onPress={runFullTest}
          disabled={testing}
        >
          <Text style={styles.buttonText}>
            {testing ? 'Testing...' : 'Run Full Test'}
          </Text>
        </Pressable>

        <Pressable
          style={[styles.button, styles.retryButton]}
          onPress={retryConnection}
          disabled={status.isConnecting}
        >
          <Text style={styles.buttonText}>Retry Connection</Text>
        </Pressable>
      </View>

      <View style={styles.demoSection}>
        <Text style={styles.sectionTitle}>Demo Data</Text>
        <Text style={styles.detailText}>
          Current nodes: {Object.keys(nodes).length}
        </Text>
        <View style={styles.actionSection}>
          <Pressable
            style={[styles.button, styles.demoButton]}
            onPress={() => {
              generateDemoData();
              Alert.alert('Demo Data Loaded', '12 realistic memories with semantic clusters loaded for demonstration');
            }}
          >
            <Text style={styles.buttonText}>Load Demo</Text>
          </Pressable>
          
          <Pressable
            style={[styles.button, styles.techButton]}
            onPress={() => {
              generateMockData();
              Alert.alert('Tech Data Loaded', '7 technical memories loaded for system testing');
            }}
          >
            <Text style={styles.buttonText}>Load Tech</Text>
          </Pressable>
        </View>
      </View>

      <View style={styles.urlSection}>
        <Text style={styles.sectionTitle}>Update URL</Text>
        <TextInput
          style={styles.urlInput}
          value={newUrl}
          onChangeText={setNewUrl}
          placeholder="Enter backend URL (e.g., https://abc123.ngrok.io)"
          placeholderTextColor="#999"
          autoCapitalize="none"
          autoCorrect={false}
        />
        <Pressable
          style={[styles.button, styles.updateButton]}
          onPress={updateUrl}
          disabled={!newUrl.trim()}
        >
          <Text style={styles.buttonText}>Update & Test</Text>
        </Pressable>
      </View>

      {testResults && (
        <ScrollView style={styles.resultsSection}>
          <Text style={styles.sectionTitle}>Test Results</Text>
          {renderTestResult('Health Check', testResults.health)}
          {renderTestResult('Memory Nodes', testResults.memoryNodes)}
          {renderTestResult('Chat API', testResults.chat)}
        </ScrollView>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 16,
    marginTop: 0,
    borderWidth: 1,
    borderColor: '#E5E5EA',
    maxHeight: '80%',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 5,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  closeButton: {
    padding: 4,
    borderRadius: 4,
    backgroundColor: '#F2F2F7',
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statusSection: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    marginBottom: 8,
  },
  statusText: {
    fontSize: 14,
    color: '#333',
    marginBottom: 4,
  },
  errorText: {
    fontSize: 12,
    color: '#F44336',
    marginBottom: 4,
  },
  detailText: {
    fontSize: 12,
    color: '#666',
    fontFamily: 'monospace',
  },
  actionSection: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  button: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  testButton: {
    backgroundColor: '#007AFF',
  },
  retryButton: {
    backgroundColor: '#34C759',
  },
  updateButton: {
    backgroundColor: '#FF9500',
  },
  demoButton: {
    backgroundColor: '#8E44AD',
  },
  techButton: {
    backgroundColor: '#2E86AB',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '500',
  },
  demoSection: {
    marginBottom: 16,
  },
  resultsSection: {
    maxHeight: 200,
  },
  testResult: {
    backgroundColor: '#FFFFFF',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#E5E5EA',
  },
  testName: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  testDetail: {
    fontSize: 12,
    color: '#666',
    marginBottom: 2,
  },
  urlSection: {
    marginBottom: 16,
  },
  urlInput: {
    borderWidth: 1,
    borderColor: '#E5E5EA',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    marginBottom: 8,
    backgroundColor: '#FFFFFF',
  },
});