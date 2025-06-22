import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Text, ScrollView, Pressable, SafeAreaView, Dimensions } from 'react-native';
import { PanGestureHandler, State } from 'react-native-gesture-handler';
import { MemoryGraph3D } from '../components/Memory3D/MemoryGraph3D';
import { useMemoryStore } from '../stores/memoryStore';
import { Feather } from '@expo/vector-icons';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

export const MemoryVisualizationScreen: React.FC = () => {
  const [detailsEnabled, setDetailsEnabled] = useState(true); // Controls if clicking nodes opens panels
  const [panelVisible, setPanelVisible] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [graphSize, setGraphSize] = useState({ width: screenWidth, height: screenHeight });
  
  const {
    nodes,
    selectedNode,
    recentAccesses,
    selectNode,
    generateMockData,
    simulateMemoryAccess,
    loadMemoryNodes,
    loading,
    error,
    isUsingBackend,
    initialized,
  } = useMemoryStore();

  // Load memory data on component mount
  useEffect(() => {
    const initializeMemoryData = async () => {
      const nodeCount = Object.keys(nodes).length;
      console.log(`🔄 MemoryViz: Initialized: ${initialized}, Node count: ${nodeCount}, Loading: ${loading}`);
      
      // Only initialize if not already initialized and not currently loading
      if (!initialized && !loading) {
        console.log('🔄 MemoryViz: Initializing memory data...');
        
        // Try to load from backend first
        const backendSuccess = await loadMemoryNodes();
        
        if (!backendSuccess) {
          console.log('📦 MemoryViz: Backend unavailable, using mock data');
          generateMockData();
        }
      }
    };
    
    initializeMemoryData();
  }, []); // Remove dependencies to prevent infinite loop

  // Calculate static graph size (no resize when details panel opens)
  useEffect(() => {
    const headerHeight = 72; // Header with padding
    const statsHeight = 84; // Stats bar with padding
    const tabBarHeight = 60; // Bottom tab bar
    
    const totalReservedHeight = headerHeight + statsHeight + tabBarHeight;
    const availableHeight = screenHeight - totalReservedHeight;
    
    setGraphSize({
      width: screenWidth,
      height: Math.max(200, availableHeight),
    });
  }, []); // Remove showDetails dependency

  const selectedNodeData = selectedNode ? nodes[selectedNode] : null;

  const handleNodeSelect = (nodeId: string) => {
    if (detailsEnabled) {
      setPanelVisible(true);
      setIsExpanded(false); // Start with stub when selecting a node
    }
    // Don't simulate AI access for user clicks - only AI thinking triggers reads
  };

  const handleSimulateThinking = () => {
    // Simulate AI thinking by accessing multiple nodes
    const nodeIds = Object.keys(nodes);
    if (nodeIds.length === 0) return;

    let delay = 0;
    const accessTypes = ['read', 'traverse', 'strengthen', 'read'] as const;
    
    // Simulate a sequence of memory accesses
    nodeIds.slice(0, 4).forEach((nodeId, index) => {
      setTimeout(() => {
        simulateMemoryAccess(nodeId, accessTypes[index % accessTypes.length]);
      }, delay);
      delay += 1000;
    });
  };

  // Handle pan gestures for expanding/collapsing/dismissing panel
  const handlePanGesture = (event: any) => {
    // Handle during active gesture if needed
  };

  const handlePanStateChange = (event: any) => {
    const { state, translationY } = event.nativeEvent;
    
    if (state === State.END) {
      const expandThreshold = 50; // Minimum distance to trigger expand/collapse
      const dismissThreshold = 100; // Minimum distance to dismiss
      
      if (Math.abs(translationY) > expandThreshold) {
        if (translationY < -expandThreshold && !isExpanded) {
          // Swiped up while collapsed - expand
          setIsExpanded(true);
        } else if (translationY > expandThreshold && isExpanded) {
          // Swiped down while expanded - collapse
          setIsExpanded(false);
        } else if (translationY > dismissThreshold && !isExpanded) {
          // Swiped down while collapsed - dismiss completely
          setPanelVisible(false);
          setIsExpanded(false);
        }
      }
    }
  };

  const getTimeSinceAccess = (timestamp: number): string => {
    const diff = Date.now() - timestamp;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return `${seconds}s ago`;
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Memory Graph</Text>
        <View style={styles.headerButtons}>
          <Pressable style={styles.headerButton} onPress={handleSimulateThinking}>
            <Feather name="zap" size={20} color="#007AFF" />
            <Text style={styles.headerButtonText}>Simulate</Text>
          </Pressable>
          <Pressable 
            style={styles.headerButton} 
            onPress={() => {
              setDetailsEnabled(!detailsEnabled);
              if (!detailsEnabled) {
                // If disabling details, hide any open panel
                setPanelVisible(false);
                setIsExpanded(false);
              }
            }}
          >
            <Feather name={detailsEnabled ? "eye" : "eye-off"} size={20} color="#007AFF" />
            <Text style={styles.headerButtonText}>Details</Text>
          </Pressable>
        </View>
      </View>

      {/* Stats Bar */}
      <View style={styles.statsBar}>
        <View style={styles.stat}>
          <Text style={styles.statValue}>{Object.keys(nodes).length}</Text>
          <Text style={styles.statLabel}>Nodes</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statValue}>
            {Object.values(nodes).reduce((sum, node) => sum + Object.keys(node.connections).length, 0)}
          </Text>
          <Text style={styles.statLabel}>Connections</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statValue}>{recentAccesses.length}</Text>
          <Text style={styles.statLabel}>Recent Access</Text>
        </View>
      </View>

      {/* 3D Visualization */}
      <View style={styles.visualizationContainer}>
        <MemoryGraph3D
          width={graphSize.width}
          height={graphSize.height}
          onNodeSelect={handleNodeSelect}
        />
        
        
        {/* Details Panel Overlay - Only show when enabled and node is selected */}
        {detailsEnabled && panelVisible && selectedNodeData && (
          <PanGestureHandler
            onGestureEvent={handlePanGesture}
            onHandlerStateChange={handlePanStateChange}
          >
            <View style={[styles.detailsOverlay, { height: isExpanded ? screenHeight * 0.6 : 120 }]}>
              <Pressable 
                style={styles.detailsHeader}
                onPress={() => setIsExpanded(!isExpanded)}
              >
                <View style={styles.dragHandle}>
                  <Feather 
                    name={isExpanded ? "chevron-down" : "chevron-up"} 
                    size={16} 
                    color="#8E8E93" 
                  />
                </View>
                <View style={styles.headerContent}>
                  <Text style={styles.detailsTitle}>
                    {selectedNodeData.summary}
                  </Text>
                </View>
              </Pressable>
              
              <Pressable 
                style={styles.closeButton}
                onPress={() => {
                  setPanelVisible(false);
                  setIsExpanded(false);
                }}
              >
                <Feather name="x" size={16} color="#666" />
              </Pressable>
          
          {isExpanded && (
            <ScrollView style={styles.detailsContent}>
              <View>
                <Text style={styles.nodeTitle}>{selectedNodeData.summary}</Text>
                
                <View style={styles.detailSection}>
                  <Text style={styles.sectionTitle}>Tags</Text>
                  <View style={styles.tagsContainer}>
                    {selectedNodeData.tags.map((tag, index) => (
                      <View key={index} style={styles.tag}>
                        <Text style={styles.tagText}>{tag}</Text>
                      </View>
                    ))}
                  </View>
                </View>

                <View style={styles.detailSection}>
                  <Text style={styles.sectionTitle}>Content</Text>
                  <Text style={styles.contentText}>{selectedNodeData.content}</Text>
                </View>

                <View style={styles.detailSection}>
                  <Text style={styles.sectionTitle}>Concepts</Text>
                  <Text style={styles.conceptsText}>
                    {selectedNodeData.concepts.join(', ')}
                  </Text>
                </View>

                <View style={styles.detailSection}>
                  <Text style={styles.sectionTitle}>Statistics</Text>
                  <View style={styles.statsGrid}>
                    <View style={styles.statItem}>
                      <Text style={styles.statItemValue}>{selectedNodeData.access_count}</Text>
                      <Text style={styles.statItemLabel}>Access Count</Text>
                    </View>
                    <View style={styles.statItem}>
                      <Text style={styles.statItemValue}>
                        {Object.keys(selectedNodeData.connections).length}
                      </Text>
                      <Text style={styles.statItemLabel}>Connections</Text>
                    </View>
                    <View style={styles.statItem}>
                      <Text style={styles.statItemValue}>
                        {getTimeSinceAccess(selectedNodeData.last_accessed)}
                      </Text>
                      <Text style={styles.statItemLabel}>Last Access</Text>
                    </View>
                  </View>
                </View>

                {Object.keys(selectedNodeData.connections).length > 0 && (
                  <View style={styles.detailSection}>
                    <Text style={styles.sectionTitle}>Connections</Text>
                    {Object.entries(selectedNodeData.connections).map(([targetId, connection]) => {
                      const targetNode = nodes[targetId];
                      return (
                        <Pressable
                          key={targetId}
                          style={styles.connectionItem}
                          onPress={() => selectNode(targetId)}
                        >
                          <Text style={styles.connectionTarget}>
                            {targetNode?.summary || `Node ${targetId}`}
                          </Text>
                          <Text style={styles.connectionWeight}>
                            Weight: {connection.outbound_weight.toFixed(2)}
                          </Text>
                        </Pressable>
                      );
                    })}
                  </View>
                )}
              </View>
            </ScrollView>
          )}
            </View>
          </PanGestureHandler>
        )}
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000011',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
    zIndex: 10,
    position: 'relative',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#000',
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  headerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#F2F2F7',
    borderRadius: 16,
  },
  headerButtonText: {
    fontSize: 14,
    color: '#007AFF',
    fontWeight: '500',
  },
  statsBar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 12,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
    zIndex: 10,
    position: 'relative',
  },
  stat: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#007AFF',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  visualizationContainer: {
    flex: 1,
    position: 'relative',
  },
  detailsOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#FFFFFF',
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
    zIndex: 10,
  },
  detailsHeader: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  dragHandle: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 4,
    marginBottom: 4,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingRight: 40, // Make space for close button
  },
  closeButton: {
    position: 'absolute',
    top: 8,
    right: 12,
    padding: 8,
    zIndex: 20,
  },
  detailsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    flex: 1,
    marginRight: 8,
  },
  detailsContent: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  nodeTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#000',
    marginVertical: 12,
  },
  detailSection: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  tag: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  tagText: {
    fontSize: 12,
    color: '#FFFFFF',
    fontWeight: '500',
  },
  contentText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  conceptsText: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statItemValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#007AFF',
  },
  statItemLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  connectionItem: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#F8F9FA',
    borderRadius: 8,
    marginBottom: 6,
  },
  connectionTarget: {
    fontSize: 14,
    fontWeight: '500',
    color: '#000',
  },
  connectionWeight: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  noSelectionText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 32,
    paddingHorizontal: 24,
    lineHeight: 20,
  },
});