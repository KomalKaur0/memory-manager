import React, { useMemo, useRef, useState } from 'react';
import { View, StyleSheet, Text, Animated, Dimensions, Pressable } from 'react-native';
import { PanGestureHandler, PinchGestureHandler, State } from 'react-native-gesture-handler';
import { Feather } from '@expo/vector-icons';
import Svg, { Line } from 'react-native-svg';
import { useMemoryStore } from '../../stores/memoryStore';
import { MemoryNode } from '../../types/memory';

interface MemoryGraph3DProps {
  width?: number;
  height?: number;
  onNodeSelect?: (nodeId: string) => void;
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// 3D Vector Space Memory Visualization with Zoom/Pan Controls
export const MemoryGraph3D: React.FC<MemoryGraph3DProps> = ({
  width = 400,
  height = 400,
  onNodeSelect,
}) => {
  const {
    nodes,
    selectedNode,
    recentAccesses,
    selectNode,
  } = useMemoryStore();

  // Transform states
  const [translateX] = useState(new Animated.Value(0));
  const [translateY] = useState(new Animated.Value(0));
  const [scale] = useState(new Animated.Value(1));
  const [rotation] = useState(new Animated.Value(0));

  // Gesture refs
  const panRef = useRef();
  const pinchRef = useRef();

  // Reset transform function
  const resetTransform = () => {
    Animated.parallel([
      Animated.spring(translateX, { toValue: 0, useNativeDriver: true }),
      Animated.spring(translateY, { toValue: 0, useNativeDriver: true }),
      Animated.spring(scale, { toValue: 1, useNativeDriver: true }),
      Animated.spring(rotation, { toValue: 0, useNativeDriver: true }),
    ]).start();
  };

  // Pan gesture handler
  const onPanGestureEvent = Animated.event(
    [{ nativeEvent: { translationX: translateX, translationY: translateY } }],
    { useNativeDriver: true }
  );

  // Pinch gesture handler
  const onPinchGestureEvent = Animated.event(
    [{ nativeEvent: { scale: scale } }],
    { useNativeDriver: true }
  );

  // Enhanced memory positioning based on semantic embeddings
  const nodePositions = useMemo(() => {
    const positions: Record<string, { x: number; y: number; z: number }> = {};
    
    // Create semantic clusters in 3D vector space
    const clusters = {
      'conversation': { center: [0, 0, 0], spread: 80 },
      'technical': { center: [120, 40, -60], spread: 70 },
      'visualization': { center: [-100, 80, 70], spread: 60 },
      'frontend': { center: [60, -80, 100], spread: 70 },
      'memory_system': { center: [-40, 120, -80], spread: 80 },
      'vector_space': { center: [80, 60, -40], spread: 50 },
    };
    
    Object.entries(nodes).forEach(([nodeId, node]) => {
      // Determine primary cluster based on tags/concepts
      let primaryCluster = 'conversation';
      for (const tag of node.tags) {
        if (clusters[tag]) {
          primaryCluster = tag;
          break;
        }
      }
      
      const cluster = clusters[primaryCluster];
      
      // Position based on semantic similarity within cluster
      const angle = Math.random() * Math.PI * 2;
      const distance = Math.random() * cluster.spread;
      const height = (Math.random() - 0.5) * cluster.spread;
      
      const x = cluster.center[0] + Math.cos(angle) * distance;
      const y = cluster.center[1] + height;
      const z = cluster.center[2] + Math.sin(angle) * distance;
      
      positions[nodeId] = { x, y, z };
    });

    return positions;
  }, [nodes]);

  // Project 3D to 2D with perspective
  const project3DTo2D = (x: number, y: number, z: number) => {
    const perspective = 400;
    const distance = perspective + z;
    const projectedX = (x * perspective) / distance;
    const projectedY = (y * perspective) / distance;
    
    return {
      x: width / 2 + projectedX,
      y: height / 2 + projectedY,
      scale: perspective / distance,
    };
  };

  // Get recent access events for highlighting
  const recentAccessMap = useMemo(() => {
    const map: Record<string, { accessType: string; timestamp: number }> = {};
    const cutoffTime = Date.now() - 5000; // 5 seconds
    
    recentAccesses.forEach(access => {
      if (access.timestamp > cutoffTime) {
        map[access.node_id] = {
          accessType: access.access_type,
          timestamp: access.timestamp,
        };
      }
    });
    
    return map;
  }, [recentAccesses]);

  const handleNodePress = (nodeId: string) => {
    selectNode(nodeId === selectedNode ? null : nodeId);
    onNodeSelect?.(nodeId);
  };

  const getNodeColor = (node: MemoryNode, isRecentlyAccessed: boolean, accessType?: string) => {
    if (selectedNode === node.id) return '#FFD700'; // Gold for selected
    if (isRecentlyAccessed) {
      switch (accessType) {
        case 'read': return '#4A90E2';
        case 'write': return '#7ED321';
        case 'strengthen': return '#F5A623';
        case 'traverse': return '#BD10E0';
        default: return '#FF6B6B';
      }
    }
    
    // Color based on semantic clusters
    if (node.tags.includes('conversation')) return '#87CEEB';
    if (node.tags.includes('technical')) return '#98FB98';
    if (node.tags.includes('visualization')) return '#DDA0DD';
    if (node.tags.includes('frontend')) return '#F0E68C';
    if (node.tags.includes('memory_system')) return '#FFB6C1';
    if (node.tags.includes('vector_space')) return '#FFA07A';
    
    return '#B0B0B0'; // Default gray
  };

  const getNodeSize = (node: MemoryNode, projectedScale: number) => {
    const baseSize = 25;
    const accessMultiplier = Math.log(node.access_count + 1) * 5;
    return Math.max(15, (baseSize + accessMultiplier) * projectedScale);
  };

  // Calculate all projected positions
  const projectedNodes = useMemo(() => {
    return Object.entries(nodes).map(([nodeId, node]) => {
      const pos = nodePositions[nodeId];
      if (!pos) return null;
      
      const projected = project3DTo2D(pos.x, pos.y, pos.z);
      const recentAccess = recentAccessMap[nodeId];
      const isRecentlyAccessed = !!recentAccess;
      const nodeColor = getNodeColor(node, isRecentlyAccessed, recentAccess?.accessType);
      const nodeSize = getNodeSize(node, projected.scale);
      const isSelected = selectedNode === nodeId;

      return {
        nodeId,
        node,
        projected,
        recentAccess,
        isRecentlyAccessed,
        nodeColor,
        nodeSize,
        isSelected,
        depth: pos.z,
      };
    }).filter(Boolean).sort((a, b) => b!.depth - a!.depth); // Sort by depth (back to front)
  }, [nodes, nodePositions, recentAccessMap, selectedNode]);

  // Calculate connections for SVG
  const connections = useMemo(() => {
    const connectionList: Array<{
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      weight: number;
      color: string;
    }> = [];

    Object.entries(nodes).forEach(([nodeId, node]) => {
      const startPos = nodePositions[nodeId];
      if (!startPos) return;

      const startProjected = project3DTo2D(startPos.x, startPos.y, startPos.z);

      Object.entries(node.connections).forEach(([targetId, connection]) => {
        const endPos = nodePositions[targetId];
        if (!endPos) return;

        const endProjected = project3DTo2D(endPos.x, endPos.y, endPos.z);
        const weight = connection.outbound_weight;
        
        const color = weight > 0.8 ? '#FF6B6B' : 
                     weight > 0.6 ? '#FFA500' : 
                     weight > 0.4 ? '#4A90E2' : '#666666';

        connectionList.push({
          x1: startProjected.x,
          y1: startProjected.y,
          x2: endProjected.x,
          y2: endProjected.y,
          weight,
          color,
        });
      });
    });

    return connectionList;
  }, [nodes, nodePositions]);

  return (
    <View style={[styles.container, { width, height }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Memory Vector Space</Text>
        <Text style={styles.subtitle}>
          {Object.keys(nodes).length} memories • 3D projection of embedding space
        </Text>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <Pressable style={styles.resetButton} onPress={resetTransform}>
          <Feather name="home" size={16} color="#FFFFFF" />
          <Text style={styles.resetButtonText}>Reset View</Text>
        </Pressable>
        <Text style={styles.controlsHint}>
          Pinch to zoom • Drag to pan
        </Text>
      </View>

      {/* Interactive 3D Visualization */}
      <PinchGestureHandler
        ref={pinchRef}
        onGestureEvent={onPinchGestureEvent}
        onHandlerStateChange={(event) => {
          if (event.nativeEvent.state === State.END) {
            scale.extractOffset();
          }
        }}
      >
        <Animated.View style={styles.gestureContainer}>
          <PanGestureHandler
            ref={panRef}
            onGestureEvent={onPanGestureEvent}
            onHandlerStateChange={(event) => {
              if (event.nativeEvent.state === State.END) {
                translateX.extractOffset();
                translateY.extractOffset();
              }
            }}
            simultaneousHandlers={pinchRef}
          >
            <Animated.View
              style={[
                styles.visualizationArea,
                {
                  transform: [
                    { translateX },
                    { translateY },
                    { scale },
                  ],
                },
              ]}
            >
              {/* SVG for connection lines */}
              <Svg style={StyleSheet.absoluteFill} width={width} height={height - 120}>
                {connections.map((connection, index) => (
                  <Line
                    key={index}
                    x1={connection.x1}
                    y1={connection.y1}
                    x2={connection.x2}
                    y2={connection.y2}
                    stroke={connection.color}
                    strokeWidth={Math.max(1, connection.weight * 3)}
                    strokeOpacity={connection.weight * 0.8}
                  />
                ))}
              </Svg>

              {/* Memory nodes */}
              {projectedNodes?.map((nodeData) => {
                if (!nodeData) return null;
                
                const {
                  nodeId,
                  node,
                  projected,
                  isRecentlyAccessed,
                  nodeColor,
                  nodeSize,
                  isSelected,
                  recentAccess,
                } = nodeData;

                // Clamp position to stay within bounds
                const clampedX = Math.max(nodeSize/2, Math.min(width - nodeSize/2, projected.x));
                const clampedY = Math.max(nodeSize/2, Math.min(height - 120, projected.y));

                return (
                  <Animated.View
                    key={nodeId}
                    style={[
                      styles.memoryNode,
                      {
                        left: clampedX - nodeSize/2,
                        top: clampedY - nodeSize/2,
                        width: nodeSize,
                        height: nodeSize,
                        borderRadius: nodeSize/2,
                        backgroundColor: nodeColor,
                        borderColor: isSelected ? '#FFD700' : nodeColor,
                        borderWidth: isSelected ? 3 : 1,
                        opacity: projected.scale * 0.8 + 0.2,
                        shadowColor: nodeColor,
                        shadowOpacity: isRecentlyAccessed ? 0.8 : 0.3,
                        shadowRadius: isRecentlyAccessed ? 8 : 4,
                        elevation: 5,
                        transform: [
                          { scale: isRecentlyAccessed ? 1.2 : 1 },
                        ],
                      }
                    ]}
                    onTouchEnd={() => handleNodePress(nodeId)}
                  >
                    {/* Access type indicator */}
                    {isRecentlyAccessed && (
                      <View style={styles.accessIndicator}>
                        <Text style={styles.accessText}>
                          {recentAccess?.accessType.charAt(0).toUpperCase()}
                        </Text>
                      </View>
                    )}
                  </Animated.View>
                );
              })}
            </Animated.View>
          </PanGestureHandler>
        </Animated.View>
      </PinchGestureHandler>

      {/* Selected node details */}
      {selectedNode && nodes[selectedNode] && (
        <View style={styles.selectedNodePanel}>
          <Text style={styles.selectedNodeTitle}>
            {nodes[selectedNode].summary}
          </Text>
          <Text style={styles.selectedNodeDetails}>
            {nodes[selectedNode].concepts.join(' • ')}
          </Text>
          <Text style={styles.selectedNodeStats}>
            Accessed {nodes[selectedNode].access_count} times • {Object.keys(nodes[selectedNode].connections).length} connections
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#000011',
    flex: 1,
  },
  header: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  title: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
  },
  subtitle: {
    color: '#AAAAAA',
    fontSize: 12,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#001122',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#007AFF',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  resetButtonText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  controlsHint: {
    color: '#888888',
    fontSize: 11,
  },
  gestureContainer: {
    flex: 1,
  },
  visualizationArea: {
    flex: 1,
    position: 'relative',
  },
  memoryNode: {
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
    shadowOffset: { width: 0, height: 2 },
  },
  accessIndicator: {
    position: 'absolute',
    top: -5,
    right: -5,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#FF4444',
    justifyContent: 'center',
    alignItems: 'center',
  },
  accessText: {
    color: '#FFFFFF',
    fontSize: 8,
    fontWeight: 'bold',
  },
  selectedNodePanel: {
    backgroundColor: '#001122',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  selectedNodeTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  selectedNodeDetails: {
    color: '#CCCCCC',
    fontSize: 14,
    marginBottom: 8,
  },
  selectedNodeStats: {
    color: '#888888',
    fontSize: 12,
  },
});