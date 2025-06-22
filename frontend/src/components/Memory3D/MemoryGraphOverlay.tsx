import React, { useMemo, useRef, useState, useEffect } from 'react';
import { View, StyleSheet, Pressable, Dimensions, Text } from 'react-native';
import { PanGestureHandler, PinchGestureHandler, State } from 'react-native-gesture-handler';
import Svg, { Line, ClipPath, Rect, G } from 'react-native-svg';
import { useMemoryStore } from '../../stores/memoryStore';
import { MemoryNode } from '../../types/memory';

interface MemoryGraphOverlayProps {
  width?: number;
  height?: number;
}

// Simple Memory Node Component for overlay - completely isolated
const OverlayMemoryNode: React.FC<{
  nodeData: any;
}> = ({ nodeData }) => {
  const halfSize = nodeData.size / 2;

  return (
    <View
      style={[
        styles.memoryNode,
        {
          position: 'absolute',
          left: nodeData.position.x - halfSize,
          top: nodeData.position.y - halfSize,
          width: nodeData.size,
          height: nodeData.size,
          borderRadius: halfSize,
          backgroundColor: nodeData.color,
          borderColor: nodeData.color,
          borderWidth: 0,
          opacity: nodeData.glowIntensity,
          shadowColor: nodeData.color,
          shadowOffset: { width: 0, height: 0 },
          shadowOpacity: nodeData.glowIntensity * 0.8,
          shadowRadius: nodeData.size * nodeData.glowIntensity * 0.8,
        },
      ]}
    >
      {/* Access type indicator for overlay - larger and more visible */}
      {nodeData.isRecentlyAccessed && nodeData.accessType && (
        <View
          style={[
            styles.accessIndicator,
            {
              backgroundColor: nodeData.color,
              shadowColor: nodeData.color,
              shadowOpacity: 0.9,
              shadowRadius: 12,
              shadowOffset: { width: 0, height: 0 },
              width: Math.max(16, nodeData.size * 0.4),
              height: Math.max(16, nodeData.size * 0.4),
              borderRadius: Math.max(8, nodeData.size * 0.2),
              top: -Math.max(8, nodeData.size * 0.2),
              right: -Math.max(8, nodeData.size * 0.2),
            }
          ]}
        >
          <Text style={[
            styles.accessText,
            {
              fontSize: Math.max(10, nodeData.size * 0.25),
              color: '#000011',
              fontWeight: 'bold',
            }
          ]}>
            {nodeData.accessType.charAt(0).toUpperCase()}
          </Text>
        </View>
      )}
      
      {/* Pulsing glow ring for recently accessed nodes - more visible */}
      {nodeData.isRecentlyAccessed && (
        <View
          style={[
            styles.pulseRing,
            {
              width: nodeData.size * 1.8,
              height: nodeData.size * 1.8,
              borderRadius: nodeData.size * 0.9,
              borderColor: nodeData.color,
              borderWidth: Math.max(2, nodeData.size * 0.08),
              left: -(nodeData.size * 0.4),
              top: -(nodeData.size * 0.4),
              opacity: nodeData.glowIntensity * 0.7,
            }
          ]}
        />
      )}
    </View>
  );
};

// Isolated Memory Graph for Chat Overlay
export const MemoryGraphOverlay: React.FC<MemoryGraphOverlayProps> = ({
  width = 400,
  height = 400,
}) => {
  const { nodes, recentAccesses } = useMemoryStore();

  // Generate fixed 3D positions for overlay nodes first
  const overlayNodePositions = useMemo(() => {
    const positions: Record<string, { x: number; y: number; z: number }> = {};
    
    const clusters: Record<string, { center: [number, number, number]; radius: number }> = {
      'conversation': { center: [0, 0, 0], radius: 100 },
      'technical': { center: [150, 50, -100], radius: 80 },
      'visualization': { center: [-120, 100, 80], radius: 70 },
      'frontend': { center: [80, -100, 120], radius: 80 },
      'memory_system': { center: [-60, 150, -100], radius: 90 },
      'vector_space': { center: [100, 80, -50], radius: 60 },
    };
    
    Object.entries(nodes).forEach(([nodeId, node]) => {
      let clusterKey = 'conversation';
      for (const tag of node.tags) {
        if (clusters[tag]) {
          clusterKey = tag;
          break;
        }
      }
      
      const cluster = clusters[clusterKey];
      
      // Use seed for consistent positioning
      const seed = parseInt(nodeId) || 1;
      const theta = (seed * 137.5) % 360 * (Math.PI / 180);
      const phi = Math.acos(1 - 2 * ((seed * 0.618) % 1));
      const r = ((seed * 0.382) % 1) * cluster.radius;
      
      const x = cluster.center[0] + r * Math.sin(phi) * Math.cos(theta);
      const y = cluster.center[1] + r * Math.cos(phi);
      const z = cluster.center[2] + r * Math.sin(phi) * Math.sin(theta);
      
      positions[nodeId] = { x, y, z };
    });
    
    return positions;
  }, [nodes]);

  // Calculate optimal camera distance to fit all nodes
  const calculateOverlayCamera = useMemo(() => {
    if (Object.keys(overlayNodePositions).length === 0) {
      return {
        target: { x: 0, y: 0, z: 0 },
        spherical: { 
          azimuth: Math.PI / 4,
          elevation: Math.PI / 6,
          distance: 600,
        },
        zoom: 1,
      };
    }

    // Calculate bounding box of all nodes
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    Object.values(overlayNodePositions).forEach(pos => {
      minX = Math.min(minX, pos.x);
      maxX = Math.max(maxX, pos.x);
      minY = Math.min(minY, pos.y);
      maxY = Math.max(maxY, pos.y);
      minZ = Math.min(minZ, pos.z);
      maxZ = Math.max(maxZ, pos.z);
    });
    
    // Calculate center and size
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    
    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ);
    
    // Calculate distance to fit all nodes with padding
    const fov = Math.PI / 4; // 45 degrees
    const distance = (maxSize / 2) / Math.tan(fov / 2) * 2.0; // 2x padding for better view
    
    return {
      target: { x: centerX, y: centerY, z: centerZ },
      spherical: { 
        azimuth: Math.PI / 4,
        elevation: Math.PI / 6,
        distance: Math.max(400, Math.min(1000, distance)),
      },
      zoom: 1.1, // Slightly zoomed in for better node visibility
    };
  }, [overlayNodePositions]);

  // Define viewport bounds
  const viewport = {
    left: 0,
    top: 0,
    right: width,
    bottom: height,
    width: width,
    height: height,
    centerX: width / 2,
    centerY: height / 2,
  };

  // Calculate camera position from spherical coordinates
  const getCameraPosition = () => {
    const { azimuth, elevation, distance } = calculateOverlayCamera.spherical;
    const { target } = calculateOverlayCamera;
    
    const x = target.x + distance * Math.cos(elevation) * Math.sin(azimuth);
    const y = target.y + distance * Math.sin(elevation);
    const z = target.z + distance * Math.cos(elevation) * Math.cos(azimuth);
    
    return { x, y, z };
  };

  // 3D to 2D projection for overlay
  const project3DToScreen = (worldPos: { x: number; y: number; z: number }) => {
    const cameraPos = getCameraPosition();
    const { target, zoom } = calculateOverlayCamera;
    
    const relativeToCamera = {
      x: worldPos.x - cameraPos.x,
      y: worldPos.y - cameraPos.y,
      z: worldPos.z - cameraPos.z,
    };
    
    const forward = {
      x: target.x - cameraPos.x,
      y: target.y - cameraPos.y,
      z: target.z - cameraPos.z,
    };
    
    const fLen = Math.sqrt(forward.x ** 2 + forward.y ** 2 + forward.z ** 2);
    if (fLen < 0.001) return null;
    
    forward.x /= fLen;
    forward.y /= fLen;
    forward.z /= fLen;
    
    const right = {
      x: forward.z,
      y: 0,
      z: -forward.x,
    };
    
    const rLen = Math.sqrt(right.x ** 2 + right.z ** 2);
    if (rLen > 0.001) {
      right.x /= rLen;
      right.z /= rLen;
    } else {
      right.x = 1;
      right.z = 0;
    }
    
    const up = {
      x: right.y * forward.z - right.z * forward.y,
      y: right.z * forward.x - right.x * forward.z,
      z: right.x * forward.y - right.y * forward.x,
    };
    
    const viewSpace = {
      x: relativeToCamera.x * right.x + relativeToCamera.y * right.y + relativeToCamera.z * right.z,
      y: relativeToCamera.x * up.x + relativeToCamera.y * up.y + relativeToCamera.z * up.z,
      z: relativeToCamera.x * forward.x + relativeToCamera.y * forward.y + relativeToCamera.z * forward.z,
    };
    
    if (viewSpace.z <= 0.1) {
      return null;
    }
    
    const perspective = 400;
    const projected = {
      x: (viewSpace.x * perspective) / viewSpace.z,
      y: (viewSpace.y * perspective) / viewSpace.z,
      depth: viewSpace.z,
      scale: perspective / viewSpace.z,
    };
    
    const screenX = viewport.centerX + projected.x * zoom;
    const screenY = viewport.centerY - projected.y * zoom;
    
    const isVisible = screenX >= viewport.left && 
                     screenX <= viewport.right && 
                     screenY >= viewport.top && 
                     screenY <= viewport.bottom;
    
    return {
      x: screenX,
      y: screenY,
      scale: projected.scale * zoom,
      depth: projected.depth,
      isVisible,
    };
  };


  // Track recent memory accesses for overlay
  const overlayRecentAccessMap = useMemo(() => {
    const map: Record<string, { accessType: string; timestamp: number }> = {};
    const cutoffTime = Date.now() - 5000;
    
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

  // Get overlay node color
  const getOverlayNodeColor = (node: MemoryNode, isRecentlyAccessed: boolean, accessType?: string) => {
    if (isRecentlyAccessed && accessType) {
      const colors = {
        'read': '#87CEEB',
        'write': '#98FB98',
        'strengthen': '#FFD700',
        'traverse': '#DDA0DD',
      };
      return colors[accessType] || '#FFA07A';
    }
    return '#E6E6FA';
  };

  // Get overlay glow intensity
  const getOverlayGlowIntensity = (isRecentlyAccessed: boolean) => {
    return isRecentlyAccessed ? 0.8 : 0.3;
  };

  // Calculate overlay node size - larger for better visibility
  const getOverlayNodeSize = (node: MemoryNode, scale: number) => {
    const baseSize = 35; // Increased from 20
    const importanceBonus = Math.log(node.access_count + 1) * 4;
    return Math.max(25, Math.min(60, (baseSize + importanceBonus) * scale));
  };

  // Project all overlay nodes
  const overlayProjectedNodes = useMemo(() => {
    const projected: Array<{
      nodeId: string;
      node: MemoryNode;
      position: { x: number; y: number };
      scale: number;
      depth: number;
      isVisible: boolean;
      color: string;
      size: number;
      isRecentlyAccessed: boolean;
      accessType?: string;
      glowIntensity: number;
    }> = [];
    
    Object.entries(nodes).forEach(([nodeId, node]) => {
      const worldPos = overlayNodePositions[nodeId];
      if (!worldPos) return;
      
      const screenPos = project3DToScreen(worldPos);
      if (!screenPos || !screenPos.isVisible) return;
      
      const recentAccess = overlayRecentAccessMap[nodeId];
      const isRecentlyAccessed = !!recentAccess;
      const color = getOverlayNodeColor(node, isRecentlyAccessed, recentAccess?.accessType);
      const size = getOverlayNodeSize(node, screenPos.scale);
      const glowIntensity = getOverlayGlowIntensity(isRecentlyAccessed);
      
      projected.push({
        nodeId,
        node,
        position: { x: screenPos.x, y: screenPos.y },
        scale: screenPos.scale,
        depth: screenPos.depth,
        isVisible: true,
        color,
        size,
        isRecentlyAccessed,
        accessType: recentAccess?.accessType,
        glowIntensity,
      });
    });
    
    return projected.sort((a, b) => b.depth - a.depth);
  }, [nodes, overlayNodePositions, overlayRecentAccessMap]);

  // Project overlay connections
  const overlayProjectedConnections = useMemo(() => {
    const connections: Array<{
      id: string;
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      weight: number;
      color: string;
      glowIntensity: number;
    }> = [];
    
    const processed = new Set<string>();
    
    Object.entries(nodes).forEach(([nodeId, node]) => {
      const startPos = overlayNodePositions[nodeId];
      if (!startPos) return;
      
      const startScreen = project3DToScreen(startPos);
      if (!startScreen) return;
      
      Object.entries(node.connections).forEach(([targetId, connection]) => {
        const connectionId = nodeId < targetId ? `${nodeId}-${targetId}` : `${targetId}-${nodeId}`;
        if (processed.has(connectionId)) return;
        processed.add(connectionId);
        
        const endPos = overlayNodePositions[targetId];
        if (!endPos) return;
        
        const endScreen = project3DToScreen(endPos);
        if (!endScreen) return;
        
        const isVisible = startScreen.isVisible || endScreen.isVisible;
        if (!isVisible) return;
        
        const weight = connection.outbound_weight;
        const color = weight > 0.7 ? '#00BFFF' :
                     weight > 0.5 ? '#4169E1' :
                     weight > 0.3 ? '#6495ED' : '#483D8B';
        
        const hasRecentAccess = overlayRecentAccessMap[nodeId] || overlayRecentAccessMap[targetId];
        const baseGlowIntensity = Math.max(0.2, weight);
        const boostedGlow = hasRecentAccess ? Math.min(1, baseGlowIntensity * 1.5) : baseGlowIntensity;
        
        connections.push({
          id: connectionId,
          x1: startScreen.x,
          y1: startScreen.y,
          x2: endScreen.x,
          y2: endScreen.y,
          weight,
          color,
          glowIntensity: boostedGlow,
        });
      });
    });
    
    return connections;
  }, [nodes, overlayNodePositions, overlayRecentAccessMap]);

  return (
    <View style={[styles.container, { width, height }]}>
      <View style={styles.visualizationArea}>
        {/* SVG for connections */}
        <Svg
          width={width}
          height={height}
          style={StyleSheet.absoluteFill}
        >
          <ClipPath id="overlayViewportClip">
            <Rect
              x={viewport.left}
              y={viewport.top}
              width={viewport.width}
              height={viewport.height}
            />
          </ClipPath>
          
          <G clipPath="url(#overlayViewportClip)">
            {overlayProjectedConnections.map((connection) => {
              const baseWidth = Math.max(1, connection.weight * 4);
              
              return (
                <G key={connection.id}>
                  <Line
                    x1={connection.x1}
                    y1={connection.y1}
                    x2={connection.x2}
                    y2={connection.y2}
                    stroke={connection.color}
                    strokeWidth={baseWidth * 3}
                    strokeOpacity={connection.glowIntensity * 0.2}
                  />
                  <Line
                    x1={connection.x1}
                    y1={connection.y1}
                    x2={connection.x2}
                    y2={connection.y2}
                    stroke={connection.color}
                    strokeWidth={baseWidth * 2}
                    strokeOpacity={connection.glowIntensity * 0.4}
                  />
                  <Line
                    x1={connection.x1}
                    y1={connection.y1}
                    x2={connection.x2}
                    y2={connection.y2}
                    stroke={connection.color}
                    strokeWidth={baseWidth}
                    strokeOpacity={connection.glowIntensity * 0.8}
                  />
                </G>
              );
            })}
          </G>
        </Svg>
        
        {/* Render overlay nodes */}
        {overlayProjectedNodes.map((nodeData) => {
          const halfSize = nodeData.size / 2;
          
          if (nodeData.position.x - halfSize > viewport.right ||
              nodeData.position.x + halfSize < viewport.left ||
              nodeData.position.y - halfSize > viewport.bottom ||
              nodeData.position.y + halfSize < viewport.top) {
            return null;
          }
          
          return (
            <OverlayMemoryNode
              key={nodeData.nodeId}
              nodeData={nodeData}
            />
          );
        })}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'transparent',
    overflow: 'hidden',
  },
  visualizationArea: {
    flex: 1,
    position: 'relative',
  },
  memoryNode: {
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  accessIndicator: {
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 17, 0.2)',
  },
  accessText: {
    fontWeight: 'bold',
  },
  pulseRing: {
    position: 'absolute',
    backgroundColor: 'transparent',
  },
});