import React, { useMemo, useRef, useState, useEffect, useCallback, startTransition } from 'react';
import { View, StyleSheet, Text, Dimensions, Pressable, Animated } from 'react-native';
import { PanGestureHandler, PinchGestureHandler, State } from 'react-native-gesture-handler';
import { Feather } from '@expo/vector-icons';
import Svg, { Line, ClipPath, Rect, G } from 'react-native-svg';
import { useMemoryStore } from '../../stores/memoryStore';
import { MemoryNode } from '../../types/memory';

interface MemoryGraph3DProps {
  width?: number;
  height?: number;
  onNodeSelect?: (nodeId: string) => void;
}

// Simple Memory Node Component - using basic styling to avoid animation conflicts
const SimpleMemoryNode: React.FC<{
  nodeData: any;
  onPress: () => void;
}> = ({ nodeData, onPress }) => {
  const halfSize = nodeData.size / 2;

  return (
    <Pressable
      style={[
        styles.memoryNode,
        {
          position: 'absolute',
          left: nodeData.position.x - halfSize,
          top: nodeData.position.y - halfSize,
          width: nodeData.size,
          height: nodeData.size,
          borderRadius: halfSize,
          backgroundColor: 'transparent', // Make main container transparent
          borderColor: nodeData.color,
          borderWidth: 0,
          transform: [
            { scale: nodeData.isSelected ? 1.3 : 1 },
          ],
        },
      ]}
      onPress={onPress}
    >
      {/* Soft outer glow - subtle */}
      <View
        style={[
          StyleSheet.absoluteFill,
          {
            borderRadius: halfSize,
            backgroundColor: nodeData.color,
            opacity: Math.max(0.1, nodeData.glowIntensity * 0.2),
            shadowColor: nodeData.color,
            shadowOffset: { width: 0, height: 0 },
            shadowOpacity: 0.6,
            shadowRadius: nodeData.size * 0.4,
            transform: [{ scale: 1.15 }],
          },
        ]}
      />
      
      {/* Main solid core - more prominent */}
      <View
        style={[
          StyleSheet.absoluteFill,
          {
            borderRadius: halfSize,
            backgroundColor: nodeData.color,
            opacity: Math.max(0.6, nodeData.glowIntensity),
            shadowColor: nodeData.color,
            shadowOffset: { width: 0, height: 0 },
            shadowOpacity: 0.3,
            shadowRadius: nodeData.size * 0.2,
          },
        ]}
      />
      {/* Access type indicator - glowing ring for active nodes */}
      {nodeData.isRecentlyAccessed && nodeData.accessType && (
        <View
          style={[
            styles.accessIndicator,
            {
              backgroundColor: nodeData.color,
              shadowColor: nodeData.color,
              shadowOpacity: 0.8,
              shadowRadius: 8,
              shadowOffset: { width: 0, height: 0 },
            }
          ]}
        >
          <Text style={styles.accessText}>
            {nodeData.accessType.charAt(0).toUpperCase()}
          </Text>
        </View>
      )}
      
      {/* Pulsing glow ring for recently accessed nodes */}
      {nodeData.isRecentlyAccessed && (
        <View
          style={[
            styles.pulseRing,
            {
              width: nodeData.size * 1.5,
              height: nodeData.size * 1.5,
              borderRadius: nodeData.size * 0.75,
              borderColor: nodeData.color,
              left: -(nodeData.size * 0.25),
              top: -(nodeData.size * 0.25),
              opacity: nodeData.glowIntensity,
            }
          ]}
        />
      )}
    </Pressable>
  );
};


// 3D Vector Space Memory Visualization with proper viewport clipping
export const MemoryGraph3D: React.FC<MemoryGraph3DProps> = ({
  width = 400,
  height = 400,
  onNodeSelect,
}) => {
  const {
    nodes,
    selectedNode,
    recentAccesses,
    isThinking,
    selectNode,
  } = useMemoryStore();

  // Define viewport bounds (content renders within full component area)
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

  // Camera state - single source of truth
  const [camera, setCamera] = useState({
    // World space position (what we're looking at)
    target: { x: 0, y: 0, z: 0 },
    // Spherical coordinates around target (azimuth, elevation, distance)
    spherical: { 
      azimuth: Math.PI / 4,    // Horizontal rotation angle
      elevation: Math.PI / 6,   // Vertical rotation angle  
      distance: 400,            // Will be auto-calculated to fit nodes
    },
    // Zoom factor
    zoom: 1,
  });

  // Gesture state tracking - capture start state
  const gestureStateRef = useRef({
    lastTranslationX: 0,
    lastTranslationY: 0,
    accumulatedAzimuth: 0,    // Total accumulated rotation
    accumulatedElevation: 0,  // Total accumulated rotation
    lastPinchScale: 1,
    frameCount: 0,
  });

  // Refs for gesture handlers
  const panRef = useRef<any>();
  const pinchRef = useRef<any>();

  // Calculate camera position from spherical coordinates
  const getCameraPosition = () => {
    const { azimuth, elevation, distance } = camera.spherical;
    const { target } = camera;
    
    // Convert spherical to cartesian
    const x = target.x + distance * Math.cos(elevation) * Math.sin(azimuth);
    const y = target.y + distance * Math.sin(elevation);
    const z = target.z + distance * Math.cos(elevation) * Math.cos(azimuth);
    
    return { x, y, z };
  };

  // Clean 3D to 2D projection pipeline
  const project3DToScreen = (worldPos: { x: number; y: number; z: number }) => {
    const cameraPos = getCameraPosition();
    const { target, zoom } = camera;
    
    // Step 1: Transform world point to be relative to camera
    const relativeToCamera = {
      x: worldPos.x - cameraPos.x,
      y: worldPos.y - cameraPos.y,
      z: worldPos.z - cameraPos.z,
    };
    
    // Step 2: Create view matrix (camera looking at target)
    // Forward vector: from camera to target (what we're looking at)
    const forward = {
      x: target.x - cameraPos.x,
      y: target.y - cameraPos.y,
      z: target.z - cameraPos.z,
    };
    
    // Normalize forward
    const fLen = Math.sqrt(forward.x ** 2 + forward.y ** 2 + forward.z ** 2);
    if (fLen < 0.001) return null; // Camera too close to target
    
    forward.x /= fLen;
    forward.y /= fLen;
    forward.z /= fLen;
    
    // Right vector (cross product of forward and world up)
    const worldUp = { x: 0, y: 1, z: 0 };
    const right = {
      x: forward.z,  // Simplified for y-up coordinate system
      y: 0,
      z: -forward.x,
    };
    
    // Normalize right
    const rLen = Math.sqrt(right.x ** 2 + right.z ** 2);
    if (rLen > 0.001) {
      right.x /= rLen;
      right.z /= rLen;
    } else {
      // Forward is parallel to up, use alternative right vector
      right.x = 1;
      right.z = 0;
    }
    
    // Up vector (cross product of right and forward)
    const up = {
      x: right.y * forward.z - right.z * forward.y,
      y: right.z * forward.x - right.x * forward.z,
      z: right.x * forward.y - right.y * forward.x,
    };
    
    // Apply view transformation (transform to camera space)
    const viewSpace = {
      x: relativeToCamera.x * right.x + relativeToCamera.y * right.y + relativeToCamera.z * right.z,
      y: relativeToCamera.x * up.x + relativeToCamera.y * up.y + relativeToCamera.z * up.z,
      z: relativeToCamera.x * forward.x + relativeToCamera.y * forward.y + relativeToCamera.z * forward.z,
    };
    
    // Step 3: Perspective projection
    if (viewSpace.z <= 0.1) {
      // Behind or too close to camera
      return null;
    }
    
    const perspective = 400;
    const projected = {
      x: (viewSpace.x * perspective) / viewSpace.z,
      y: (viewSpace.y * perspective) / viewSpace.z,
      depth: viewSpace.z,
      scale: perspective / viewSpace.z,
    };
    
    // Step 4: Map to viewport with zoom
    const screenX = viewport.centerX + projected.x * zoom;
    const screenY = viewport.centerY - projected.y * zoom; // Flip Y for screen coordinates
    
    // Step 5: Check if within viewport bounds
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

  // Calculate camera distance to fit all nodes
  const calculateFitDistance = () => {
    if (Object.keys(nodePositions).length === 0) return 400;
    
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    Object.values(nodePositions).forEach(pos => {
      minX = Math.min(minX, pos.x);
      maxX = Math.max(maxX, pos.x);
      minY = Math.min(minY, pos.y);
      maxY = Math.max(maxY, pos.y);
      minZ = Math.min(minZ, pos.z);
      maxZ = Math.max(maxZ, pos.z);
    });
    
    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ);
    
    const fov = Math.PI / 4;
    const distance = (maxSize / 2) / Math.tan(fov / 2) * 1.5;
    
    return Math.max(200, Math.min(800, distance));
  };

  // Reset camera to show all nodes
  const resetCamera = () => {
    // Don't reset camera during thinking to prevent jumping
    if (isThinking) return;
    
    const positions = Object.values(nodePositions);
    if (positions.length === 0) return;
    
    // Calculate center of all nodes
    const center = positions.reduce((acc, pos) => ({
      x: acc.x + pos.x / positions.length,
      y: acc.y + pos.y / positions.length,
      z: acc.z + pos.z / positions.length,
    }), { x: 0, y: 0, z: 0 });
    
    setCamera({
      target: center,
      spherical: {
        azimuth: Math.PI / 4,
        elevation: Math.PI / 6,
        distance: calculateFitDistance(),
      },
      zoom: 1,
    });
  };

  // Handle pan gestures for rotation - using delta approach
  const onPanGestureEvent = (event: any) => {
    // Don't allow camera movement during thinking to prevent jumping
    if (isThinking) return;
    
    const { translationX, translationY } = event.nativeEvent;
    
    // Calculate frame-to-frame delta
    const deltaX = translationX - gestureStateRef.current.lastTranslationX;
    const deltaY = translationY - gestureStateRef.current.lastTranslationY;
    
    // Detect new gesture by large jump in translation or reset to near zero
    const isNewGesture = Math.abs(deltaX) > 20 || Math.abs(deltaY) > 20 ||
                        (Math.abs(translationX) < 5 && Math.abs(translationY) < 5 && 
                         gestureStateRef.current.frameCount > 5);
    
    if (isNewGesture) {
      // Reset for new gesture
      gestureStateRef.current.lastTranslationX = translationX;
      gestureStateRef.current.lastTranslationY = translationY;
      gestureStateRef.current.accumulatedAzimuth = camera.spherical.azimuth;
      gestureStateRef.current.accumulatedElevation = camera.spherical.elevation;
      gestureStateRef.current.frameCount = 0;
      
      console.log('🎯 New gesture detected:', {
        translation: { x: translationX, y: translationY },
        camera: { 
          azimuth: camera.spherical.azimuth.toFixed(3), 
          elevation: camera.spherical.elevation.toFixed(3) 
        }
      });
      return;
    }
    
    // Apply incremental rotation
    const rotationSpeed = 0.008;
    const azimuthDelta = -deltaX * rotationSpeed;
    const elevationDelta = deltaY * rotationSpeed;
    
    // Update accumulated rotation
    gestureStateRef.current.accumulatedAzimuth += azimuthDelta;
    gestureStateRef.current.accumulatedElevation = Math.max(
      -Math.PI / 2 + 0.1,
      Math.min(
        Math.PI / 2 - 0.1,
        gestureStateRef.current.accumulatedElevation + elevationDelta
      )
    );
    
    // Apply to camera
    setCamera(prev => ({
      ...prev,
      spherical: {
        ...prev.spherical,
        azimuth: gestureStateRef.current.accumulatedAzimuth,
        elevation: gestureStateRef.current.accumulatedElevation,
      },
    }));
    
    // Update last position for next frame
    gestureStateRef.current.lastTranslationX = translationX;
    gestureStateRef.current.lastTranslationY = translationY;
    gestureStateRef.current.frameCount++;
  };

  // Handle pinch gestures for zoom - delta based (no acceleration)
  const onPinchGestureEvent = (event: any) => {
    // Don't allow zoom during thinking to prevent jumping
    if (isThinking) return;
    
    const { scale, state } = event.nativeEvent;
    
    if (state === State.BEGAN) {
      gestureStateRef.current.lastPinchScale = scale;
    }
    
    if (state === State.ACTIVE) {
      // Calculate scale delta since last frame
      const deltaScale = scale - gestureStateRef.current.lastPinchScale;
      gestureStateRef.current.lastPinchScale = scale;
      
      // Apply zoom delta directly to current values
      const zoomSpeed = 0.5;
      
      setCamera(prev => ({
        ...prev,
        zoom: Math.max(0.3, Math.min(5, prev.zoom + deltaScale * zoomSpeed)),
      }));
    }
  };

  // Generate 3D positions for nodes based on their semantic properties
  const nodePositions = useMemo(() => {
    const positions: Record<string, { x: number; y: number; z: number }> = {};
    
    // Define semantic clusters in 3D space
    const clusters: Record<string, { center: [number, number, number]; radius: number }> = {
      'conversation': { center: [0, 0, 0], radius: 100 },
      'technical': { center: [150, 50, -100], radius: 80 },
      'visualization': { center: [-120, 100, 80], radius: 70 },
      'frontend': { center: [80, -100, 120], radius: 80 },
      'memory_system': { center: [-60, 150, -100], radius: 90 },
      'vector_space': { center: [100, 80, -50], radius: 60 },
    };
    
    Object.entries(nodes).forEach(([nodeId, node]) => {
      // Find primary cluster based on tags
      let clusterKey = 'conversation';
      for (const tag of node.tags) {
        if (clusters[tag]) {
          clusterKey = tag;
          break;
        }
      }
      
      const cluster = clusters[clusterKey];
      
      // Generate position within cluster sphere
      const theta = Math.random() * Math.PI * 2; // Azimuth
      const phi = Math.acos(2 * Math.random() - 1); // Elevation
      const r = Math.random() * cluster.radius;
      
      const x = cluster.center[0] + r * Math.sin(phi) * Math.cos(theta);
      const y = cluster.center[1] + r * Math.cos(phi);
      const z = cluster.center[2] + r * Math.sin(phi) * Math.sin(theta);
      
      positions[nodeId] = { x, y, z };
    });
    
    return positions;
  }, [nodes]);


  // Smoothly transition camera to focus on selected node
  const animationRef = useRef<number>();
  
  useEffect(() => {
    // Don't animate camera during thinking to prevent jumping
    if (isThinking) return;
    
    if (selectedNode && nodePositions[selectedNode]) {
      // Cancel any existing animation
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      
      const targetPos = nodePositions[selectedNode];
      const startTarget = { ...camera.target };
      const startTime = Date.now();
      const duration = 600;
      
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // Ease out cubic
        
        setCamera(prev => ({
          ...prev,
          target: {
            x: startTarget.x + (targetPos.x - startTarget.x) * eased,
            y: startTarget.y + (targetPos.y - startTarget.y) * eased,
            z: startTarget.z + (targetPos.z - startTarget.z) * eased,
          },
        }));
        
        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          animationRef.current = undefined;
        }
      };
      
      animationRef.current = requestAnimationFrame(animate);
    }
    
    // Cleanup on unmount
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [selectedNode, nodePositions, isThinking]);

  // Auto-fit camera to show all nodes on initial load
  useEffect(() => {
    if (Object.keys(nodePositions).length === 0) return;
    // Don't auto-fit during thinking to prevent jumping
    if (isThinking) return;
    
    // Calculate bounding box of all nodes
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    Object.values(nodePositions).forEach(pos => {
      minX = Math.min(minX, pos.x);
      maxX = Math.max(maxX, pos.x);
      minY = Math.min(minY, pos.y);
      maxY = Math.max(maxY, pos.y);
      minZ = Math.min(minZ, pos.z);
      maxZ = Math.max(maxZ, pos.z);
    });
    
    // Calculate center and size of bounding box
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    
    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ);
    
    // Calculate distance needed to fit all nodes
    // Using field of view calculation
    const fov = Math.PI / 4; // 45 degrees
    const distance = (maxSize / 2) / Math.tan(fov / 2) * 1.5; // 1.5x for padding
    
    // Set initial camera to fit all nodes
    setCamera(prev => ({
      ...prev,
      target: { x: centerX, y: centerY, z: centerZ },
      spherical: {
        ...prev.spherical,
        distance: Math.max(200, Math.min(800, distance)),
      },
    }));
  }, [nodePositions, isThinking]);

  // Track recent memory accesses for visualization
  const recentAccessMap = useMemo(() => {
    const map: Record<string, { accessType: string; timestamp: number }> = {};
    const cutoffTime = Date.now() - 5000; // Show accesses from last 5 seconds
    
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

  // Determine node color based on state - star-like glow colors
  const getNodeColor = (node: MemoryNode, isRecentlyAccessed: boolean, accessType?: string) => {
    if (isRecentlyAccessed && accessType) {
      const colors = {
        'read': '#87CEEB',     // Sky blue glow
        'write': '#98FB98',    // Pale green glow
        'strengthen': '#FFD700', // Gold glow
        'traverse': '#DDA0DD',   // Plum glow
      };
      return colors[accessType] || '#FFA07A'; // Light salmon
    }
    return '#E6E6FA'; // Lavender for dim default state
  };

  // Get glow intensity based on node state
  const getGlowIntensity = (node: MemoryNode, isRecentlyAccessed: boolean, isSelected: boolean) => {
    if (isSelected) return 1.0; // Full glow when selected
    if (isRecentlyAccessed) return 0.8; // Bright glow when recently accessed
    return 0.3; // Dim glow for inactive nodes
  };

  // Calculate node size based on importance
  const getNodeSize = (node: MemoryNode, scale: number) => {
    const baseSize = 20;
    const importanceBonus = Math.log(node.access_count + 1) * 3;
    return Math.max(12, Math.min(40, (baseSize + importanceBonus) * scale));
  };

  // Project all nodes to screen space and prepare for rendering
  const projectedNodes = useMemo(() => {
    const projected: Array<{
      nodeId: string;
      node: MemoryNode;
      position: { x: number; y: number };
      scale: number;
      depth: number;
      isVisible: boolean;
      color: string;
      size: number;
      isSelected: boolean;
      isRecentlyAccessed: boolean;
      accessType?: string;
    }> = [];
    
    Object.entries(nodes).forEach(([nodeId, node]) => {
      const worldPos = nodePositions[nodeId];
      if (!worldPos) return;
      
      const screenPos = project3DToScreen(worldPos);
      if (!screenPos || !screenPos.isVisible) return;
      
      const recentAccess = recentAccessMap[nodeId];
      const isRecentlyAccessed = !!recentAccess;
      const isSelected = selectedNode === nodeId;
      const color = getNodeColor(node, isRecentlyAccessed, recentAccess?.accessType);
      const size = getNodeSize(node, screenPos.scale);
      const glowIntensity = getGlowIntensity(node, isRecentlyAccessed, isSelected);
      
      projected.push({
        nodeId,
        node,
        position: { x: screenPos.x, y: screenPos.y },
        scale: screenPos.scale,
        depth: screenPos.depth,
        isVisible: true,
        color,
        size,
        isSelected,
        isRecentlyAccessed,
        accessType: recentAccess?.accessType,
        glowIntensity,
      });
    });
    
    // Sort by depth for proper rendering order (far to near)
    return projected.sort((a, b) => b.depth - a.depth);
  }, [nodes, nodePositions, camera, selectedNode, recentAccessMap]);

  // Project connections between nodes
  const projectedConnections = useMemo(() => {
    const connections: Array<{
      id: string;
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      weight: number;
      color: string;
      depth: number;
      isVisible: boolean;
    }> = [];
    
    // Track processed connections to avoid duplicates
    const processed = new Set<string>();
    
    Object.entries(nodes).forEach(([nodeId, node]) => {
      const startPos = nodePositions[nodeId];
      if (!startPos) return;
      
      const startScreen = project3DToScreen(startPos);
      if (!startScreen) return;
      
      Object.entries(node.connections).forEach(([targetId, connection]) => {
        // Create unique connection ID
        const connectionId = nodeId < targetId ? `${nodeId}-${targetId}` : `${targetId}-${nodeId}`;
        if (processed.has(connectionId)) return;
        processed.add(connectionId);
        
        const endPos = nodePositions[targetId];
        if (!endPos) return;
        
        const endScreen = project3DToScreen(endPos);
        if (!endScreen) return;
        
        // Show connection if at least one endpoint is visible
        const isVisible = startScreen.isVisible || endScreen.isVisible;
        if (!isVisible) return;
        
        // If one node is off-screen, clamp its position to screen edge for line drawing
        let x1 = startScreen.x;
        let y1 = startScreen.y;
        let x2 = endScreen.x;
        let y2 = endScreen.y;
        
        // Clamp off-screen coordinates to viewport bounds for visual continuity
        if (!startScreen.isVisible) {
          x1 = Math.max(viewport.left, Math.min(viewport.right, x1));
          y1 = Math.max(viewport.top, Math.min(viewport.bottom, y1));
        }
        if (!endScreen.isVisible) {
          x2 = Math.max(viewport.left, Math.min(viewport.right, x2));
          y2 = Math.max(viewport.top, Math.min(viewport.bottom, y2));
        }
        
        const weight = connection.outbound_weight;
        // Blue glow connections with intensity based on weight
        const baseBlue = '#4A90E2';
        const glowBlue = '#87CEEB';
        const color = weight > 0.7 ? '#00BFFF' :  // Deep sky blue for strong connections
                     weight > 0.5 ? '#4169E1' :  // Royal blue for medium connections  
                     weight > 0.3 ? '#6495ED' :  // Cornflower blue for weak connections
                     '#483D8B';                   // Dark slate blue for very weak connections
        
        // Check if either node was recently accessed to boost connection glow
        const startRecentAccess = recentAccessMap[nodeId];
        const endRecentAccess = recentAccessMap[targetId];
        const hasRecentAccess = startRecentAccess || endRecentAccess;
        const baseGlowIntensity = Math.max(0.2, weight);
        const boostedGlow = hasRecentAccess ? Math.min(1, baseGlowIntensity * 1.5) : baseGlowIntensity;

        // Calculate node sizes to clip lines at node edges
        const startNode = nodes[nodeId];
        const endNode = nodes[targetId];
        const startNodeSize = startNode ? getNodeSize(startNode, startScreen.scale) : 20;
        const endNodeSize = endNode ? getNodeSize(endNode, endScreen.scale) : 20;
        
        // Calculate direction vector
        const dx = x2 - x1;
        const dy = y2 - y1;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Skip if nodes are too close
        if (distance < (startNodeSize + endNodeSize) / 2) {
          return;
        }
        
        // Normalize direction vector
        const dirX = dx / distance;
        const dirY = dy / distance;
        
        // Calculate clipped line endpoints (stop at node edges)
        const startRadius = startNodeSize / 2;
        const endRadius = endNodeSize / 2;
        
        const clippedX1 = x1 + dirX * startRadius;
        const clippedY1 = y1 + dirY * startRadius;
        const clippedX2 = x2 - dirX * endRadius;
        const clippedY2 = y2 - dirY * endRadius;

        connections.push({
          id: connectionId,
          x1: clippedX1,
          y1: clippedY1,
          x2: clippedX2,
          y2: clippedY2,
          weight,
          color,
          depth: (startScreen.depth + endScreen.depth) / 2,
          isVisible: true,
          glowIntensity: boostedGlow,
        });
      });
    });
    
    // Sort by depth for proper rendering
    return connections.sort((a, b) => b.depth - a.depth);
  }, [nodes, nodePositions, camera]);

  // Handle node selection
  const handleNodePress = (nodeId: string) => {
    // Toggle selection
    const newSelection = nodeId === selectedNode ? null : nodeId;
    selectNode(newSelection);
    onNodeSelect?.(nodeId);
  };

  return (
    <View style={[styles.container, { width, height }]}>
      <PanGestureHandler
        ref={panRef}
        onGestureEvent={onPanGestureEvent}
        simultaneousHandlers={pinchRef}
        minPointers={1}
        maxPointers={1}
      >
        <PinchGestureHandler
          ref={pinchRef}
          onGestureEvent={onPinchGestureEvent}
          simultaneousHandlers={panRef}
        >
          <View style={styles.visualizationArea}>
            {/* SVG with viewport clipping */}
            <Svg
              width={width}
              height={height}
              style={StyleSheet.absoluteFill}
            >
              {/* Define clipping path for viewport */}
              <ClipPath id="viewportClip">
                <Rect
                  x={viewport.left}
                  y={viewport.top}
                  width={viewport.width}
                  height={viewport.height}
                />
              </ClipPath>
              
              {/* Render connections within clipped area */}
              <G clipPath="url(#viewportClip)">
                {projectedConnections.map((connection) => {
                  const baseWidth = Math.max(1, connection.weight * 4);
                  const baseOpacity = connection.glowIntensity * 0.8;
                  
                  return (
                    <G key={connection.id}>
                      {/* Outer glow layer */}
                      <Line
                        x1={connection.x1}
                        y1={connection.y1}
                        x2={connection.x2}
                        y2={connection.y2}
                        stroke={connection.color}
                        strokeWidth={baseWidth * 3}
                        strokeOpacity={baseOpacity * 0.2}
                      />
                      {/* Middle glow layer */}
                      <Line
                        x1={connection.x1}
                        y1={connection.y1}
                        x2={connection.x2}
                        y2={connection.y2}
                        stroke={connection.color}
                        strokeWidth={baseWidth * 2}
                        strokeOpacity={baseOpacity * 0.4}
                      />
                      {/* Inner core line */}
                      <Line
                        x1={connection.x1}
                        y1={connection.y1}
                        x2={connection.x2}
                        y2={connection.y2}
                        stroke={connection.color}
                        strokeWidth={baseWidth}
                        strokeOpacity={baseOpacity}
                      />
                    </G>
                  );
                })}
              </G>
            </Svg>
            
            {/* Render nodes that are within viewport */}
            {projectedNodes.map((nodeData) => {
              const halfSize = nodeData.size / 2;
              
              // Additional bounds check for node visibility
              if (nodeData.position.x - halfSize > viewport.right ||
                  nodeData.position.x + halfSize < viewport.left ||
                  nodeData.position.y - halfSize > viewport.bottom ||
                  nodeData.position.y + halfSize < viewport.top) {
                return null;
              }
              
              return (
                <SimpleMemoryNode
                  key={nodeData.nodeId}
                  nodeData={nodeData}
                  onPress={() => handleNodePress(nodeData.nodeId)}
                />
              );
            })}
            
            {/* UI Controls */}
            <View style={styles.controls}>
              <Pressable style={styles.resetButton} onPress={resetCamera}>
                <Feather name="home" size={16} color="#FFFFFF" />
                <Text style={styles.resetButtonText}>Reset</Text>
              </Pressable>
            </View>
            
          </View>
        </PinchGestureHandler>
      </PanGestureHandler>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#000011',
    overflow: 'hidden',
  },
  visualizationArea: {
    flex: 1,
    position: 'relative',
  },
  viewportBorder: {
    position: 'absolute',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 4,
    backgroundColor: 'transparent',
  },
  controls: {
    position: 'absolute',
    top: 20,
    right: 20,
    gap: 10,
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 122, 255, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  resetButtonText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  memoryNode: {
    position: 'absolute',
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
    top: -8,
    right: -8,
    width: 16,
    height: 16,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  accessText: {
    color: '#000011',
    fontSize: 8,
    fontWeight: 'bold',
  },
  pulseRing: {
    position: 'absolute',
    borderWidth: 2,
    backgroundColor: 'transparent',
    opacity: 0.6,
    // Animation would be added here if supported
  },
});