import React, { useMemo, useRef, useState, useEffect } from 'react';
import { View, StyleSheet, Text, Dimensions, Pressable } from 'react-native';
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

  // Camera state with clear separation of concerns
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

  // Gesture state tracking with proper initialization
  const gestureStateRef = useRef({
    panStart: { x: 0, y: 0 },
    rotationStart: { azimuth: 0, elevation: 0 },
    targetStart: { x: 0, y: 0, z: 0 },
    zoomStart: 1,
    isPanning: false,
    isActive: false,
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

  // Handle pan gestures for rotation or panning
  const onPanGestureEvent = (event: any) => {
    const { translationX, translationY, numberOfPointers, state } = event.nativeEvent;
    
    if (numberOfPointers === 1) {
      // One finger = rotate camera around target
      if (state === State.BEGAN) {
        gestureStateRef.current.rotationStart = { 
          azimuth: camera.spherical.azimuth,
          elevation: camera.spherical.elevation 
        };
        gestureStateRef.current.isPanning = false;
        gestureStateRef.current.isActive = true;
      }
      
      if (state === State.ACTIVE) {
        const rotationSpeed = 0.005;
        const newAzimuth = gestureStateRef.current.rotationStart.azimuth - translationX * rotationSpeed;
        const newElevation = Math.max(
          -Math.PI / 2 + 0.1,
          Math.min(
            Math.PI / 2 - 0.1,
            gestureStateRef.current.rotationStart.elevation + translationY * rotationSpeed
          )
        );
        
        setCamera(prev => ({
          ...prev,
          spherical: {
            ...prev.spherical,
            azimuth: newAzimuth,
            elevation: newElevation,
          },
        }));
      }
      
      if (state === State.END || state === State.CANCELLED) {
        gestureStateRef.current.isActive = false;
      }
      
    } else if (numberOfPointers === 2) {
      // Two fingers = pan target position
      if (state === State.BEGAN) {
        gestureStateRef.current.targetStart = { 
          x: camera.target.x,
          y: camera.target.y,
          z: camera.target.z 
        };
        gestureStateRef.current.isPanning = true;
        gestureStateRef.current.isActive = true;
      }
      
      if (state === State.ACTIVE) {
        const { azimuth } = camera.spherical;
        
        // Get camera right and up vectors for panning
        const rightX = Math.cos(azimuth);
        const rightZ = -Math.sin(azimuth);
        
        const panSpeed = 0.5;
        const panX = -translationX * panSpeed * rightX;
        const panZ = -translationX * panSpeed * rightZ;
        const panY = translationY * panSpeed;
        
        setCamera(prev => ({
          ...prev,
          target: {
            x: gestureStateRef.current.targetStart.x + panX,
            y: gestureStateRef.current.targetStart.y + panY,
            z: gestureStateRef.current.targetStart.z + panZ,
          },
        }));
      }
      
      if (state === State.END || state === State.CANCELLED) {
        gestureStateRef.current.isPanning = false;
        gestureStateRef.current.isActive = false;
      }
    }
  };

  // Handle pinch gestures for zoom
  const onPinchGestureEvent = (event: any) => {
    const { scale, state } = event.nativeEvent;
    
    if (state === State.BEGAN) {
      gestureStateRef.current.zoomStart = camera.zoom;
    }
    
    if (state === State.ACTIVE) {
      const newZoom = gestureStateRef.current.zoomStart * scale;
      const clampedZoom = Math.max(0.3, Math.min(5, newZoom));
      
      setCamera(prev => ({
        ...prev,
        zoom: clampedZoom,
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
  }, [selectedNode, nodePositions]);

  // Auto-fit camera to show all nodes on initial load
  useEffect(() => {
    if (Object.keys(nodePositions).length === 0) return;
    
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
  }, [nodePositions]);

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

  // Determine node color based on state
  const getNodeColor = (node: MemoryNode, isRecentlyAccessed: boolean, accessType?: string) => {
    if (isRecentlyAccessed && accessType) {
      const colors = {
        'read': '#4A90E2',
        'write': '#7ED321',
        'strengthen': '#F5A623',
        'traverse': '#BD10E0',
      };
      return colors[accessType] || '#FF6B6B';
    }
    return '#8A8A8A'; // Default gray
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
      const color = getNodeColor(node, isRecentlyAccessed, recentAccess?.accessType);
      const size = getNodeSize(node, screenPos.scale);
      
      projected.push({
        nodeId,
        node,
        position: { x: screenPos.x, y: screenPos.y },
        scale: screenPos.scale,
        depth: screenPos.depth,
        isVisible: true,
        color,
        size,
        isSelected: selectedNode === nodeId,
        isRecentlyAccessed,
        accessType: recentAccess?.accessType,
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
        
        // Both endpoints must be visible
        const isVisible = startScreen.isVisible && endScreen.isVisible;
        if (!isVisible) return;
        
        const weight = connection.outbound_weight;
        const color = weight > 0.7 ? '#FF6B6B' :
                     weight > 0.5 ? '#FFA500' :
                     weight > 0.3 ? '#4A90E2' : '#666666';
        
        connections.push({
          id: connectionId,
          x1: startScreen.x,
          y1: startScreen.y,
          x2: endScreen.x,
          y2: endScreen.y,
          weight,
          color,
          depth: (startScreen.depth + endScreen.depth) / 2,
          isVisible: true,
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
        maxPointers={2}
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
                {projectedConnections.map((connection) => (
                  <Line
                    key={connection.id}
                    x1={connection.x1}
                    y1={connection.y1}
                    x2={connection.x2}
                    y2={connection.y2}
                    stroke={connection.color}
                    strokeWidth={Math.max(1, connection.weight * 3)}
                    strokeOpacity={connection.weight * 0.6}
                  />
                ))}
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
                <Pressable
                  key={nodeData.nodeId}
                  style={[
                    styles.memoryNode,
                    {
                      left: nodeData.position.x - halfSize,
                      top: nodeData.position.y - halfSize,
                      width: nodeData.size,
                      height: nodeData.size,
                      borderRadius: halfSize,
                      backgroundColor: nodeData.color,
                      borderColor: nodeData.isSelected ? '#FFD700' : nodeData.color,
                      borderWidth: nodeData.isSelected ? 3 : 1,
                      opacity: Math.min(1, nodeData.scale * 0.8 + 0.2),
                      transform: [
                        { scale: nodeData.isSelected ? 1.1 : 1 },
                      ],
                    },
                  ]}
                  onPress={() => handleNodePress(nodeData.nodeId)}
                >
                  {/* Access type indicator */}
                  {nodeData.isRecentlyAccessed && nodeData.accessType && (
                    <View style={styles.accessIndicator}>
                      <Text style={styles.accessText}>
                        {nodeData.accessType.charAt(0).toUpperCase()}
                      </Text>
                    </View>
                  )}
                </Pressable>
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
    backgroundColor: '#FF4444',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#000011',
  },
  accessText: {
    color: '#FFFFFF',
    fontSize: 8,
    fontWeight: 'bold',
  },
});