import React, { useRef, useMemo } from 'react';
import { useFrame, ThreeEvent } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { MemoryNode, MemoryAccessEvent } from '../../types/memory';

interface MemoryNode3DProps {
  node: MemoryNode;
  position: [number, number, number];
  isSelected: boolean;
  isRecentlyAccessed: boolean;
  accessType?: MemoryAccessEvent['access_type'];
  onNodeClick: (nodeId: string) => void;
}

export const MemoryNode3D: React.FC<MemoryNode3DProps> = ({
  node,
  position,
  isSelected,
  isRecentlyAccessed,
  accessType,
  onNodeClick,
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef(0);

  // Calculate node size based on access count and importance
  const nodeSize = useMemo(() => {
    const baseSize = 0.3;
    const accessMultiplier = Math.log(node.access_count + 1) * 0.1;
    return Math.min(baseSize + accessMultiplier, 1.0);
  }, [node.access_count]);

  // Get node color based on type and state
  const nodeColor = useMemo(() => {
    if (isSelected) return '#FFD700'; // Gold for selected
    if (isRecentlyAccessed) {
      switch (accessType) {
        case 'read': return '#4A90E2';
        case 'write': return '#7ED321';
        case 'strengthen': return '#F5A623';
        case 'traverse': return '#BD10E0';
        default: return '#FF6B6B';
      }
    }
    
    // Default color based on memory type (using tags)
    if (node.tags.includes('conversation')) return '#87CEEB';
    if (node.tags.includes('technical')) return '#98FB98';
    if (node.tags.includes('visualization')) return '#DDA0DD';
    if (node.tags.includes('frontend')) return '#F0E68C';
    
    return '#B0B0B0'; // Default gray
  }, [isSelected, isRecentlyAccessed, accessType, node.tags]);

  // Animation for pulsing effect when recently accessed
  useFrame((state) => {
    if (meshRef.current) {
      if (isRecentlyAccessed) {
        pulseRef.current += 0.1;
        const scale = 1 + Math.sin(pulseRef.current) * 0.2;
        meshRef.current.scale.setScalar(scale);
      } else {
        meshRef.current.scale.setScalar(1);
        pulseRef.current = 0;
      }

      // Add subtle floating animation
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime + position[0]) * 0.05;
    }
  });

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    onNodeClick(node.id);
  };

  const handlePointerOver = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    // Skip cursor change for React Native
  };

  const handlePointerOut = () => {
    // Skip cursor change for React Native
  };

  return (
    <group position={position}>
      {/* Main node sphere */}
      <mesh
        ref={meshRef}
        onClick={handleClick}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        scale={nodeSize}
      >
        <sphereGeometry args={[1, 32, 32]} />
        <meshPhongMaterial
          color={nodeColor}
          transparent
          opacity={isSelected ? 1.0 : 0.8}
          emissive={isRecentlyAccessed ? nodeColor : '#000000'}
          emissiveIntensity={isRecentlyAccessed ? 0.2 : 0}
        />
      </mesh>

      {/* Node label */}
      <Text
        position={[0, nodeSize + 0.5, 0]}
        fontSize={0.3}
        color="#333"
        anchorX="center"
        anchorY="middle"
        maxWidth={3}
        textAlign="center"
      >
        {node.summary.length > 30 ? `${node.summary.substring(0, 30)}...` : node.summary}
      </Text>

      {/* Selection ring */}
      {isSelected && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[nodeSize * 1.5, nodeSize * 1.7, 32]} />
          <meshBasicMaterial color="#FFD700" transparent opacity={0.6} />
        </mesh>
      )}

      {/* Recent access indicator */}
      {isRecentlyAccessed && (
        <mesh>
          <sphereGeometry args={[nodeSize * 1.2, 16, 16]} />
          <meshBasicMaterial
            color={nodeColor}
            transparent
            opacity={0.3}
            wireframe
          />
        </mesh>
      )}
    </group>
  );
};