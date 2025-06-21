import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { MemoryConnection } from '../../types/memory';

interface MemoryConnection3DProps {
  startPosition: [number, number, number];
  endPosition: [number, number, number];
  connection: MemoryConnection;
  isActive: boolean;
  connectionId: string;
}

export const MemoryConnection3D: React.FC<MemoryConnection3DProps> = ({
  startPosition,
  endPosition,
  connection,
  isActive,
  connectionId,
}) => {
  const lineRef = useRef<THREE.Line>(null);
  const flowRef = useRef(0);

  // Create curved line geometry
  const { curve, points } = useMemo(() => {
    const start = new THREE.Vector3(...startPosition);
    const end = new THREE.Vector3(...endPosition);
    
    // Create a slight curve by adding control points
    const distance = start.distanceTo(end);
    const midPoint = new THREE.Vector3().lerpVectors(start, end, 0.5);
    
    // Add some height to the curve for visual appeal
    const curveHeight = Math.min(distance * 0.3, 2);
    midPoint.y += curveHeight;
    
    // Create quadratic curve
    const curve = new THREE.QuadraticBezierCurve3(start, midPoint, end);
    const points = curve.getPoints(50);
    
    return { curve, points };
  }, [startPosition, endPosition]);

  // Calculate line properties based on connection weight
  const lineWidth = useMemo(() => {
    return Math.max(connection.outbound_weight * 0.1, 0.02);
  }, [connection.outbound_weight]);

  const lineColor = useMemo(() => {
    const weight = connection.outbound_weight;
    if (isActive) return '#FFD700'; // Gold for active connections
    
    // Color based on connection strength
    if (weight > 0.8) return '#FF4444'; // Strong - Red
    if (weight > 0.6) return '#FF8844'; // Medium-Strong - Orange
    if (weight > 0.4) return '#FFBB44'; // Medium - Yellow
    if (weight > 0.2) return '#44BBFF'; // Weak - Blue
    return '#8888BB'; // Very weak - Purple
  }, [connection.outbound_weight, isActive]);

  const lineOpacity = useMemo(() => {
    return isActive ? 1.0 : Math.max(connection.outbound_weight * 0.8, 0.3);
  }, [connection.outbound_weight, isActive]);

  // Animation for active connections
  useFrame((state) => {
    if (isActive && lineRef.current) {
      flowRef.current += 0.05;
      // Add pulsing effect for active connections
      const pulse = 1 + Math.sin(flowRef.current) * 0.3;
      lineRef.current.scale.setScalar(pulse);
    } else if (lineRef.current) {
      lineRef.current.scale.setScalar(1);
    }
  });

  return (
    <group>
      {/* Main connection line */}
      <line ref={lineRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={points.length}
            array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={lineColor}
          transparent
          opacity={lineOpacity}
          linewidth={lineWidth}
        />
      </line>

      {/* Flow particles for active connections */}
      {isActive && (
        <FlowParticles
          curve={curve}
          color={lineColor}
          speed={0.02}
          particleCount={3}
        />
      )}

      {/* Connection strength indicator */}
      {connection.outbound_weight > 0.7 && (
        <ConnectionStrengthIndicator
          position={[
            (startPosition[0] + endPosition[0]) / 2,
            (startPosition[1] + endPosition[1]) / 2 + 0.5,
            (startPosition[2] + endPosition[2]) / 2,
          ]}
          strength={connection.outbound_weight}
        />
      )}
    </group>
  );
};

// Component for animated flow particles along the connection
const FlowParticles: React.FC<{
  curve: THREE.QuadraticBezierCurve3;
  color: string;
  speed: number;
  particleCount: number;
}> = ({ curve, color, speed, particleCount }) => {
  const particlesRef = useRef<THREE.Group>(null);
  const particleOffsets = useRef<number[]>(
    Array.from({ length: particleCount }, (_, i) => i / particleCount)
  );

  useFrame(() => {
    if (particlesRef.current) {
      particleOffsets.current = particleOffsets.current.map(offset => {
        const newOffset = (offset + speed) % 1;
        return newOffset;
      });

      particlesRef.current.children.forEach((particle, index) => {
        const position = curve.getPoint(particleOffsets.current[index]);
        particle.position.copy(position);
      });
    }
  });

  return (
    <group ref={particlesRef}>
      {Array.from({ length: particleCount }).map((_, index) => (
        <mesh key={index}>
          <sphereGeometry args={[0.05, 8, 8]} />
          <meshBasicMaterial color={color} />
        </mesh>
      ))}
    </group>
  );
};

// Component for connection strength indicators
const ConnectionStrengthIndicator: React.FC<{
  position: [number, number, number];
  strength: number;
}> = ({ position, strength }) => {
  return (
    <mesh position={position}>
      <cylinderGeometry args={[0.02, 0.02, strength * 0.5, 8]} />
      <meshBasicMaterial color="#FFD700" transparent opacity={0.7} />
    </mesh>
  );
};