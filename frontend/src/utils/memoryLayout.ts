import { MemoryNode } from '../types/memory';

/**
 * Force-directed layout algorithm for positioning memory nodes in 3D space
 * This creates a more natural clustering based on connection weights
 */
export class MemoryLayout {
  private nodes: Record<string, MemoryNode>;
  private positions: Record<string, [number, number, number]>;
  private velocities: Record<string, [number, number, number]>;
  
  constructor(nodes: Record<string, MemoryNode>) {
    this.nodes = nodes;
    this.positions = {};
    this.velocities = {};
    this.initializePositions();
  }

  private initializePositions() {
    const nodeIds = Object.keys(this.nodes);
    
    nodeIds.forEach((nodeId, index) => {
      // Initialize with random positions or existing positions
      if (this.nodes[nodeId].position_3d) {
        this.positions[nodeId] = [...this.nodes[nodeId].position_3d];
      } else {
        // Random initialization
        this.positions[nodeId] = [
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20,
        ];
      }
      
      this.velocities[nodeId] = [0, 0, 0];
    });
  }

  /**
   * Run force-directed simulation to improve node positioning
   */
  simulate(iterations: number = 100): Record<string, [number, number, number]> {
    const nodeIds = Object.keys(this.nodes);
    const repulsionStrength = 5;
    const attractionStrength = 0.1;
    const damping = 0.9;
    
    for (let iter = 0; iter < iterations; iter++) {
      const forces: Record<string, [number, number, number]> = {};
      
      // Initialize forces
      nodeIds.forEach(nodeId => {
        forces[nodeId] = [0, 0, 0];
      });
      
      // Calculate repulsion forces (nodes push each other away)
      for (let i = 0; i < nodeIds.length; i++) {
        for (let j = i + 1; j < nodeIds.length; j++) {
          const nodeA = nodeIds[i];
          const nodeB = nodeIds[j];
          const posA = this.positions[nodeA];
          const posB = this.positions[nodeB];
          
          const dx = posA[0] - posB[0];
          const dy = posA[1] - posB[1];
          const dz = posA[2] - posB[2];
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1; // Avoid division by zero
          
          const repulsionForce = repulsionStrength / (distance * distance);
          const fx = (dx / distance) * repulsionForce;
          const fy = (dy / distance) * repulsionForce;
          const fz = (dz / distance) * repulsionForce;
          
          forces[nodeA][0] += fx;
          forces[nodeA][1] += fy;
          forces[nodeA][2] += fz;
          forces[nodeB][0] -= fx;
          forces[nodeB][1] -= fy;
          forces[nodeB][2] -= fz;
        }
      }
      
      // Calculate attraction forces (connected nodes pull towards each other)
      nodeIds.forEach(nodeId => {
        const node = this.nodes[nodeId];
        const posA = this.positions[nodeId];
        
        Object.entries(node.connections).forEach(([targetId, connection]) => {
          if (this.positions[targetId]) {
            const posB = this.positions[targetId];
            const dx = posB[0] - posA[0];
            const dy = posB[1] - posA[1];
            const dz = posB[2] - posA[2];
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1;
            
            const attractionForce = attractionStrength * connection.outbound_weight * distance;
            const fx = (dx / distance) * attractionForce;
            const fy = (dy / distance) * attractionForce;
            const fz = (dz / distance) * attractionForce;
            
            forces[nodeId][0] += fx;
            forces[nodeId][1] += fy;
            forces[nodeId][2] += fz;
          }
        });
      });
      
      // Update velocities and positions
      nodeIds.forEach(nodeId => {
        const velocity = this.velocities[nodeId];
        const force = forces[nodeId];
        const position = this.positions[nodeId];
        
        // Update velocity
        velocity[0] = (velocity[0] + force[0]) * damping;
        velocity[1] = (velocity[1] + force[1]) * damping;
        velocity[2] = (velocity[2] + force[2]) * damping;
        
        // Update position
        position[0] += velocity[0];
        position[1] += velocity[1];
        position[2] += velocity[2];
      });
    }
    
    return { ...this.positions };
  }

  /**
   * Apply clustering based on semantic similarity
   */
  applySemanticClustering(): Record<string, [number, number, number]> {
    const nodeIds = Object.keys(this.nodes);
    
    // Group nodes by primary tag
    const clusters: Record<string, string[]> = {};
    nodeIds.forEach(nodeId => {
      const primaryTag = this.nodes[nodeId].tags[0] || 'default';
      if (!clusters[primaryTag]) {
        clusters[primaryTag] = [];
      }
      clusters[primaryTag].push(nodeId);
    });
    
    // Position clusters in 3D space
    const clusterKeys = Object.keys(clusters);
    const clusterPositions: Record<string, [number, number, number]> = {};
    
    clusterKeys.forEach((cluster, index) => {
      const angle = (index / clusterKeys.length) * Math.PI * 2;
      const radius = 10;
      clusterPositions[cluster] = [
        Math.cos(angle) * radius,
        (Math.random() - 0.5) * 8,
        Math.sin(angle) * radius,
      ];
    });
    
    // Position nodes within clusters
    Object.entries(clusters).forEach(([clusterName, nodeIds]) => {
      const clusterCenter = clusterPositions[clusterName];
      
      nodeIds.forEach((nodeId, index) => {
        const localAngle = (index / nodeIds.length) * Math.PI * 2;
        const localRadius = 2 + Math.random() * 2;
        
        this.positions[nodeId] = [
          clusterCenter[0] + Math.cos(localAngle) * localRadius,
          clusterCenter[1] + (Math.random() - 0.5) * 3,
          clusterCenter[2] + Math.sin(localAngle) * localRadius,
        ];
      });
    });
    
    return { ...this.positions };
  }
}

/**
 * Generate optimized 3D positions for memory nodes
 */
export function generateOptimizedLayout(nodes: Record<string, MemoryNode>): Record<string, [number, number, number]> {
  const layout = new MemoryLayout(nodes);
  
  // First apply semantic clustering
  layout.applySemanticClustering();
  
  // Then run force-directed simulation to refine positions
  return layout.simulate(50);
}