import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Canvas } from '@react-three/fiber';

export const SimpleTest3D: React.FC = () => {
  return (
    <View style={styles.container}>
      <Canvas style={styles.canvas}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <mesh>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="orange" />
        </mesh>
      </Canvas>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  canvas: {
    flex: 1,
  },
});