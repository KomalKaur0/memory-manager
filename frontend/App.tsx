import React, { useEffect } from 'react';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { AppNavigator } from './src/navigation/AppNavigator';
import { useMemoryStore } from './src/stores/memoryStore';

export default function App() {
  const { generateMockData, nodes } = useMemoryStore();

  // Generate mock data on app start
  useEffect(() => {
    if (Object.keys(nodes).length === 0) {
      generateMockData();
    }
  }, [generateMockData, nodes]);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <AppNavigator />
    </GestureHandlerRootView>
  );
}
