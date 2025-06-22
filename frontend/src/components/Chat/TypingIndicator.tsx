import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { useMemoryStore } from '../../stores/memoryStore';

interface TypingIndicatorProps {
  visible: boolean;
}

export const TypingIndicator: React.FC<TypingIndicatorProps> = ({ visible }) => {
  const dot1Opacity = useRef(new Animated.Value(0.3)).current;
  const dot2Opacity = useRef(new Animated.Value(0.3)).current;
  const dot3Opacity = useRef(new Animated.Value(0.3)).current;
  
  const { recentAccesses, isThinking } = useMemoryStore();
  const [lastAccessType, setLastAccessType] = useState<string | null>(null);
  const colorAnimValue = useRef(new Animated.Value(0)).current;

  // Track most recent memory access for color indication
  useEffect(() => {
    if (recentAccesses.length > 0) {
      const latestAccess = recentAccesses[0];
      const timeSinceAccess = Date.now() - latestAccess.timestamp;
      
      if (timeSinceAccess < 2000) { // Show color for 2 seconds after access
        setLastAccessType(latestAccess.access_type);
        
        // Stop any existing animations first
        colorAnimValue.stopAnimation();
        
        // Animate color change - quick bright-up then settle
        Animated.sequence([
          Animated.timing(colorAnimValue, {
            toValue: 1,
            duration: 200,
            useNativeDriver: false,
          }),
          Animated.timing(colorAnimValue, {
            toValue: 0.8,
            duration: 500,
            useNativeDriver: false,
          }),
        ]).start();
        
        // Clear after delay with fade-out
        const timeout = setTimeout(() => {
          colorAnimValue.stopAnimation();
          Animated.timing(colorAnimValue, {
            toValue: 0,
            duration: 800,
            useNativeDriver: false,
          }).start(() => {
            setLastAccessType(null);
          });
        }, 1500);
        
        return () => clearTimeout(timeout);
      }
    }
  }, [recentAccesses]);

  // Get dot color based on recent memory access
  const getDotColor = () => {
    if (!lastAccessType) return '#007AFF';
    
    const accessColors = {
      'read': '#87CEEB',     // Sky blue glow
      'write': '#98FB98',    // Pale green glow  
      'strengthen': '#FFD700', // Gold glow
      'traverse': '#DDA0DD',   // Plum glow
    };
    
    return accessColors[lastAccessType] || '#007AFF';
  };

  useEffect(() => {
    if (visible) {
      const animateDots = () => {
        const duration = 600;
        const delay = 200;

        Animated.sequence([
          Animated.timing(dot1Opacity, { toValue: 1, duration, useNativeDriver: false }),
          Animated.timing(dot1Opacity, { toValue: 0.3, duration, useNativeDriver: false }),
        ]).start();

        setTimeout(() => {
          Animated.sequence([
            Animated.timing(dot2Opacity, { toValue: 1, duration, useNativeDriver: false }),
            Animated.timing(dot2Opacity, { toValue: 0.3, duration, useNativeDriver: false }),
          ]).start();
        }, delay);

        setTimeout(() => {
          Animated.sequence([
            Animated.timing(dot3Opacity, { toValue: 1, duration, useNativeDriver: false }),
            Animated.timing(dot3Opacity, { toValue: 0.3, duration, useNativeDriver: false }),
          ]).start();
        }, delay * 2);
      };

      const interval = setInterval(animateDots, 1800);
      animateDots();

      return () => clearInterval(interval);
    } else {
      // Reset opacity when not visible
      dot1Opacity.setValue(0.3);
      dot2Opacity.setValue(0.3);
      dot3Opacity.setValue(0.3);
    }
  }, [visible, dot1Opacity, dot2Opacity, dot3Opacity]);

  if (!visible) return null;

  const baseDotColor = getDotColor();
  const isAccessingMemory = lastAccessType !== null;

  // Create interpolated color value for smooth transitions
  const animatedShadowOpacity = colorAnimValue.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 0.8],
  });

  const animatedShadowRadius = colorAnimValue.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 6],
  });

  return (
    <View style={styles.container}>
      <View style={styles.bubble}>
        <Text style={styles.label}>
          AI is thinking
        </Text>
        <View style={styles.dotsContainer}>
          <Animated.View style={[
            styles.dot, 
            { 
              opacity: dot1Opacity,
              backgroundColor: baseDotColor,
              shadowColor: baseDotColor,
              shadowOpacity: animatedShadowOpacity,
              shadowRadius: animatedShadowRadius,
              shadowOffset: { width: 0, height: 0 },
            }
          ]} />
          <Animated.View style={[
            styles.dot, 
            { 
              opacity: dot2Opacity,
              backgroundColor: baseDotColor,
              shadowColor: baseDotColor,
              shadowOpacity: animatedShadowOpacity,
              shadowRadius: animatedShadowRadius,
              shadowOffset: { width: 0, height: 0 },
            }
          ]} />
          <Animated.View style={[
            styles.dot, 
            { 
              opacity: dot3Opacity,
              backgroundColor: baseDotColor,
              shadowColor: baseDotColor,
              shadowOpacity: animatedShadowOpacity,
              shadowRadius: animatedShadowRadius,
              shadowOffset: { width: 0, height: 0 },
            }
          ]} />
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
    marginHorizontal: 16,
    alignItems: 'flex-start',
  },
  bubble: {
    backgroundColor: '#F2F2F2',
    borderRadius: 20,
    borderBottomLeftRadius: 6,
    paddingHorizontal: 16,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  label: {
    fontSize: 14,
    color: '#666',
    marginRight: 12,
    fontStyle: 'italic',
  },
  dotsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#007AFF',
    marginHorizontal: 2,
  },
});