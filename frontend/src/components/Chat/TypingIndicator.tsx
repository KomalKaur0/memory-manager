import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';

interface TypingIndicatorProps {
  visible: boolean;
}

export const TypingIndicator: React.FC<TypingIndicatorProps> = ({ visible }) => {
  const dot1Opacity = useRef(new Animated.Value(0.3)).current;
  const dot2Opacity = useRef(new Animated.Value(0.3)).current;
  const dot3Opacity = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    if (visible) {
      const animateDots = () => {
        const duration = 600;
        const delay = 200;

        Animated.sequence([
          Animated.timing(dot1Opacity, { toValue: 1, duration, useNativeDriver: true }),
          Animated.timing(dot1Opacity, { toValue: 0.3, duration, useNativeDriver: true }),
        ]).start();

        setTimeout(() => {
          Animated.sequence([
            Animated.timing(dot2Opacity, { toValue: 1, duration, useNativeDriver: true }),
            Animated.timing(dot2Opacity, { toValue: 0.3, duration, useNativeDriver: true }),
          ]).start();
        }, delay);

        setTimeout(() => {
          Animated.sequence([
            Animated.timing(dot3Opacity, { toValue: 1, duration, useNativeDriver: true }),
            Animated.timing(dot3Opacity, { toValue: 0.3, duration, useNativeDriver: true }),
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

  return (
    <View style={styles.container}>
      <View style={styles.bubble}>
        <Text style={styles.label}>AI is thinking</Text>
        <View style={styles.dotsContainer}>
          <Animated.View style={[styles.dot, { opacity: dot1Opacity }]} />
          <Animated.View style={[styles.dot, { opacity: dot2Opacity }]} />
          <Animated.View style={[styles.dot, { opacity: dot3Opacity }]} />
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