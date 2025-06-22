import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { NavigationContainer } from '@react-navigation/native';
import { Feather } from '@expo/vector-icons';
import { ChatScreen } from '../screens/ChatScreen';
import { MemoryVisualizationScreen } from '../screens/MemoryVisualizationScreen';
import { useChatStore } from '../stores/chatStore';

const Tab = createBottomTabNavigator();

export const AppNavigator: React.FC = () => {
  const { memoryVisualizationVisible } = useChatStore();

  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName: keyof typeof Feather.glyphMap;

            if (route.name === 'Chat') {
              iconName = 'message-circle';
            } else if (route.name === 'Memory') {
              iconName = 'globe';
            } else {
              iconName = 'help-circle';
            }

            return <Feather name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#007AFF',
          tabBarInactiveTintColor: '#8E8E93',
          tabBarStyle: {
            backgroundColor: '#FFFFFF',
            borderTopColor: '#E5E5EA',
            paddingBottom: 8,
            paddingTop: 8,
            height: 60,
          },
          tabBarLabelStyle: {
            fontSize: 12,
            fontWeight: '500',
          },
          headerShown: false,
        })}
        initialRouteName="Chat"
      >
        <Tab.Screen 
          name="Chat" 
          component={ChatScreen}
          options={{
            tabBarBadge: memoryVisualizationVisible ? '●' : undefined,
            tabBarBadgeStyle: {
              backgroundColor: '#FF6B6B',
              color: '#FFFFFF',
              fontSize: 8,
              minWidth: 8,
              height: 8,
              borderRadius: 4,
              marginTop: -2,
              marginLeft: -2,
            },
          }}
        />
        <Tab.Screen 
          name="Memory" 
          component={MemoryVisualizationScreen}
          options={{
            tabBarLabel: 'Memory Graph',
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
};