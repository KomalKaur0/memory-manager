import React, { useState } from 'react';
import { View, TextInput, Pressable, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native';
import { Feather } from '@expo/vector-icons';

interface ChatInputProps {
  value: string;
  onChangeText: (text: string) => void;
  onSend: () => void;
  disabled?: boolean;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChangeText,
  onSend,
  disabled = false,
  placeholder = "Ask about the memory system..."
}) => {
  const [inputHeight, setInputHeight] = useState(40);

  const handleSend = () => {
    if (value.trim() && !disabled) {
      onSend();
    }
  };

  const handleContentSizeChange = (event: any) => {
    const height = Math.max(40, Math.min(120, event.nativeEvent.contentSize.height));
    setInputHeight(height);
  };

  return (
    <KeyboardAvoidingView 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={90}
    >
      <View style={styles.container}>
        <View style={styles.inputContainer}>
          <TextInput
            style={[styles.textInput, { height: inputHeight }]}
            value={value}
            onChangeText={onChangeText}
            placeholder={placeholder}
            placeholderTextColor="#999"
            multiline
            onContentSizeChange={handleContentSizeChange}
            editable={!disabled}
            returnKeyType="send"
            onSubmitEditing={handleSend}
            blurOnSubmit={false}
          />
          <Pressable
            style={[
              styles.sendButton,
              { opacity: (value.trim() && !disabled) ? 1 : 0.5 }
            ]}
            onPress={handleSend}
            disabled={!value.trim() || disabled}
          >
            <Feather name="send" size={20} color="#007AFF" />
          </Pressable>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#F2F2F7',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    minHeight: 40,
  },
  textInput: {
    flex: 1,
    fontSize: 16,
    color: '#000',
    maxHeight: 120,
    paddingVertical: 8,
    textAlignVertical: 'center',
  },
  sendButton: {
    marginLeft: 12,
    padding: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
});