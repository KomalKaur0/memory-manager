# AI Memory System - Frontend

React Native/Expo frontend for the AI Memory System with 3D memory visualization.

## 💻 Quick Setup for Local Development

### Prerequisites
- Node.js and npm installed

### Setup Steps

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start Expo development server:**
   ```bash
   npm start
   ```

3. **Choose your platform:**
   - Press `w` for web browser (recommended for development)
   - Press `a` for Android emulator
   - Press `i` for iOS simulator (macOS only)

## 🏗️ Architecture

### Simple Local Development
- Frontend: React Native/Expo with mock data
- Backend: Python FastAPI (for future integration)
- Platform: Web browser, iOS simulator, or Android emulator

### Current Setup
```
Web Browser / Emulator
    ↓ 
Expo Dev Server (localhost:8081)
    ↓
Mock Data (no backend needed for UI development)
```

## 🎯 Features

### Chat Interface
- Claude-like message bubbles
- Real-time typing indicators
- Memory access visualization during AI thinking
- Responsive design for mobile

### 3D Memory Visualization
- Interactive memory nodes in 3D space
- Connection weights visualized as lines
- Real-time memory access highlighting
- Node clustering by semantic similarity
- Animated memory traversal

### Memory System
- Force-directed layout algorithm
- Semantic clustering
- Real-time memory access tracking
- Connection weight visualization
- Node importance scaling

## 📁 Project Structure

```
src/
├── components/
│   ├── Chat/                 # Chat interface components
│   ├── Memory3D/             # 3D visualization components
│   ├── MemoryOverlay/        # Overlay for memory visualization
│   └── Connection/           # Backend connection status
├── screens/                  # Main app screens
├── stores/                   # Zustand state management
├── services/                 # API services and WebSocket
├── hooks/                    # Custom React hooks
├── types/                    # TypeScript interfaces
├── config/                   # API configuration
└── utils/                    # Utility functions
```

## 🔧 Development

### Scripts
- `npm start` - Start Expo development server
- `npm run android` - Run on Android emulator
- `npm run ios` - Run on iOS simulator (macOS only)
- `npm run web` - Run in web browser

### Development
- `npm start` - Start Expo development server
- Choose web, Android, or iOS from the menu
- Hot reloading for instant updates

## 🌐 Backend Integration

The frontend connects to your Python backend via:
- REST API endpoints for memory operations
- WebSocket for real-time updates
- Local connection (no ngrok needed for backend)

### API Endpoints
- `GET /api/memory/nodes` - Load memory graph
- `POST /api/chat/send` - Send chat messages
- `WebSocket /ws` - Real-time memory updates

## 📱 Mobile Features

### Touch Controls
- Pinch to zoom in 3D view
- Rotate to explore memory space
- Tap nodes to view details
- Swipe to navigate between screens

### Responsive Design
- Optimized for phone screens
- Adaptive 3D visualization size
- Mobile-friendly chat interface
- Connection status indicators

## 🚀 Next Steps

1. **Test on Phone:**
   - Run `./setup-expo-tunnel.sh`
   - Scan QR code with Expo Go
   - Test chat and 3D visualization

2. **Backend Integration:**
   - Ensure backend API endpoints match frontend expectations
   - Test real-time memory access visualization
   - Verify WebSocket connections

3. **Production Deployment:**
   - Build standalone app with `expo build`
   - Deploy backend to cloud service
   - Update API endpoints for production

## 🔍 Troubleshooting

### Common Issues

**Expo Go can't connect:**
- Make sure ngrok tunnel is running
- Check if firewall is blocking connections
- Verify phone and computer are on same network (not required with ngrok)

**Backend connection fails:**
- Ensure backend is running on port 8000
- Check API endpoints in `src/config/api.ts`
- Verify CORS settings in backend

**3D visualization not working:**
- Check device WebGL support
- Try restarting Expo development server
- Clear Expo cache: `expo start -c`

## 📚 Dependencies

### Core
- React Native 0.79.4
- Expo ~53.0.12
- TypeScript ~5.8.3

### 3D Visualization
- Three.js ^0.177.0
- @react-three/fiber ^9.1.2
- @react-three/drei ^10.3.0

### State Management
- Zustand ^5.0.5

### Navigation
- @react-navigation/native ^7.1.13
- @react-navigation/bottom-tabs ^7.3.17

The frontend is ready for phone testing via Expo Go with ngrok tunneling!