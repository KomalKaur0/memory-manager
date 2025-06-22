#!/usr/bin/env node

/**
 * Test script to verify frontend environment variable configuration
 */

const path = require('path');
const fs = require('fs');

function testEnvConfig() {
    console.log('🧪 Testing Frontend Environment Configuration');
    console.log('============================================');
    
    const frontendDir = path.join(__dirname, 'frontend');
    const envFile = path.join(frontendDir, '.env');
    
    // Check if .env file exists
    if (!fs.existsSync(envFile)) {
        console.log('❌ No frontend/.env file found');
        console.log('   Run: ./update_tunnel_url.sh <your-url>');
        return false;
    }
    
    console.log('✅ frontend/.env file found');
    
    // Read and parse .env file
    const envContent = fs.readFileSync(envFile, 'utf8');
    const envVars = {};
    
    envContent.split('\n').forEach(line => {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith('#')) {
            const [key, ...valueParts] = trimmed.split('=');
            envVars[key] = valueParts.join('=');
        }
    });
    
    console.log('\n📋 Environment Variables:');
    console.log('-------------------------');
    
    const importantVars = [
        'EXPO_PUBLIC_API_BASE_URL',
        'EXPO_PUBLIC_LOCAL_URL', 
        'TUNNEL_URL',
        'EXPO_PUBLIC_DEBUG_MODE',
        'EXPO_PUBLIC_FALLBACK_URLS'
    ];
    
    let allGood = true;
    
    importantVars.forEach(varName => {
        const value = envVars[varName];
        if (value) {
            console.log(`✅ ${varName}: ${value}`);
            
            // Test URL format for URL variables
            if (varName.includes('URL') && !varName.includes('FALLBACK')) {
                try {
                    new URL(value);
                    console.log(`   ✅ Valid URL format`);
                } catch (e) {
                    console.log(`   ❌ Invalid URL format: ${e.message}`);
                    allGood = false;
                }
            }
        } else {
            console.log(`⚠️  ${varName}: Not set`);
            if (varName === 'EXPO_PUBLIC_API_BASE_URL') {
                allGood = false;
            }
        }
    });
    
    // Test fallback URLs
    if (envVars.EXPO_PUBLIC_FALLBACK_URLS) {
        console.log('\n🔄 Testing Fallback URLs:');
        const fallbackUrls = envVars.EXPO_PUBLIC_FALLBACK_URLS.split(',');
        fallbackUrls.forEach((url, index) => {
            const trimmedUrl = url.trim();
            try {
                new URL(trimmedUrl);
                console.log(`   ${index + 1}. ✅ ${trimmedUrl}`);
            } catch (e) {
                console.log(`   ${index + 1}. ❌ ${trimmedUrl} - Invalid format`);
            }
        });
    }
    
    console.log('\n🎯 Recommendations:');
    console.log('-------------------');
    
    if (allGood) {
        console.log('✅ Configuration looks good!');
        console.log('   Frontend should be able to connect to backend');
        console.log('   Start frontend with: cd frontend && npm start');
    } else {
        console.log('⚠️  Configuration issues detected:');
        if (!envVars.EXPO_PUBLIC_API_BASE_URL) {
            console.log('   - Set EXPO_PUBLIC_API_BASE_URL');
            console.log('   - Run: ./update_tunnel_url.sh <your-tunnel-url>');
        }
    }
    
    // Test primary URL if it's set
    if (envVars.EXPO_PUBLIC_API_BASE_URL) {
        console.log(`\n🧪 Test the primary URL manually:`);
        console.log(`   curl ${envVars.EXPO_PUBLIC_API_BASE_URL}/health`);
    }
    
    return allGood;
}

if (require.main === module) {
    testEnvConfig();
}

module.exports = { testEnvConfig };