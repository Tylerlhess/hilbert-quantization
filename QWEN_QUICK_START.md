# 🚀 Qwen Conversation Scripts - Quick Start Guide

I've created two versions of the Qwen conversation script to handle different system capabilities:

## 📋 **Choose Your Version**

### 🔥 **Full Version** (`qwen_conversation_script.py`)
- **Model**: Qwen3-Coder-30B-A3B-Instruct (30 billion parameters)
- **Memory**: Requires 8-16GB RAM
- **Performance**: Best quality responses
- **Encoding Time**: 10-30 minutes first run

### ⚡ **Lite Version** (`qwen_conversation_lite.py`) - **RECOMMENDED FOR MOST USERS**
- **Model**: Qwen2.5-Coder-7B-Instruct (7 billion parameters)
- **Memory**: Requires 2-4GB RAM
- **Performance**: Fast and efficient
- **Encoding Time**: 2-5 minutes first run

## 🎯 **Quick Start Commands**

### For Most Users (Lite Version):
```bash
python qwen_conversation_lite.py
```

### For High-Memory Systems (Full Version):
```bash
# Memory-optimized settings
python qwen_conversation_script.py --max-params 25000 --chunk-size 2500
```

### For Very Limited Memory:
```bash
# Ultra-conservative settings
python qwen_conversation_lite.py --max-params 10000 --chunk-size 1000
```

## 🛠️ **Troubleshooting the "Killed" Error**

If you got a "killed" error, it means the system ran out of memory. Here's how to fix it:

### 1. **Use the Lite Version** (Easiest Solution)
```bash
python qwen_conversation_lite.py
```

### 2. **Reduce Memory Usage** (For Full Version)
```bash
# Very conservative settings
python qwen_conversation_script.py --max-params 15000 --chunk-size 1500

# Or even smaller
python qwen_conversation_script.py --max-params 10000 --chunk-size 1000
```

### 3. **Check Available Memory**
```bash
# On macOS/Linux
free -h
# or
top

# Make sure you have at least 4GB free RAM
```

## 💡 **Memory Usage Guide**

| Script Version | RAM Needed | Parameters | Chunk Size | Encoding Time |
|---------------|------------|------------|------------|---------------|
| **Lite** (Recommended) | 2-4GB | 25,000 | 2,500 | 2-5 min |
| **Full** (Conservative) | 4-8GB | 25,000 | 2,500 | 5-15 min |
| **Full** (Standard) | 8-16GB | 50,000 | 5,000 | 10-30 min |

## 🎮 **Chat Interface**

Once running, you'll see:
```
🤖 Qwen Interactive Chat
==================================================
Ready to chat! Type your first message:

👤 You: Write a Python function to sort a list

🤖 Qwen: [Generated response with code example]

👤 You: /quit
```

### Available Commands:
- `/help` - Show commands
- `/quit` - Exit chat
- `/clear` - Clear history
- `/stats` - Show statistics

## 🔧 **Advanced Configuration**

### Custom Model Storage:
```bash
python qwen_conversation_lite.py --model-dir my_custom_storage
```

### Skip Re-encoding (if model already exists):
```bash
python qwen_conversation_lite.py --skip-download
```

### Use Different Model:
```bash
# Use an even smaller model
python qwen_conversation_lite.py --model-name "Qwen/Qwen2.5-Coder-1.5B-Instruct"
```

## 🚨 **If You Still Have Issues**

1. **Close other applications** to free up memory
2. **Restart your terminal** to clear any memory leaks
3. **Use the smallest settings**:
   ```bash
   python qwen_conversation_lite.py --max-params 5000 --chunk-size 500
   ```
4. **Check system resources** before running

## 📊 **Expected Performance**

### Lite Version (7B model):
- **First run**: 2-5 minutes encoding
- **Subsequent runs**: 5-10 seconds startup
- **Response time**: 1-3 seconds
- **Memory usage**: 2-4GB during encoding, 1-2GB during chat

### Full Version (30B model):
- **First run**: 10-30 minutes encoding
- **Subsequent runs**: 10-20 seconds startup  
- **Response time**: 2-5 seconds
- **Memory usage**: 8-16GB during encoding, 2-4GB during chat

## 🎯 **Recommended Workflow**

1. **Start with Lite**: `python qwen_conversation_lite.py`
2. **Test the chat interface** to make sure everything works
3. **If you want better responses** and have enough memory, try the full version
4. **Save conversations** using `/save` command for future reference

The Lite version should work on most modern systems and provides excellent conversation quality for coding assistance!
