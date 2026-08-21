---
title: YouTube RAG Chatbot
emoji: 🎥
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# 🎥 YouTube RAG Chatbot

An interactive RAG (Retrieval-Augmented Generation) application built with **Streamlit**, **LangChain**, **FAISS**, and **Groq LLM**. Chat with any YouTube video transcript and get precise answer citations with timestamped links!

## 🚀 Features
- 📜 **Automatic Captions & Transcript Fetching**
- 🧠 **Local Embeddings**: Powered by HuggingFace sentence-transformers/all-MiniLM-L6-v2
- ⚡ **Lightning Fast Generation**: Powered by Groq llama-3.3-70b-versatile / openai/gpt-oss-20b
- ⏱️ **Timestamp Citations**: Direct click-to-watch links targeting precise video timestamps
