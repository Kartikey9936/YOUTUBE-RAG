import streamlit as st
import streamlit.components.v1 as components
import os
import urllib.request
import json
from dotenv import load_dotenv

# Add workspace directory to path if running from python path directly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

# Load Environment variables
load_dotenv()

# Import pipeline modules
from src.youtube_loader import load_transcript, extract_video_id
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore, load_vectorstore, vectorstore_exists
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain

# --- Page Layout & Theme Configuration ---
st.set_page_config(
    page_title="YouTube ChatBot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple clean styling — lets Streamlit's native theme control colours
st.markdown("""
<style>
    /* Citation cards — bordered box with visible text */
    .citation-card {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .citation-time {
        font-weight: 600;
        color: #dc2626;
        font-size: 0.85rem;
        margin-bottom: 6px;
    }

    .citation-text {
        font-size: 0.9rem;
        line-height: 1.5;
        font-style: italic;
        margin-bottom: 6px;
    }

    .citation-link {
        font-size: 0.85rem;
        color: #2563eb !important;
        text-decoration: underline !important;
        font-weight: 500;
    }

    /* Video info box in sidebar */
    .video-info-box {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 12px;
        margin-top: 12px;
    }

    .video-info-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 4px;
        line-height: 1.4;
    }

    .video-info-author {
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to fetch YouTube Video Details via oEmbed
def get_youtube_metadata(video_id: str) -> dict:
    url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return {
                "title": data.get("title", "YouTube Video"),
                "author": data.get("author_name", "Unknown Creator"),
                "thumbnail_url": data.get("thumbnail_url", "")
            }
    except Exception:
        return {
            "title": "YouTube Video",
            "author": "Unknown Creator",
            "thumbnail_url": ""
        }

# --- Cache Resource Intensive Components ---
@st.cache_resource(show_spinner="Initializing embedding model (takes a moment on first run)...")
def get_cached_embeddings():
    return get_embeddings()

# --- Initialize Session State ---
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = get_cached_embeddings()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "video_id" not in st.session_state:
    st.session_state["video_id"] = None

if "video_metadata" not in st.session_state:
    st.session_state["video_metadata"] = None

if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None

# --- SIDEBAR: Video Processing and Options ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png", width=64)
    st.markdown("### YouTube Chatbot")
    st.markdown("Chat with Youtube Video")
    
    st.divider()
    
    youtube_url_input = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    process_button = st.button("Process Video", type="primary", use_container_width=True)
    
    if process_button and youtube_url_input:
        try:
            # Extract video id
            vid = extract_video_id(youtube_url_input)
            
            # Fetch metadata
            with st.spinner("Fetching video metadata..."):
                metadata = get_youtube_metadata(vid)
            
            st.session_state["video_id"] = vid
            st.session_state["video_metadata"] = metadata
            
            # Load or build index
            if vectorstore_exists(vid):
                with st.spinner("Found pre-existing index. Loading vector database..."):
                    db = load_vectorstore(st.session_state["embeddings"], vid)
                    retriever = get_retriever(db)
                    st.session_state["rag_chain"] = create_rag_chain(retriever)
                st.success("Loaded database from disk in 0s!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Fetching transcript
                status_text.text("1/3 Fetching transcript from YouTube...")
                progress_bar.progress(15)
                transcript_list, _ = load_transcript(vid)
                
                # Splitting
                status_text.text("2/3 Document splitting and timestamp mapping...")
                progress_bar.progress(50)
                docs = split_documents(transcript_list, vid)
                
                # Embedding and Indexing
                status_text.text("3/3 Generating embeddings and storing FAISS database...")
                progress_bar.progress(80)
                db = create_vectorstore(docs, st.session_state["embeddings"], vid)
                
                retriever = get_retriever(db)
                st.session_state["rag_chain"] = create_rag_chain(retriever)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                st.success("Successfully processed video and created database index!")
                
            # Clear chat history on video change
            st.session_state["chat_history"] = []
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

    # Show active video metadata in sidebar
    if st.session_state["video_metadata"] and st.session_state["video_id"]:
        st.divider()
        st.markdown("**Active Video:**")
        meta = st.session_state["video_metadata"]
        st.markdown(f"""
        <div class="video-info-box">
            <img src="{meta['thumbnail_url']}" style="width: 100%; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);"/>
            <div class="video-info-title">{meta['title']}</div>
            <div class="video-info-author">👤 {meta['author']}</div>
            <a href="https://youtu.be/{st.session_state['video_id']}" target="_blank" style="font-size: 0.85rem; color: #ff4b4b; font-weight: 600; text-decoration: none;">📺 View on YouTube</a>
        </div>
        """, unsafe_allow_html=True)
        
        # Reset button
        if st.button("Reset Session", use_container_width=True):
            st.session_state["video_id"] = None
            st.session_state["video_metadata"] = None
            st.session_state["rag_chain"] = None
            st.session_state["chat_history"] = []
            st.rerun()

# --- MAIN PANEL: Chat UI ---
st.title("YouTube Chatbot")
st.header("write any query in below promptbox:")

if not st.session_state["rag_chain"]:
    # Welcome & guidance screen
    st.info("👈 Please enter a YouTube video URL in the sidebar and click **Process Video** to start the chat.")
    # st.markdown("""
    # ### How it works:
    # 1. **Extracts Transcript**: Downloads captions/transcripts directly from YouTube.
    # 2. **Smart Splitting**: Segments the transcript using smart overlapping chunks to keep conversational context.
    # 3. **Semantic Mapping**: Tags each text segment with its exact starting and ending timestamp.
    # 4. **Vector Database**: Indexes the segments using a local SentenceTransformer model (`all-MiniLM-L6-v2`) in a local `FAISS` store.
    # 5. **Geared QA**: Uses LangChain Expression Language (LCEL) and Groq LLM (`openai/gpt-oss-20b`) to answer your questions with precise citations.
    
    # *Try adding any technical tutorial, podcasts, interviews, or news analysis video!*
    # """)
else:
    # Display Chat History
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["sources"]:
                with st.expander("📚 Sources & Timestamp Citations"):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span class="citation-time">⏱️ Timestamp: {src['start_time_str']} - {src['end_time_str']}</span>
                            </div>
                            <div class="citation-text">"{src['text']}"</div>
                            <a class="citation-link" href="{src['url']}" target="_blank">🔗 Watch video at {src['start_time_str']}</a>
                        </div>
                        """, unsafe_allow_html=True)

    # Animated sliding placeholder — uses components.html() so JS actually executes.
    # window.parent.document reaches the chat textarea in the parent Streamlit page.
    components.html("""
    <script>
    (function() {
        const examples = [
            "What is this video about?",
            "Summarize the key points...",
            "What does the speaker say about...?",
            "What happens at the beginning?",
            "Give me the top 3 takeaways...",
            "What conclusions does the video draw?",
        ];

        let idx = 0;
        let charIdx = 0;
        let deleting = false;

        function getInput() {
            const doc = window.parent.document;
            return doc.querySelector('textarea[data-testid="stChatInputTextArea"]')
                || doc.querySelector('.stChatInput textarea')
                || doc.querySelector('textarea[placeholder]');
        }

        function tick() {
            const el = getInput();
            if (!el || window.parent.document.activeElement === el || el.value !== "") {
                setTimeout(tick, 500);
                return;
            }

            const current = examples[idx];

            if (!deleting) {
                charIdx++;
                el.setAttribute('placeholder', current.slice(0, charIdx));
                if (charIdx === current.length) {
                    deleting = true;
                    setTimeout(tick, 1800);
                    return;
                }
                setTimeout(tick, 60);
            } else {
                charIdx--;
                el.setAttribute('placeholder', current.slice(0, charIdx));
                if (charIdx === 0) {
                    deleting = false;
                    idx = (idx + 1) % examples.length;
                    setTimeout(tick, 400);
                    return;
                }
                setTimeout(tick, 30);
            }
        }

        // Start after parent page has had time to render the chat input
        setTimeout(tick, 1500);
    })();
    </script>
    """, height=0)

    # Chat Input
    user_query = st.chat_input("Ask a question about the video:")
    
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Save to chat history
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_query,
            "sources": []
        })
        
        # Process and generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript and formulating answer..."):
                try:
                    response = st.session_state["rag_chain"].invoke(user_query)
                    answer = response["answer"]
                    retrieved_docs = response["context"]
                    
                    # Process sources
                    sources = []
                    for doc in retrieved_docs:
                        start_time = doc.metadata.get("start_time", 0.0)
                        end_time = doc.metadata.get("end_time", 0.0)
                        
                        # Format timestamps nicely (HH:MM:SS or MM:SS)
                        def format_time(seconds):
                            secs = int(seconds)
                            mins = secs // 60
                            hours = mins // 60
                            if hours > 0:
                                return f"{hours:02d}:{mins%60:02d}:{secs%60:02d}"
                            return f"{mins:02d}:{secs%60:02d}"
                        
                        start_time_str = format_time(start_time)
                        end_time_str = format_time(end_time)
                        
                        # Generate click-to-play URL
                        # Note: YouTube timestamp URL parameter uses integer seconds, e.g. &t=120s
                        video_id = doc.metadata.get("video_id", st.session_state["video_id"])
                        url = f"https://youtu.be/{video_id}?t={int(start_time)}"
                        
                        sources.append({
                            "start_time_str": start_time_str,
                            "end_time_str": end_time_str,
                            "text": doc.page_content,
                            "url": url
                        })
                    
                    # Display Answer
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Sources & Timestamp Citations"):
                            for src in sources:
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <span class="citation-time">⏱️ Timestamp: {src['start_time_str']} - {src['end_time_str']}</span>
                                    </div>
                                    <div class="citation-text">"{src['text']}"</div>
                                    <a class="citation-link" href="{src['url']}" target="_blank">🔗 Watch video at {src['start_time_str']}</a>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Save to chat history
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
