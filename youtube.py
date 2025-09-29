import streamlit as st
import re
from groq import Groq
from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd

def get_video_id(url):
    """Extract YouTube video ID from URL."""
    if not url:
        return None
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    else:
        match = re.search(r"v=([^&]+)", url)
        if match:
            return match.group(1)
    return None

def fetch_comments(video_id, max_comments=50, sort_by=0):
    """
    Fetch comments for a YouTube video.
    sort_by: 0 = popular, 1 = newest
    Returns: (comments_list, error_message)
    """
    downloader = YoutubeCommentDownloader()
    comments_list = []
    
    try:
        comments_gen = downloader.get_comments(video_id, sort_by=sort_by)
        for comment in comments_gen:
            comments_list.append(comment['text'])
            if len(comments_list) >= max_comments:
                break
                
        if not comments_list:
            return [], "No comments found. The video might have comments disabled or be private."
            
        return comments_list, None
        
    except Exception as e:
        error_msg = str(e)
        if "Video unavailable" in error_msg:
            return [], "Video not found or is private/unavailable."
        elif "comments are disabled" in error_msg.lower():
            return [], "Comments are disabled for this video."
        else:
            return [], f"Error fetching comments: {error_msg}"

def estimate_tokens(text):
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4

def summarize_comments_with_groq(groq_api_key, model_name, comments, instructions=""):
    """Send comments to Groq API for summarization."""
    if not comments:
        return None, "No comments to summarize."

    client = Groq(api_key=groq_api_key)
    
    # Build prompt
    instruction_text = f"\n\nAdditional Instructions: {instructions}" if instructions else ""
    
    prompt = f"""You are a professional content analyst. Analyze these YouTube comments and provide:

1. **Main Themes** - Key topics discussed (3-5 bullet points)
2. **Sentiment Breakdown** - Overall tone (positive/negative/mixed) with percentages
3. **Notable Insights** - Interesting or recurring observations
4. **Top Concerns/Praise** - What viewers loved or complained about most
{instruction_text}

Comments to analyze:
{chr(10).join(f"- {comment[:200]}" for comment in comments)}
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.5,
            max_tokens=1500,
        )
        return chat_completion.choices[0].message.content, None
        
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return None, "Invalid API key. Check your Groq API credentials."
        elif "rate limit" in error_msg.lower():
            return None, "Rate limit exceeded. Wait a moment and try again."
        elif "model" in error_msg.lower():
            return None, f"Model '{model_name}' not found or not accessible."
        else:
            return None, f"API Error: {error_msg}"

# --- Streamlit App ---

st.set_page_config(
    page_title="YouTube Comment Summarizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #FF0000, #CC0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        font-weight: 600;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF0000;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📺 YouTube Comment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get AI-powered insights from video comments in seconds</p>', unsafe_allow_html=True)

# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'comments' not in st.session_state:
    st.session_state.comments = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    groq_api_key = st.text_input(
        "Groq API Key", 
        type="password", 
        help="Get your key from console.groq.com"
    )
    
    model_name = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
        index=1
    )
    
    st.divider()
    
    max_comments = st.slider(
        "Max Comments to Fetch",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="More comments = better analysis but higher cost"
    )
    
    sort_order = st.radio(
        "Sort Comments By",
        ["Popular", "Newest"],
        help="Popular comments often reflect consensus"
    )
    
    instructions = st.text_area(
        "Custom Instructions (Optional)",
        placeholder="e.g., Focus on technical feedback only",
        height=100
    )
    
    st.divider()
    st.caption("💡 Tip: Start with 50 popular comments for best results")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )

with col2:
    analyze_button = st.button(
        "🚀 Analyze Comments", 
        use_container_width=True, 
        type="primary",
        disabled=not (youtube_url and groq_api_key)
    )

if not groq_api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started")

# Process when button clicked
if analyze_button:
    video_id = get_video_id(youtube_url)
    
    if not video_id:
        st.error("❌ Invalid YouTube URL. Make sure it's in the format: youtube.com/watch?v=... or youtu.be/...")
    else:
        # Fetch comments
        with st.spinner("📥 Fetching comments..."):
            sort_by = 1 if sort_order == "Newest" else 0
            comments, error = fetch_comments(video_id, max_comments, sort_by)
        
        if error:
            st.error(f"❌ {error}")
        elif comments:
            st.session_state.comments = comments
            st.session_state.video_id = video_id
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Comments Fetched", len(comments))
            with col2:
                estimated = estimate_tokens(" ".join(comments))
                st.metric("Est. Tokens", f"~{estimated:,}")
            with col3:
                st.metric("Est. Cost", f"~${estimated * 0.00001:.4f}")
            
            # Confirm and summarize
            st.info("⚡ Ready to analyze. This will use your Groq API credits.")
            
            with st.spinner("🤖 AI is analyzing comments..."):
                summary, error = summarize_comments_with_groq(
                    groq_api_key, 
                    model_name, 
                    comments,
                    instructions
                )
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.session_state.summary = summary

# Display results
if st.session_state.summary:
    st.success("✅ Analysis complete!")
    
    st.markdown("---")
    st.subheader("📊 Comment Analysis")
    st.markdown(st.session_state.summary)
    
    # Show raw comments in expander
    with st.expander(f"📝 View Raw Comments ({len(st.session_state.comments)} total)", expanded=False):
        for i, comment in enumerate(st.session_state.comments, 1):
            st.text(f"{i}. {comment[:300]}{'...' if len(comment) > 300 else ''}")
    
    # Download option
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        summary_download = f"# YouTube Comment Analysis\n\n{st.session_state.summary}\n\n---\n\n## Raw Comments\n\n"
        summary_download += "\n\n".join(f"{i}. {c}" for i, c in enumerate(st.session_state.comments, 1))
        
        st.download_button(
            "⬇️ Download Full Report",
            summary_download,
            file_name=f"youtube_analysis_{st.session_state.video_id}.txt",
            mime="text/plain"
        )
    
    with col2:
        if st.button("🔄 Analyze Another Video"):
            st.session_state.summary = None
            st.session_state.comments = None
            st.session_state.video_id = None
            st.rerun()

st.markdown("---")
st.caption("Built with Streamlit + Groq | Data sourced from public YouTube comments")
