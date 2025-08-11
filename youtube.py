import streamlit as st
import re
from groq import Groq
from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd
import time

def get_video_id(url):
    """
    Extracts the YouTube video ID from a given URL.
    """
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    else:
        # Standard YouTube URL
        match = re.search(r"v=([^&]+)", url)
        if match:
            return match.group(1)
    return None

def fetch_comments(video_id, max_comments=50):
    """
    Fetches comments for a given YouTube video ID using youtube-comment-downloader.
    This may not be able to fetch all comments if the video is very popular.
    """
    downloader = YoutubeCommentDownloader()
    comments_list = []
    try:
        # Fetching comments using the downloader library
        comments_gen = downloader.get_comments(video_id, sort_by=0) # 1 for newest comments, 0 for popular
        for comment in comments_gen:
            comments_list.append(comment['text'])
            if len(comments_list) >= max_comments:
                break
    except Exception as e:
        st.error(f"An error occurred while fetching comments: {e}")
        return []
    
    return comments_list

def summarize_comments_with_groq(groq_api_key, model_name, comments,instructions="Be precise"):
    """
    Sends the comments to the Groq API for summarization.
    """
    if not comments:
        return "No comments to summarize."

    client = Groq(api_key=groq_api_key)
    
    # Create the prompt for the LLM
    prompt = f"""
    You are a professional content summarizer. 
    Your task is to take a list of YouTube comments and condense them intopoints. 
    The points should capture the main topics, sentiments, and recurring themes found in the comments.
    Capture all types of sentiments.
    Instructions:{instructions}

    Here are the comments:
    { " ".join(comments) }
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.5,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the Groq API: {e}"

# --- Streamlit App Layout ---

st.set_page_config(page_title="YouTube Comment Summarizer", layout="wide")

st.title("YouTube Comment Summarizer 💬")
st.write("This application scrapes comments from a YouTube video and uses the Groq API to provide summary.")

# Input fields
with st.container():
    st.header("1. Enter Video & API Details")
    youtube_url = st.text_input("YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="Enter your Groq API key here")
    model_name = st.text_input("Groq Model Name", value="gemma2-9b-it", placeholder="gemma2-9b-it")
    instructions=st.text_input("Kindly give instructions to AI  (Optional)",value=None,placeholder="Be precise and concise")

    if st.button("Summarize Comments", use_container_width=True, type="primary"):
        if not youtube_url or not groq_api_key or not model_name:
            st.error("Please fill in all the fields.")
        else:
            with st.spinner("Fetching and summarizing comments... This may take a moment."):
                video_id = get_video_id(youtube_url)
                if not video_id:
                    st.error("Invalid YouTube URL provided. Please check the link and try again.")
                else:
                    comments = fetch_comments(video_id)
                
                    if not comments:
                        st.info("No comments were found for this video or there was an issue fetching them.")
                    else:
                        if len(comments)>50:
                            comments=comments[:40]
                        summary = summarize_comments_with_groq(groq_api_key, model_name, comments,instructions)
                        
                        st.header("2. Summary of Comments")
                        st.markdown(summary)
                        
                        st.header("3. Raw Comments (First 50)")
                        # Display raw comments for context
                        comment_df = pd.DataFrame(comments[:50], columns=["Comment Text"])

                        st.dataframe(comment_df, use_container_width=True)










