import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CarGPT - Automotive Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Dark theme styling with automotive theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
        background-image: url('https://img.freepik.com/free-vector/abstract-car-background_23-2148303775.jpg');
        background-size: cover;
        background-blend-mode: overlay;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff5722 0%, #e91e63 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(255, 87, 34, 0.3);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #ff5722 0%, #e91e63 100%);
        color: white;
        margin-left: 2rem;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.3);
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-right: 2rem;
        color: #ffffff;
        backdrop-filter: blur(10px);
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        box-shadow: 0 2px 10px rgba(255, 107, 107, 0.4);
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
        box-shadow: 0 2px 10px rgba(78, 205, 196, 0.4);
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    .model-info {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        color: #ffffff;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #ff5722 0%, #e91e63 100%);
        border: none;
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.5);
        background: linear-gradient(135deg, #e91e63 0%, #ff5722 100%);
    }
    
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
        padding: 0.75rem 1rem;
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 0.2rem rgba(0, 212, 255, 0.25);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        margin-right: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ff5722;
        animation: typing 1.4s infinite ease-in-out;
        box-shadow: 0 0 10px rgba(255, 87, 34, 0.5);
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Additional dark theme elements */
    .stMarkdown {
        color: #ffffff;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSidebar .stMarkdown {
        color: #ffffff;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff5722 0%, #e91e63 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #e91e63 0%, #ff5722 100%);
    }
</style>
""", unsafe_allow_html=True)

class OllamaClient:
    def _init_(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "gemma3:1b"
        self.system_prompt = """
You are CarGPT, an expert assistant specialized in automotive data. You provide detailed car specifications, pricing, and the latest models for various car brands and types. You understand and respond to user queries regarding:
- Latest car models (2023–2025)
- Price ranges (MSRP and market price)
- Engine types, fuel efficiency, horsepower, torque
- EV, hybrid, and petrol/diesel variants
- Safety ratings and features
- Comparisons between cars

Always answer in a concise, user-friendly tone. If data is not available, clearly say it's unavailable or advise checking with official sources. Prioritize clarity, accuracy, and usefulness.
"""
    
    def generate_response(self, prompt: str, stream: bool = True) -> str:
        """Generate response from Ollama API"""
        try:
            url = f"{self.base_url}/api/generate"
            
            # Combine system prompt with user query
            full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\n\nCarGPT:"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 1000
                }
            }
            
            if stream:
                return self._stream_response(url, payload)
            else:
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()["response"]
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Ollama: {str(e)}")
            return "I'm sorry, I'm having trouble connecting to the AI model. Please make sure Ollama is running and the Gemma3:1b model is available."
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return "An unexpected error occurred. Please try again."
    
    def _stream_response(self, url: str, payload: Dict[str, Any]) -> str:
        """Handle streaming response from Ollama"""
        try:
            response = requests.post(url, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            full_response += data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            return full_response
            
        except Exception as e:
            st.error(f"Error in streaming response: {str(e)}")
            return "Error occurred while generating response."

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = OllamaClient()

def display_chat_message(role: str, content: str):
    """Display a chat message with proper styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-avatar user-avatar">👤</div>
            <div style="flex: 1;">
                <strong>You:</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-avatar assistant-avatar">🚗</div>
            <div style="flex: 1;">
                <strong>CarGPT:</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚗 CarGPT</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; margin-bottom: 2rem; color: #ffffff;">Your expert automotive assistant for car specifications, pricing, and models</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙ Settings")
        
        # Model information
        st.markdown("""
        <div class="model-info">
            <strong>Model:</strong> CarGPT (Gemma3:1b)<br>
            <strong>Specialization:</strong> Automotive Data<br>
            <strong>Data Coverage:</strong> 2023-2025 Models<br>
            <strong>Status:</strong> Active
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑 Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
        
        # About section
        st.markdown("---")
        st.markdown("### ℹ About CarGPT")
        st.markdown("""
        CarGPT is your automotive expert assistant powered by the Gemma3:1b model.
        
        *Capabilities:*
        - Latest car models (2023-2025)
        - Price ranges and comparisons
        - Engine specifications and performance
        - EV, hybrid, and conventional options
        - Safety features and ratings
        - Vehicle comparisons and recommendations
        
        *Requirements:*
        - Ollama running locally
        - Gemma3:1b model installed
        """)
        
        # Installation instructions
        st.markdown("### 📥 Installation")
        st.markdown("""
        bash
        # Install Ollama
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Pull the model
        ollama pull gemma3:1b
        
        # Run CarGPT
        streamlit run app.py
        
        """)
    
    # Main chat area
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col2:
        # Display chat messages
        if not st.session_state.messages:
            # Show sample queries when no messages exist
            st.markdown("<h3 style='text-align: center; color: #ffffff; margin-bottom: 1rem;'>👋 Welcome to CarGPT!</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #ffffff; margin-bottom: 2rem;'>Try asking about car specifications, pricing, or the latest models.</p>", unsafe_allow_html=True)
            
            # Sample queries in a grid
            query_col1, query_col2 = st.columns(2)
            
            with query_col1:
                if st.button("What are the top electric vehicles for 2024?"):
                    st.session_state.messages.append({"role": "user", "content": "What are the top electric vehicles for 2024?"})
                    st.rerun()
                    
                if st.button("Compare Toyota Camry vs Honda Accord 2024"):
                    st.session_state.messages.append({"role": "user", "content": "Compare Toyota Camry vs Honda Accord 2024"})
                    st.rerun()
                    
                if st.button("What's the fuel efficiency of the 2024 Ford F-150?"):
                    st.session_state.messages.append({"role": "user", "content": "What's the fuel efficiency of the 2024 Ford F-150?"})
                    st.rerun()
            
            with query_col2:
                if st.button("What safety features come with the 2024 Subaru Outback?"):
                    st.session_state.messages.append({"role": "user", "content": "What safety features come with the 2024 Subaru Outback?"})
                    st.rerun()
                    
                if st.button("What's the price range for a 2024 Tesla Model Y?"):
                    st.session_state.messages.append({"role": "user", "content": "What's the price range for a 2024 Tesla Model Y?"})
                    st.rerun()
                    
                if st.button("What are the best hybrid SUVs for 2024?"):
                    st.session_state.messages.append({"role": "user", "content": "What are the best hybrid SUVs for 2024?"})
                    st.rerun()
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # Chat input
        st.markdown("---")
        
        # Input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "💬 Type your automotive query here...",
                height=100,
                placeholder="Ask about car models, prices, specs, or comparisons...",
                key="user_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit_button = st.form_submit_button("🚀 Send Message")
        
        # Handle form submission
        if submit_button and user_input.strip():
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            
            # Display user message
            display_chat_message("user", user_input.strip())
            
            # Show typing indicator
            with st.spinner("🚗 CarGPT is thinking..."):
                # Get AI response
                ai_response = st.session_state.ollama_client.generate_response(user_input.strip())
                
                # Add AI response to chat
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
                # Display AI response
                display_chat_message("assistant", ai_response)
            
            # Rerun to update the display
            st.rerun()

if _name_ == "_main_":
    main()