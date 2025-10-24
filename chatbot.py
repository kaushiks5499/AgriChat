import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Page config
st.set_page_config(
    page_title="AgriChat Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AgriChat Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize top_k in session state
if "top_k" not in st.session_state:
    st.session_state.top_k = 3

# Initialize session state for query engine (load once)
if "query_engine" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        try:
            # Load embedding model
            hf_embeddings = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # Load storage context from ./vectors
            storage_context = StorageContext.from_defaults(persist_dir="./vectors")
            
            # Load index
            index = load_index_from_storage(
                storage_context, 
                embed_model=hf_embeddings
            )
            
            # Initialize Groq LLM
            llm = Groq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
            )
            
            # Store LLM in session state for reuse
            st.session_state.llm = llm
            
            # Create query engine with dynamic top_k from session state
            st.session_state.query_engine = index.as_query_engine(
                llm=llm,
                similarity_top_k=st.session_state.top_k
            )
            
            # Store index for stats
            st.session_state.index = index
            
            st.success("✅ Knowledge base loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {str(e)}")
            st.info("Make sure the './vectors' directory exists and contains your indexed data.")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"*Relevance Score: {source['score']:.3f}*")
                    
                    # Show document name and page
                    metadata = source.get("metadata", {})
                    if "file_name" in metadata:
                        st.markdown(f"**📄 Document:** {metadata['file_name']}")
                    if "page_label" in metadata:
                        st.markdown(f"**📖 Page:** {metadata['page_label']}")
                    
                    # Show metadata if available
                    if metadata:
                        if "document_title" in metadata:
                            st.markdown(f"**Title:** {metadata['document_title']}")
                        if "section_summary" in metadata:
                            st.markdown(f"**Summary:** {metadata['section_summary']}")
                    
                    # Show text preview
                    st.text_area(
                        f"Content {i}",
                        source["text"][:400] + "..." if len(source["text"]) > 400 else source["text"],
                        height=100,
                        key=f"source_{message.get('timestamp', '')}_{i}"
                    )
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from query engine
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.query_engine.query(prompt)
                
                # Debug: Print to console (check your terminal)
                print("\n" + "="*50)
                print(f"Query: {prompt}")
                print(f"Number of source nodes: {len(response.source_nodes) if hasattr(response, 'source_nodes') else 0}")
                
                if hasattr(response, 'source_nodes'):
                    for i, node in enumerate(response.source_nodes):
                        print(f"\n--- Source {i+1} ---")
                        print(f"Node type: {type(node)}")
                        print(f"Score: {node.score if hasattr(node, 'score') else 'N/A'}")
                        
                        # Handle different node types
                        try:
                            node_text = node.text if hasattr(node, 'text') else str(node.node.text) if hasattr(node, 'node') else node.get_content()
                        except:
                            node_text = node.get_content() if hasattr(node, 'get_content') else str(node)
                        
                        print(f"Text preview: {node_text[:200]}...")
                        
                        node_metadata = node.metadata if hasattr(node, 'metadata') else (node.node.metadata if hasattr(node, 'node') else {})
                        print(f"Metadata: {node_metadata}")
                
                print(f"\nResponse: {response.response}")
                print("="*50 + "\n")
                
                # Check if response is empty
                if not response.response or response.response.strip() == "":
                    st.error("⚠️ Received empty response from the model. Please try rephrasing your question.")
                    st.stop()
                
                # Extract sources
                sources = []
                if hasattr(response, 'source_nodes'):
                    for node in response.source_nodes:
                        # Handle both Document and TextNode types
                        try:
                            node_text = node.text if hasattr(node, 'text') else str(node.node.text) if hasattr(node, 'node') else node.get_content()
                        except:
                            node_text = node.get_content() if hasattr(node, 'get_content') else str(node)
                        
                        sources.append({
                            "text": node_text,
                            "score": node.score if hasattr(node, 'score') else 0.0,
                            "metadata": node.metadata if hasattr(node, 'metadata') else (node.node.metadata if hasattr(node, 'node') else {})
                        })
                
                # Display response
                st.markdown(response.response)
                
                # Debug info in UI if enabled
                if st.session_state.get("debug_mode", False):
                    st.info(f"📊 Retrieved {len(sources)} source documents")
                    with st.expander("🔍 Debug: Retrieved Context"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** (Score: {source['score']:.4f})")
                            st.code(source['text'][:500], language=None)
                            if source['metadata']:
                                st.json(source['metadata'])
                
                # Show sources in expander
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"*Relevance Score: {source['score']:.3f}*")
                            
                            # Show document name and page
                            metadata = source.get("metadata", {})
                            if "file_name" in metadata:
                                st.markdown(f"**📄 Document:** {metadata['file_name']}")
                            if "page_label" in metadata:
                                st.markdown(f"**📖 Page:** {metadata['page_label']}")
                            
                            # Show metadata if available
                            if metadata:
                                if "document_title" in metadata:
                                    st.markdown(f"**Title:** {metadata['document_title']}")
                                if "section_summary" in metadata:
                                    st.markdown(f"**Summary:** {metadata['section_summary']}")
                            
                            # Show text preview
                            st.text_area(
                                f"Content {i}",
                                source["text"][:400] + "..." if len(source["text"]) > 400 else source["text"],
                                height=100,
                                key=f"current_source_{i}"
                            )
                            st.divider()
                
                # Add assistant response to chat history
                import time
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.response,
                    "sources": sources,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                st.stop()

# Sidebar with controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Top K slider
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,  # Use current value from session state
        help="Higher values provide more context but may include less relevant information"
    )
    
    # Store in session state
    # Update query engine if top_k changed
    if st.session_state.top_k != top_k:
        st.session_state.top_k = top_k
        with st.spinner("Updating retrieval settings..."):
            # Reuse the LLM from session state instead of creating a new one
            st.session_state.query_engine = st.session_state.index.as_query_engine(
                llm=st.session_state.llm,
                similarity_top_k=top_k
            )
        st.success(f"✅ Now retrieving top {top_k} sources")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Debug mode toggle
    st.divider()
    debug_mode = st.checkbox("🐛 Debug Mode", value=False)
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    st.session_state.debug_mode = debug_mode
    
    # Show index info
    st.divider()
    st.subheader("📊 Knowledge Base Info")
    
    if "index" in st.session_state:
        try:
            # Get document count from docstore
            docstore = st.session_state.index.docstore
            num_docs = len(docstore.docs)
            st.metric("Total Documents/Chunks", num_docs)
            
            # Show embedding model
            st.info("**Embedding Model:**\nBAAI/bge-small-en-v1.5")
            st.info("**LLM Model:**\nLlama 3.3 70B (Groq)")
            
        except Exception as e:
            st.warning(f"Could not retrieve index stats: {e}")
    
    st.divider()
    st.markdown("""
    ### 💡 Tips
    - Ask specific questions about your documents
    - Check sources to verify information
    - Use debug mode to see retrieval details
    - Clear chat history to start fresh
    """)
    
    # API Key status
    st.divider()
    api_key_status = "✅ Set" if os.environ.get("GROQ_API_KEY") else "❌ Not Set"
    st.caption(f"GROQ_API_KEY: {api_key_status}")