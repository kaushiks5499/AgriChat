import streamlit as st
import os
import sqlite3
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import supabase
from supabase import create_client
from io import BytesIO
import hashlib
import pickle
from datetime import datetime

# Import the update module
import update_db





def get_conversation_history(max_messages=10, include_system=True):
    """Get recent conversation history for context, excluding current query"""
    messages = st.session_state.messages

    if len(messages) <= 1:  # No previous conversation
        return ""

    # Get the last max_messages pairs, excluding the current user message
    recent_messages = messages[-max_messages-1:-1] if len(messages) > 1 else messages[:-1]

    history_parts = []
    for msg in recent_messages:
        role = msg["role"]
        content = msg["content"][:500]  # Truncate long messages
        history_parts.append(f"{role.title()}: {content}")

    return "\n".join(history_parts)


# Load in the API keys
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    UPLOAD_PASSWORD = st.secrets.get("UPLOAD_PASSWORD")  # Default password
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    UPLOAD_PASSWORD = os.environ.get("UPLOAD_PASSWORD")  # Default password

# Database configuration
try:
    # Initialize Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Fetch data from Supabase
    response = supabase.table("VARIETY_YIELD").select("*").execute()
    
    # Check if data exists
    if not response.data:
        st.error("❌ No data found in VARIETY_YIELD table")
        st.stop()
    
    # Convert to DataFrame
    DB_IN_DF = pd.DataFrame(response.data)
    
    # Validate DataFrame is not empty
    if DB_IN_DF.empty:
        st.error("❌ VARIETY_YIELD table is empty")
        st.stop()
    
    # Optional: Validate expected columns exist
    expected_columns = ['id', 'Variety', 'Lint', 'Turnout', 'Micronaire', 'Length', 
                       'Strength', 'Uniformity', 'LoanValue', 'LintValue', 
                       'TrialLocation', 'Year']
    missing_columns = set(expected_columns) - set(DB_IN_DF.columns)
    
    if missing_columns:
        st.warning(f"⚠️ Missing expected columns: {missing_columns}")
    
except Exception as e:
    error_message = str(e).lower()
    
    # Handle specific error types
    if "api key" in error_message or "unauthorized" in error_message:
        st.error("❌ Invalid Supabase API key or unauthorized access")
        st.info("Please check your SUPABASE_KEY in environment variables or secrets")
    elif "network" in error_message or "connection" in error_message:
        st.error("❌ Network error: Cannot connect to Supabase")
        st.info("Please check your internet connection and Supabase URL")
    elif "table" in error_message or "relation" in error_message:
        st.error("❌ Table 'VARIETY_YIELD' not found in Supabase database")
        st.info("Please verify the table name and database structure")
    else:
        st.error(f"❌ Error loading data from Supabase: {str(e)}")
    
    st.stop()

DB_PATH = "SOUTH_EAST_CENTRAL_TRIALS_2018_to_2024.db"

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

# Initialize mathematical mode in session state
if "math_mode" not in st.session_state:
    st.session_state.math_mode = True

# Initialize upload dialog state
if "show_upload_dialog" not in st.session_state:
    st.session_state.show_upload_dialog = False

# Initialize password verification state
if "password_verified" not in st.session_state:
    st.session_state.password_verified = False


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
            
            #st.success("✅ Knowledge base loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {str(e)}")
            st.info("Make sure the './vectors' directory exists and contains your indexed data.")
            st.stop()

def execute_sql_query(sql_query):
    """Execute SQL query on the database and return results"""
    try:
        conn = sqlite3.connect(':memory:')
        DB_IN_DF.to_sql('VARIETY_YIELD', conn, if_exists='replace', index=False)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()
        return results, column_names
    except Exception as e:
        raise Exception(f"SQL execution error: {str(e)}")

def generate_sql_from_query(user_query):
    """Use LLM to convert natural language to SQL"""
    # Get conversation context for better SQL generation
    conversation_history = get_conversation_history(max_messages=8)

    context_section = ""
    if conversation_history:
        context_section = f"""
Previous Conversation Context:
{conversation_history}

Use this context to understand follow-up questions and maintain consistency in your SQL generation.
"""

    prompt = f"""You are an expert SQL query generator for a cotton variety trials database.{context_section}

Database Schema:
- Table: VARIETY_YIELD
- Columns:
  * Variety (TEXT): Cotton variety name
  * Lint (FLOAT): Lint yield in lbs/acre
  * Turnout (FLOAT): Turnout percentage
  * Micronaire (FLOAT): Micronaire value
  * Length (FLOAT): Fiber length in inches
  * Strength (FLOAT): Fiber strength in g/tex
  * Uniformity (FLOAT): Uniformity percentage
  * LoanValue (FLOAT): Loan value in cents/lb
  * LintValue (FLOAT): Lint value in USD/acre
  * TrialLocation (TEXT): Location of the trial
  * Year (FLOAT): Year of the trial

IMPORTANT RULES:
1. When querying TrialLocation, ALWAYS use LIKE with wildcards: WHERE TrialLocation LIKE '%LocationName%'
2. Return ONLY the SQL query, no explanations or markdown
3. Use proper SQL syntax for SQLite
4. For aggregations (max, min, avg), include relevant grouping
5. Always include Year in results when specified in the query

Current User Query: {user_query}

Generate the SQL query:"""
    try:
        response = st.session_state.llm.complete(prompt)
        sql_query = response.text.strip()
        # Remove markdown code blocks if present
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        return sql_query
    except Exception as e:
        raise Exception(f"Error generating SQL: {str(e)}")

def should_visualize_heuristic(results, column_names, sql_query):
    """
    Stage 1: Fast heuristic check to reject visualization without LLM call
    Returns: (should_check_llm: bool, reason: str)
    """
    # Reject if no results
    if not results or len(results) == 0:
        return False, "No results to visualize"
    
    # Reject if single row
    if len(results) == 1:
        return False, "Single value result - no visualization needed"
    
    # Reject if single column
    if len(column_names) == 1:
        return False, "Single column result - no visualization needed"
    
    # Check if we have at least one numeric column (excluding potential grouping columns)
    df = pd.DataFrame(results, columns=column_names)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return False, "No numeric data to visualize"
    
    # Reject if less than 2 rows
    if len(results) < 2:
        return False, "Insufficient data points for visualization"
    
    # SQL-based heuristics: Check for single aggregate functions without GROUP BY
    sql_upper = sql_query.upper()
    
    # Check if query uses aggregate functions
    has_aggregate = any(func in sql_upper for func in ['COUNT(', 'MAX(', 'MIN(', 'AVG(', 'SUM('])
    
    # Check if query has GROUP BY
    has_group_by = 'GROUP BY' in sql_upper
    
    # Reject if aggregate without GROUP BY (single value queries)
    if has_aggregate and not has_group_by:
        return False, "Single aggregate value (COUNT/MAX/MIN/AVG/SUM without GROUP BY)"
    
    # Additional check: If single row with aggregate, it's likely a single value
    if has_aggregate and len(results) == 1:
        return False, "Single aggregate result - no visualization needed"
    
    # Passed all heuristics - worth asking LLM
    return True, "Passed heuristic checks"

def get_visualization_decision(user_query, sql_query, results, column_names):
    """
    Stage 2: Ask LLM to decide if visualization is needed and provide parameters
    Returns: dict with visualization parameters or None
    """
    # Create dataframe for preview
    df = pd.DataFrame(results, columns=column_names)
    results_preview = df.head(10).to_string()
    
    # Get column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    prompt = f"""You are a data visualization expert. Analyze if this query result would benefit from visualization.

User Question: {user_query}

SQL Query: {sql_query}

Results Preview (first 10 rows):
{results_preview}

Total Rows: {len(results)}
Columns: {column_names}
Numeric Columns: {numeric_cols}
Text/Categorical Columns: {text_cols}

Decide if a visualization would add value. Consider:
- Trends over time (Year column)
- Comparisons between varieties or locations
- Distribution patterns
- Relationships between variables

Respond ONLY with valid JSON in this exact format:
{{
  "should_visualize": true/false,
  "chart_type": "line/bar/scatter/box/pie",
  "x_column": "column name for x-axis",
  "y_column": "column name for y-axis",
  "color_column": "column name for grouping (optional, use null if not needed)",
  "title": "Chart title",
  "x_label": "X-axis label",
  "y_label": "Y-axis label",
  "reasoning": "Brief explanation why visualization helps or doesn't help"
}}

Chart type guidelines:
- line: Time series or trends (requires Year or sequential data)
- bar: Categorical comparisons, variety comparisons
- scatter: Relationship between two numeric variables
- box: Distribution analysis across categories
- pie: Only for proportions/percentages that sum meaningfully

If should_visualize is false, still provide the reasoning field but other fields can be null."""

    try:
        response = st.session_state.llm.complete(prompt)
        response_text = response.text.strip()
        
        # Clean up response - remove markdown code blocks if present
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON response
        viz_decision = json.loads(response_text)
        
        return viz_decision
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Response text: {response_text}")
        return None
    except Exception as e:
        print(f"Error getting visualization decision: {e}")
        return None

def create_visualization(df, viz_params):
    """
    Create a Plotly visualization based on LLM-provided parameters
    """
    try:
        chart_type = viz_params.get('chart_type', 'bar')
        x_col = viz_params.get('x_column')
        y_col = viz_params.get('y_column')
        color_col = viz_params.get('color_column')
        title = viz_params.get('title', 'Data Visualization')
        x_label = viz_params.get('x_label', x_col)
        y_label = viz_params.get('y_label', y_col)

        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(int)
            if x_col == 'Year':
                df['Year'] = df['Year'].astype(str)
        
        # Create appropriate chart based on type
        if chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, color=color_col,
                         title=title,
                         labels={x_col: x_label, y_col: y_label})
            
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                        title=title,
                        labels={x_col: x_label, y_col: y_label})
            
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=title,
                           labels={x_col: x_label, y_col: y_label})
            
        elif chart_type == 'box':
            fig = px.box(df, x=x_col, y=y_col, color=color_col,
                        title=title,
                        labels={x_col: x_label, y_col: y_label})
            
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_col, values=y_col,
                        title=title)
        else:
            # Default to bar chart
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                        title=title,
                        labels={x_col: x_label, y_col: y_label})
        
        # Update layout for better appearance
        fig.update_layout(
            template='plotly_white',
            hovermode='x unified'
        )
        if x_col == 'Year':
            fig.update_xaxes(type='category')
        return fig
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

def generate_natural_language_response(user_query, sql_query, results, column_names):
    """Use LLM to convert SQL results back to natural language"""
    # Get conversation context for more coherent responses
    conversation_history = get_conversation_history(max_messages=6)

    context_section = ""
    if conversation_history:
        context_section = f"""
Previous Conversation Context:
{conversation_history}

Use this context to provide coherent, contextual responses that reference previous questions and maintain conversation flow.
"""

    # Format results as a readable string (use all results for accurate LLM answers)
    if not results:
        results_text = "No results found."
    else:
        results_text = "Query Results:\n"
        results_text += " | ".join(column_names) + "\n"
        results_text += "-" * 50 + "\n"
        for row in results:  # Use all rows for LLM context to ensure accurate answers
            results_text += " | ".join(str(val) for val in row) + "\n"

    prompt = f"""You are a helpful agricultural data analyst having an ongoing conversation.{context_section}

Current User Question: {user_query}

SQL Query Executed:
{sql_query}

{results_text}

Provide a clear, concise answer to the user's question based on the query results.
- If there are specific numbers, include them
- Be conversational and natural, referencing previous context when relevant
- If no results were found, explain that clearly
- If there are multiple results, summarize the key findings
- Maintain continuity with the ongoing conversation"""

    try:
        response = st.session_state.llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

def refresh_database():
    """Refresh the database from Supabase"""
    global DB_IN_DF
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        response = supabase_client.table("VARIETY_YIELD").select("*").execute()
        DB_IN_DF = pd.DataFrame(response.data)
        return True, f"✅ Database refreshed: {len(DB_IN_DF)} records loaded"
    except Exception as e:
        return False, f"❌ Error refreshing database: {str(e)}"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show visualization if available
        if message["role"] == "assistant" and "visualization" in message and message["visualization"] is not None:
            # Use timestamp as unique key to avoid duplicate ID errors
            chart_key = f"chart_{message.get('timestamp', 0)}"
            st.plotly_chart(message["visualization"], use_container_width=True, key=chart_key)
            if "viz_reasoning" in message:
                st.caption(f"📊 {message['viz_reasoning']}")
        
        # Show SQL results if in math mode
        if message["role"] == "assistant" and "sql_query" in message:
            with st.expander("🔢 Tabular Results"):
                if "sql_results" in message and message["sql_results"] is not None:
                    st.dataframe(message["sql_results"])
        
        # Show sources if available (RAG mode)
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"*Relevance Score: {source['score']:.3f}*")
                    
                    metadata = source.get("metadata", {})
                    if "file_name" in metadata:
                        st.markdown(f"**📄 Document:** {metadata['file_name']}")
                    if "page_label" in metadata:
                        st.markdown(f"**📖 Page:** {metadata['page_label']}")
                    
                    if metadata:
                        if "document_title" in metadata:
                            st.markdown(f"**Title:** {metadata['document_title']}")
                        if "section_summary" in metadata:
                            st.markdown(f"**Summary:** {metadata['section_summary']}")
                    
                    st.text_area(
                        f"Content {i}",
                        source["text"][:400] + "..." if len(source["text"]) > 400 else source["text"],
                        height=100,
                        key=f"source_{message.get('timestamp', '')}_{i}"
                    )
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything about cotton variety trials..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response based on mode
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.math_mode:
                    # Mathematical Mode: SQL-based query with visualization
                    with st.spinner("Generating SQL query..."):
                        sql_query = generate_sql_from_query(prompt)
                        print(f"Generated SQL Query: {sql_query}")  # Print to console

                    with st.spinner("Executing query..."):
                        results, column_names = execute_sql_query(sql_query)
                        print(f"Query returned {len(results)} results")  # Print to console
                    
                    # Visualization pipeline
                    visualization = None
                    viz_reasoning = None
                    
                    if results:
                        # Stage 1: Heuristic check
                        should_check_llm, heuristic_reason = should_visualize_heuristic(results, column_names, sql_query)
                        
                        print(f"Heuristic Check: {heuristic_reason}")  # Print to console
                        
                        if should_check_llm:
                            # Stage 2: LLM decision
                            with st.spinner("Analyzing if visualization would be helpful..."):
                                viz_decision = get_visualization_decision(prompt, sql_query, results, column_names)
                                
                                if st.session_state.get("debug_mode", False) and viz_decision:
                                    st.json(viz_decision)
                                
                                if viz_decision and viz_decision.get('should_visualize', False):
                                    # Create visualization
                                    df = pd.DataFrame(results, columns=column_names)
                                    visualization = create_visualization(df, viz_decision)
                                    viz_reasoning = viz_decision.get('reasoning', '')
                                    
                                    if visualization:
                                        st.plotly_chart(visualization, use_container_width=True, key="current_response_chart")
                                        st.caption(f"📊 {viz_reasoning}")
                                elif viz_decision:
                                    viz_reasoning = viz_decision.get('reasoning', 'Visualization not needed')
                                    print(f"Visualization decision: {viz_reasoning}")  # Print to console
                    
                    with st.spinner("Generating answer..."):
                        response_text = generate_natural_language_response(
                            prompt, sql_query, results, column_names
                        )
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Show results in expander
                    with st.expander("🔢 Tabular Results"):
                        if results:
                            df = pd.DataFrame(results, columns=column_names)
                            st.dataframe(df)
                        else:
                            st.warning("No results found")
                    
                    # Add to chat history
                    import time
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sql_query": sql_query,
                        "sql_results": pd.DataFrame(results, columns=column_names) if results else None,
                        "visualization": visualization,
                        "viz_reasoning": viz_reasoning,
                        "timestamp": time.time()
                    })
                    
                else:
                    # RAG Mode: Document-based query with conversation context
                    # Get conversation history for context-aware queries
                    conversation_history = get_conversation_history(max_messages=10)

                    # Enhance the query with conversation context
                    if conversation_history:
                        enhanced_prompt = f"""Previous conversation context:
{conversation_history}

Current question: {prompt}

Please provide a comprehensive answer based on the documents, taking into account the conversation history above."""
                    else:
                        enhanced_prompt = prompt

                    response = st.session_state.query_engine.query(enhanced_prompt)
                    
                    # Debug: Print to console
                    print("\n" + "="*50)
                    print(f"Query: {prompt}")
                    print(f"Number of source nodes: {len(response.source_nodes) if hasattr(response, 'source_nodes') else 0}")
                    
                    if hasattr(response, 'source_nodes'):
                        for i, node in enumerate(response.source_nodes):
                            print(f"\n--- Source {i+1} ---")
                            print(f"Node type: {type(node)}")
                            print(f"Score: {node.score if hasattr(node, 'score') else 'N/A'}")
                            
                            try:
                                node_text = node.text if hasattr(node, 'text') else str(node.node.text) if hasattr(node, 'node') else node.get_content()
                            except:
                                node_text = node.get_content() if hasattr(node, 'get_content') else str(node)
                            
                            print(f"Text preview: {node_text[:200]}...")
                            
                            node_metadata = node.metadata if hasattr(node, 'metadata') else (node.node.metadata if hasattr(node, 'node') else {})
                            print(f"Metadata: {node_metadata}")
                    
                    print(f"\nResponse: {response.response}")
                    print("="*50 + "\n")
                    
                    if not response.response or response.response.strip() == "":
                        st.error("⚠️ Received empty response from the model. Please try rephrasing your question.")
                        st.stop()
                    
                    # Extract sources
                    sources = []
                    if hasattr(response, 'source_nodes'):
                        for node in response.source_nodes:
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
                                
                                metadata = source.get("metadata", {})
                                if "file_name" in metadata:
                                    st.markdown(f"**📄 Document:** {metadata['file_name']}")
                                if "page_label" in metadata:
                                    st.markdown(f"**📖 Page:** {metadata['page_label']}")
                                
                                if metadata:
                                    if "document_title" in metadata:
                                        st.markdown(f"**Title:** {metadata['document_title']}")
                                    if "section_summary" in metadata:
                                        st.markdown(f"**Summary:** {metadata['section_summary']}")
                                
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

    # Data Guide - Always visible at top of sidebar
    st.subheader("📖 What You Can Ask About")
    st.markdown("""
    Use these terms to ask specific questions about the cotton trials:

    **🌾 Basic Info**
    * **Variety**: The name of the cotton type.
    * **Trial Location**: Where the cotton was grown.
    * **Year**: The year of the harvest.

    **💰 Production & Money**
    * **Lint**: The amount of cotton produced (Yield in lbs/acre).
    * **Lint Value**: Total revenue earned per acre ($/acre).
    * **Turnout**: The percentage of usable cotton.

    **🧵 Quality Metrics**
    * **Micronaire**: How thick/fine the fiber is.
    * **Length**: How long the fibers are (inches).
    * **Strength**: How strong the fibers are (g/tex).
    """)
    st.info("💡 **Example Questions:**\n"
           "* 'Which Variety had the highest Lint yield in 2023?'\n"
           "* 'How did PHY 400 W3FE yield change over years?'\n"
           "* 'What was max yield in Burleson 2024?'")

    st.divider()

    # Data Upload Section (Password Protected)
    st.subheader("📤 Data Management")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Upload New Trial Data**")
    with col2:
        if st.button("➕", help="Upload new trial data (password required)"):
            st.session_state.show_upload_dialog = True
            st.session_state.password_verified = False
    
    # Upload Dialog
    if st.session_state.show_upload_dialog:
        st.divider()
        st.markdown("### 🔒 Secure Data Upload")
        
        if not st.session_state.password_verified:
            # Password input
            password_input = st.text_input(
                "Enter Password:",
                type="password",
                key="upload_password_input"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Verify", use_container_width=True):
                    if password_input == UPLOAD_PASSWORD:
                        st.session_state.password_verified = True
                        st.success("✅ Password verified!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_upload_dialog = False
                    st.session_state.password_verified = False
                    st.rerun()
        
        else:
            # File upload interface
            st.success("🔓 Access granted")
            
            st.info("📋 **File Format Requirements:**\n"
                   "- Filename: `[YEAR] description.xlsx` (e.g., '2024 County RACE.xlsx')\n"
                   "- Must contain sheets with trial data\n"
                   "- Required columns: Variety, Yield, Turnout, Mic, Length, Strength, Uniformity, Loan , Lint value")
            
            uploaded_file = st.file_uploader(
                "Choose an Excel file",
                type=['xlsx'],
                key="trial_data_uploader"
            )
            
            if uploaded_file is not None:
                st.markdown(f"**Selected file:** `{uploaded_file.name}`")
                
                # Preview file info
                file_size = uploaded_file.size / 1024  # Convert to KB
                st.caption(f"File size: {file_size:.2f} KB")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📤 Upload & Process", use_container_width=True, type="primary"):
                        with st.spinner("Processing Excel file..."):
                            try:
                                # Call the update_db module to process and upload
                                success, message = update_db.update_database_from_excel(
                                    uploaded_file,
                                    uploaded_file.name
                                )
                                
                                if success:
                                    st.success(message)
                                    
                                    # Refresh the database
                                    with st.spinner("Refreshing database..."):
                                        refresh_success, refresh_msg = refresh_database()
                                        if refresh_success:
                                            st.success(refresh_msg)
                                        else:
                                            st.warning(refresh_msg)
                                    
                                    # Reset upload dialog
                                    st.session_state.show_upload_dialog = False
                                    st.session_state.password_verified = False
                                    
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error(message)
                                    
                            except Exception as e:
                                st.error(f"❌ Unexpected error: {str(e)}")
                
                with col2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.show_upload_dialog = False
                        st.session_state.password_verified = False
                        st.rerun()
            
            else:
                if st.button("❌ Close", use_container_width=True):
                    st.session_state.show_upload_dialog = False
                    st.session_state.password_verified = False
                    st.rerun()
    
    st.divider()
    
    # Mathematical Mode Toggle
    st.subheader("🧮 Query Mode")
    math_mode = st.toggle(
        "Mathematical Mode (SQL Queries)",
        value=st.session_state.math_mode,
        help="Toggle between RAG mode (document search) and Mathematical mode (SQL database queries with smart visualization)"
    )
    
    if math_mode != st.session_state.math_mode:
        st.session_state.math_mode = math_mode
        if math_mode:
            st.success("✅ Switched to Mathematical Mode (SQL + Viz)")
        else:
            st.success("✅ Switched to RAG Mode (Documents)")
    
    # Show current mode
    current_mode = "🔢 Mathematical Mode (SQL + Viz)" if st.session_state.math_mode else "📚 RAG Mode (Documents)"
    st.info(f"**Current Mode:** {current_mode}")
    
    st.divider()
    
    # Top K slider (only for RAG mode)
    if not st.session_state.math_mode:
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Higher values provide more context but may include less relevant information"
        )
        
        if st.session_state.top_k != top_k:
            st.session_state.top_k = top_k
            with st.spinner("Updating retrieval settings..."):
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
    
    # Show index info (only in debug mode)
    if st.session_state.debug_mode:
        st.divider()
        st.subheader("📊 System Info")

        if st.session_state.math_mode:
            # Show database info
            try:
                conn = sqlite3.connect(':memory:')
                DB_IN_DF.to_sql('VARIETY_YIELD', conn, if_exists='replace', index=False)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM VARIETY_YIELD")
                count = cursor.fetchone()[0]
                conn.close()
                st.metric("Database Records", count)
                st.info("**Table:** VARIETY_YIELD\n**Fields:** Variety, Lint, Turnout, Micronaire, Length, Strength, Uniformity, LoanValue, LintValue, TrialLocation, Year")
            except Exception as e:
                st.warning(f"Database not found: {e}")
        else:
            # Show RAG info
            if "index" in st.session_state:
                try:
                    docstore = st.session_state.index.docstore
                    num_docs = len(docstore.docs)
                    st.metric("Total Documents/Chunks", num_docs)

                    st.info("**Embedding Model:**\nBAAI/bge-small-en-v1.5")
                    st.info("**LLM Model:**\nLlama 3.3 70B (Groq)")

                except Exception as e:
                    st.warning(f"Could not retrieve index stats: {e}")

        # API Key status
        st.divider()
        api_key_status = "✅ Set" if os.environ.get("GROQ_API_KEY") else "❌ Not Set"
        st.caption(f"GROQ_API_KEY: {api_key_status}")
