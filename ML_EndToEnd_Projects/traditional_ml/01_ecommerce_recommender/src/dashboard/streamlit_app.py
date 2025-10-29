"""
Streamlit dashboard for recommendation system monitoring and insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

# Configure page
st.set_page_config(
    page_title="E-commerce Recommendation Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #ff6b6b;
    }
    .success-card {
        border-left-color: #51cf66;
    }
    .warning-card {
        border-left-color: #ffd43b;
    }
    .info-card {
        border-left-color: #339af0;
    }
</style>
""", unsafe_allow_html=True)


# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 30  # seconds


@st.cache_data(ttl=30)
def fetch_api_data(endpoint: str):
    """Fetch data from API with caching."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from {endpoint}: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sample_data():
    """Load sample data for demonstration."""
    # Generate sample interaction data
    np.random.seed(42)
    n_users = 1000
    n_items = 500
    n_interactions = 10000
    
    users = range(1, n_users + 1)
    items = range(1, n_items + 1)
    
    data = []
    for _ in range(n_interactions):
        user_id = np.random.choice(users)
        item_id = np.random.choice(items)
        interaction_type = np.random.choice(['view', 'cart', 'purchase'], p=[0.7, 0.2, 0.1])
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3]) if interaction_type == 'purchase' else None
        timestamp = datetime.now() - timedelta(days=np.random.randint(0, 30))
        
        data.append({
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'timestamp': timestamp
        })
    
    return pd.DataFrame(data)


def main():
    """Main dashboard function."""
    st.title("🛒 E-commerce Recommendation Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        st.sidebar.write(f"Refreshing every {REFRESH_INTERVAL}s")
        time.sleep(REFRESH_INTERVAL)
        st.experimental_rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Fetch system health
    health_data = fetch_api_data("/health")
    stats_data = fetch_api_data("/stats")
    
    # Main content
    show_system_status(health_data, stats_data)
    show_recommendation_metrics(stats_data)
    show_user_analytics()
    show_item_analytics()
    show_real_time_testing()


def show_system_status(health_data, stats_data):
    """Show system status section."""
    st.header("🔧 System Status")
    
    if health_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if health_data.get('status') == 'healthy' else "🟡"
            st.markdown(f"""
            <div class="metric-card {'success-card' if health_data.get('status') == 'healthy' else 'warning-card'}">
                <h4>{status_color} System Status</h4>
                <h3>{health_data.get('status', 'Unknown').title()}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            model_status = "🟢" if health_data.get('model_loaded') else "🔴"
            st.markdown(f"""
            <div class="metric-card {'success-card' if health_data.get('model_loaded') else 'warning-card'}">
                <h4>{model_status} Models</h4>
                <h3>{'Loaded' if health_data.get('model_loaded') else 'Not Loaded'}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cache_status = "🟢" if health_data.get('cache_connected') else "🔴"
            st.markdown(f"""
            <div class="metric-card {'success-card' if health_data.get('cache_connected') else 'warning-card'}">
                <h4>{cache_status} Cache</h4>
                <h3>{'Connected' if health_data.get('cache_connected') else 'Disconnected'}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            uptime_hours = health_data.get('uptime_seconds', 0) / 3600
            st.markdown(f"""
            <div class="metric-card info-card">
                <h4>⏱️ Uptime</h4>
                <h3>{uptime_hours:.1f}h</h3>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("❌ API is not responding")


def show_recommendation_metrics(stats_data):
    """Show recommendation system metrics."""
    st.header("📊 Recommendation Metrics")
    
    if stats_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Users",
                value=f"{stats_data.get('total_users', 0):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Items",
                value=f"{stats_data.get('total_items', 0):,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Total Interactions",
                value=f"{stats_data.get('total_interactions', 0):,}",
                delta=None
            )
        
        with col4:
            # Calculate coverage
            coverage = 0
            if stats_data.get('total_users', 0) > 0 and stats_data.get('total_items', 0) > 0:
                coverage = (stats_data.get('total_interactions', 0) / 
                           (stats_data.get('total_users', 1) * stats_data.get('total_items', 1))) * 100
            
            st.metric(
                label="Coverage",
                value=f"{coverage:.2f}%",
                delta=None
            )
        
        # Model information
        model_info = stats_data.get('model_info', {})
        if model_info:
            st.subheader("Model Information")
            
            model_cols = st.columns(len(model_info))
            for i, (model_name, model_data) in enumerate(model_info.items()):
                with model_cols[i]:
                    st.info(f"**{model_name.title()}**\n\n"
                           f"Status: {'✅ Loaded' if model_data.get('loaded') else '❌ Not Loaded'}\n\n"
                           f"Last Updated: {model_data.get('last_updated', 'N/A')}")


def show_user_analytics():
    """Show user analytics section."""
    st.header("👥 User Analytics")
    
    # Load sample data for demonstration
    df = load_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User interaction distribution
        user_interactions = df.groupby('user_id').size().reset_index(name='interaction_count')
        
        fig = px.histogram(
            user_interactions, 
            x='interaction_count',
            title="User Interaction Distribution",
            labels={'interaction_count': 'Number of Interactions', 'count': 'Number of Users'},
            color_discrete_sequence=['#ff6b6b']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Interaction types over time
        df['date'] = df['timestamp'].dt.date
        interaction_trends = df.groupby(['date', 'interaction_type']).size().reset_index(name='count')
        
        fig = px.line(
            interaction_trends,
            x='date',
            y='count',
            color='interaction_type',
            title="Interaction Trends Over Time",
            labels={'date': 'Date', 'count': 'Number of Interactions'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # User segments
    st.subheader("User Segments")
    
    # Create user segments based on activity
    user_stats = df.groupby('user_id').agg({
        'interaction_type': 'count',
        'timestamp': ['min', 'max']
    }).round(2)
    
    user_stats.columns = ['total_interactions', 'first_interaction', 'last_interaction']
    user_stats['days_active'] = (user_stats['last_interaction'] - user_stats['first_interaction']).dt.days
    
    # Segment users
    def segment_user(row):
        if row['total_interactions'] >= 20:
            return 'Power User'
        elif row['total_interactions'] >= 10:
            return 'Regular User'
        elif row['total_interactions'] >= 5:
            return 'Occasional User'
        else:
            return 'New User'
    
    user_stats['segment'] = user_stats.apply(segment_user, axis=1)
    segment_counts = user_stats['segment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="User Segments Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top users table
        top_users = user_stats.nlargest(10, 'total_interactions')[['total_interactions', 'segment']]
        st.subheader("Top 10 Most Active Users")
        st.dataframe(top_users, use_container_width=True)


def show_item_analytics():
    """Show item analytics section."""
    st.header("📦 Item Analytics")
    
    df = load_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Item popularity distribution
        item_interactions = df.groupby('item_id').size().reset_index(name='interaction_count')
        
        fig = px.histogram(
            item_interactions,
            x='interaction_count',
            title="Item Popularity Distribution",
            labels={'interaction_count': 'Number of Interactions', 'count': 'Number of Items'},
            color_discrete_sequence=['#339af0']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Conversion funnel
        funnel_data = df['interaction_type'].value_counts().reset_index()
        funnel_data.columns = ['interaction_type', 'count']
        
        fig = go.Figure(go.Funnel(
            y=funnel_data['interaction_type'],
            x=funnel_data['count'],
            textinfo="value+percent initial"
        ))
        fig.update_layout(title="Interaction Funnel", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top items
    st.subheader("Top Performing Items")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Most Viewed Items**")
        most_viewed = df[df['interaction_type'] == 'view'].groupby('item_id').size().nlargest(10)
        st.dataframe(most_viewed.rename('views'), use_container_width=True)
    
    with col2:
        st.write("**Most Added to Cart**")
        most_carted = df[df['interaction_type'] == 'cart'].groupby('item_id').size().nlargest(10)
        st.dataframe(most_carted.rename('cart_adds'), use_container_width=True)
    
    with col3:
        st.write("**Most Purchased**")
        most_purchased = df[df['interaction_type'] == 'purchase'].groupby('item_id').size().nlargest(10)
        st.dataframe(most_purchased.rename('purchases'), use_container_width=True)


def show_real_time_testing():
    """Show real-time recommendation testing."""
    st.header("🧪 Real-time Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Get User Recommendations")
        
        user_id = st.number_input("User ID", min_value=1, max_value=1000, value=1)
        num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)
        model_type = st.selectbox("Model Type", ["hybrid", "collaborative", "content", "popularity"])
        
        if st.button("Get Recommendations"):
            try:
                with st.spinner("Fetching recommendations..."):
                    response = requests.post(
                        f"{API_BASE_URL}/recommend/user",
                        json={
                            "user_id": user_id,
                            "num_recommendations": num_recommendations,
                            "model_type": model_type,
                            "include_metadata": True
                        },
                        timeout=10
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Generated {len(data['recommendations'])} recommendations in {data['total_time_ms']:.1f}ms")
                    
                    # Display recommendations
                    recs_df = pd.DataFrame([
                        {
                            'Rank': rec['rank'],
                            'Item ID': rec['item_id'],
                            'Score': f"{rec['score']:.3f}"
                        }
                        for rec in data['recommendations']
                    ])
                    st.dataframe(recs_df, use_container_width=True)
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: {e}")
    
    with col2:
        st.subheader("Get Similar Items")
        
        item_id = st.number_input("Item ID", min_value=1, max_value=500, value=1)
        num_similar = st.slider("Number of Similar Items", 1, 20, 10)
        
        if st.button("Get Similar Items"):
            try:
                with st.spinner("Fetching similar items..."):
                    response = requests.post(
                        f"{API_BASE_URL}/recommend/item",
                        json={
                            "item_id": item_id,
                            "num_recommendations": num_similar,
                            "include_metadata": True
                        },
                        timeout=10
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Found {len(data)} similar items")
                    
                    # Display similar items
                    similar_df = pd.DataFrame([
                        {
                            'Rank': item['rank'],
                            'Item ID': item['item_id'],
                            'Similarity': f"{item['score']:.3f}"
                        }
                        for item in data
                    ])
                    st.dataframe(similar_df, use_container_width=True)
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: {e}")
    
    # Performance testing
    st.subheader("Performance Testing")
    
    if st.button("Run Performance Test"):
        with st.spinner("Running performance test..."):
            test_results = []
            
            for i in range(10):
                user_id = np.random.randint(1, 101)
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/recommend/user",
                        json={
                            "user_id": user_id,
                            "num_recommendations": 10,
                            "model_type": "hybrid"
                        },
                        timeout=5
                    )
                    
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000
                    
                    test_results.append({
                        'Request': i + 1,
                        'User ID': user_id,
                        'Status': response.status_code,
                        'Response Time (ms)': response_time
                    })
                
                except Exception as e:
                    test_results.append({
                        'Request': i + 1,
                        'User ID': user_id,
                        'Status': 'Error',
                        'Response Time (ms)': None
                    })
            
            # Display results
            results_df = pd.DataFrame(test_results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            successful_requests = results_df[results_df['Status'] == 200]
            if len(successful_requests) > 0:
                avg_response_time = successful_requests['Response Time (ms)'].mean()
                success_rate = len(successful_requests) / len(results_df) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col2:
                    st.metric("Avg Response Time", f"{avg_response_time:.1f}ms")
                with col3:
                    st.metric("Total Requests", len(results_df))


if __name__ == "__main__":
    main()