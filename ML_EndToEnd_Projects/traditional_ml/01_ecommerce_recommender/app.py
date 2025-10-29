"""
Streamlit dashboard for e-commerce recommendation system.
Interactive interface for exploring recommendations and system performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import time

from src.utils.config import get_config
from src.utils.logging import setup_logging

# Page configuration
st.set_page_config(
    page_title="E-commerce Recommender Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup
config = get_config()
logger = setup_logging(__name__)

# Constants
API_BASE_URL = f"http://{config.api_host}:{config.api_port}"
ALGORITHMS = ["hybrid", "collaborative", "content"]
INTERACTION_TYPES = ["view", "cart", "purchase", "rating"]


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sample_data():
    """Load sample data for demonstration."""
    try:
        # Try to load processed data
        users = pd.read_csv("data/processed/users.csv")
        items = pd.read_csv("data/processed/items.csv")
        interactions = pd.read_csv("data/processed/train.csv")
        return users, items, interactions
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return None, None, None


def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """Make API call to recommendation service."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
        return None


def get_recommendations(user_id: int, algorithm: str = "hybrid", num_recs: int = 10) -> Optional[List[Dict]]:
    """Get recommendations for a user."""
    data = {
        "user_id": user_id,
        "algorithm": algorithm,
        "num_recommendations": num_recs
    }
    
    result = call_api("recommend", "POST", data)
    return result.get("recommendations") if result else None


def get_similar_items(item_id: int, algorithm: str = "content", num_similar: int = 10) -> Optional[List[Dict]]:
    """Get similar items for an item."""
    data = {
        "item_id": item_id,
        "algorithm": algorithm,
        "num_similar": num_similar
    }
    
    result = call_api("similar", "POST", data)
    return result.get("similar_items") if result else None


def record_interaction(user_id: int, item_id: int, interaction_type: str, rating: Optional[float] = None):
    """Record user interaction."""
    data = {
        "user_id": user_id,
        "item_id": item_id,
        "interaction_type": interaction_type,
        "rating": rating
    }
    
    return call_api("interaction", "POST", data)


def main():
    """Main dashboard function."""
    st.title("🛒 E-commerce Recommender Dashboard")
    st.markdown("---")
    
    # Check API health
    health_status = call_api("health")
    if health_status:
        if health_status["status"] == "healthy":
            st.success(f"✅ API is healthy (Model: {health_status['model_status']}, Cache: {health_status['cache_status']})")
        else:
            st.warning(f"⚠️ API status: {health_status['status']}")
    else:
        st.error("❌ API is not responding")
        st.stop()
    
    # Load data
    users, items, interactions = load_sample_data()
    if users is None:
        st.error("Cannot load sample data. Please ensure data files are available.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Overview", "👤 User Recommendations", "📦 Item Similarity", "📊 Analytics", "⚙️ System Stats"]
    )
    
    if page == "🏠 Overview":
        show_overview(users, items, interactions)
    elif page == "👤 User Recommendations":
        show_user_recommendations(users, items)
    elif page == "📦 Item Similarity":
        show_item_similarity(items)
    elif page == "📊 Analytics":
        show_analytics(users, items, interactions)
    elif page == "⚙️ System Stats":
        show_system_stats()


def show_overview(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame):
    """Show system overview."""
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{len(users):,}")
    with col2:
        st.metric("Total Items", f"{len(items):,}")
    with col3:
        st.metric("Total Interactions", f"{len(interactions):,}")
    with col4:
        avg_rating = interactions[interactions['rating'] > 0]['rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A")
    
    st.markdown("---")
    
    # Data distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Demographics")
        if 'age' in users.columns:
            fig = px.histogram(users, x='age', title="Age Distribution", nbins=20)
            st.plotly_chart(fig, use_container_width=True)
        
        if 'gender' in users.columns:
            gender_counts = users['gender'].value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, title="Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Item Analytics")
        if 'category' in items.columns:
            category_counts = items['category'].value_counts().head(10)
            fig = px.bar(x=category_counts.values, y=category_counts.index, 
                        orientation='h', title="Top Categories")
            st.plotly_chart(fig, use_container_width=True)
        
        if 'price' in items.columns:
            fig = px.histogram(items, x='price', title="Price Distribution", nbins=50)
            fig.update_xaxis(range=[0, items['price'].quantile(0.95)])
            st.plotly_chart(fig, use_container_width=True)
    
    # Interaction patterns
    st.subheader("Interaction Patterns")
    if 'interaction_type' in interactions.columns:
        interaction_counts = interactions['interaction_type'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=interaction_counts.values, names=interaction_counts.index,
                        title="Interaction Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time series of interactions
            if 'timestamp' in interactions.columns:
                interactions_copy = interactions.copy()
                interactions_copy['timestamp'] = pd.to_datetime(interactions_copy['timestamp'])
                interactions_copy['date'] = interactions_copy['timestamp'].dt.date
                
                daily_interactions = interactions_copy.groupby('date').size().reset_index(name='count')
                fig = px.line(daily_interactions, x='date', y='count', title="Daily Interactions")
                st.plotly_chart(fig, use_container_width=True)


def show_user_recommendations(users: pd.DataFrame, items: pd.DataFrame):
    """Show user recommendation interface."""
    st.header("👤 User Recommendations")
    
    # User selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_id = st.selectbox(
            "Select User ID",
            options=sorted(users['user_id'].unique()[:1000]),  # Limit for performance
            help="Select a user to get recommendations"
        )
    
    with col2:
        algorithm = st.selectbox(
            "Algorithm",
            options=ALGORITHMS,
            index=0,
            help="Choose recommendation algorithm"
        )
    
    with col3:
        num_recs = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of recommendations to generate"
        )
    
    # User profile
    if user_id:
        user_info = users[users['user_id'] == user_id].iloc[0]
        
        st.subheader(f"User Profile: {user_id}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'age' in user_info:
                st.write(f"**Age:** {user_info['age']}")
        with col2:
            if 'gender' in user_info:
                st.write(f"**Gender:** {user_info['gender']}")
        with col3:
            if 'location' in user_info:
                st.write(f"**Location:** {user_info['location']}")
    
    # Generate recommendations
    if st.button("🎯 Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommendations = get_recommendations(user_id, algorithm, num_recs)
        
        if recommendations:
            st.subheader("📋 Recommendations")
            
            # Display recommendations
            rec_df = pd.DataFrame(recommendations)
            
            # Add item details if available
            if 'item_id' in rec_df.columns and not items.empty:
                rec_df = rec_df.merge(
                    items[['item_id', 'category', 'brand', 'price']].head(1000), 
                    on='item_id', 
                    how='left'
                )
            
            # Format and display
            for idx, rec in rec_df.iterrows():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.write(f"**Item {rec['item_id']}**")
                    if 'category' in rec:
                        st.write(f"Category: {rec['category']}")
                
                with col2:
                    if 'brand' in rec:
                        st.write(f"Brand: {rec['brand']}")
                    if 'price' in rec:
                        st.write(f"Price: ${rec['price']:.2f}")
                
                with col3:
                    st.write(f"Score: {rec['score']:.3f}")
                
                with col4:
                    # Record interaction buttons
                    if st.button(f"👁️", key=f"view_{rec['item_id']}", help="Mark as viewed"):
                        record_interaction(user_id, rec['item_id'], "view")
                        st.success("Interaction recorded!")
                
                st.markdown("---")
        else:
            st.error("Could not generate recommendations")
    
    # Recent interactions
    st.subheader("📈 Recent Interactions")
    # This would show user's recent interactions from the database


def show_item_similarity(items: pd.DataFrame):
    """Show item similarity interface."""
    st.header("📦 Item Similarity")
    
    # Item selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        item_id = st.selectbox(
            "Select Item ID",
            options=sorted(items['item_id'].unique()[:1000]),  # Limit for performance
            help="Select an item to find similar items"
        )
    
    with col2:
        algorithm = st.selectbox(
            "Algorithm",
            options=["content", "collaborative"],
            help="Choose similarity algorithm"
        )
    
    with col3:
        num_similar = st.slider(
            "Number of Similar Items",
            min_value=5,
            max_value=20,
            value=10
        )
    
    # Item details
    if item_id:
        item_info = items[items['item_id'] == item_id].iloc[0]
        
        st.subheader(f"Item Details: {item_id}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'category' in item_info:
                st.write(f"**Category:** {item_info['category']}")
        with col2:
            if 'brand' in item_info:
                st.write(f"**Brand:** {item_info['brand']}")
        with col3:
            if 'price' in item_info:
                st.write(f"**Price:** ${item_info['price']:.2f}")
    
    # Find similar items
    if st.button("🔍 Find Similar Items", type="primary"):
        with st.spinner("Finding similar items..."):
            similar_items = get_similar_items(item_id, algorithm, num_similar)
        
        if similar_items:
            st.subheader("🔗 Similar Items")
            
            similar_df = pd.DataFrame(similar_items)
            
            # Add item details
            if not items.empty:
                similar_df = similar_df.merge(
                    items[['item_id', 'category', 'brand', 'price']].head(1000),
                    on='item_id',
                    how='left'
                )
            
            # Display similar items
            for idx, item in similar_df.iterrows():
                col1, col2, col3 = st.columns([3, 3, 2])
                
                with col1:
                    st.write(f"**Item {item['item_id']}**")
                    if 'category' in item:
                        st.write(f"Category: {item['category']}")
                
                with col2:
                    if 'brand' in item:
                        st.write(f"Brand: {item['brand']}")
                    if 'price' in item:
                        st.write(f"Price: ${item['price']:.2f}")
                
                with col3:
                    st.write(f"Similarity: {item['similarity']:.3f}")
                
                st.markdown("---")
        else:
            st.error("Could not find similar items")


def show_analytics(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame):
    """Show system analytics and insights."""
    st.header("📊 Analytics & Insights")
    
    # Algorithm comparison
    st.subheader("🔄 Algorithm Performance Comparison")
    
    # Sample users for testing
    sample_users = st.multiselect(
        "Select users for algorithm comparison",
        options=sorted(users['user_id'].unique()[:100]),
        default=sorted(users['user_id'].unique()[:5])
    )
    
    if sample_users and st.button("🧪 Run Algorithm Comparison"):
        results = []
        
        progress_bar = st.progress(0)
        for i, user_id in enumerate(sample_users):
            for algorithm in ALGORITHMS:
                with st.spinner(f"Testing {algorithm} for user {user_id}..."):
                    start_time = time.time()
                    recs = get_recommendations(user_id, algorithm, 10)
                    end_time = time.time()
                    
                    if recs:
                        results.append({
                            'user_id': user_id,
                            'algorithm': algorithm,
                            'response_time': (end_time - start_time) * 1000,
                            'num_recommendations': len(recs),
                            'avg_score': np.mean([r['score'] for r in recs])
                        })
            
            progress_bar.progress((i + 1) / len(sample_users))
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Response time comparison
                fig = px.box(results_df, x='algorithm', y='response_time',
                           title="Response Time by Algorithm (ms)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average score comparison
                fig = px.box(results_df, x='algorithm', y='avg_score',
                           title="Average Recommendation Score by Algorithm")
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Performance Summary")
            summary = results_df.groupby('algorithm').agg({
                'response_time': ['mean', 'std'],
                'avg_score': ['mean', 'std'],
                'num_recommendations': 'mean'
            }).round(3)
            
            st.dataframe(summary)
    
    # User behavior analysis
    st.subheader("👥 User Behavior Analysis")
    
    if 'timestamp' in interactions.columns:
        interactions_copy = interactions.copy()
        interactions_copy['timestamp'] = pd.to_datetime(interactions_copy['timestamp'])
        interactions_copy['hour'] = interactions_copy['timestamp'].dt.hour
        interactions_copy['day_of_week'] = interactions_copy['timestamp'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly activity
            hourly_activity = interactions_copy.groupby('hour').size()
            fig = px.bar(x=hourly_activity.index, y=hourly_activity.values,
                        title="User Activity by Hour of Day")
            fig.update_xaxis(title="Hour")
            fig.update_yaxis(title="Number of Interactions")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily activity
            daily_activity = interactions_copy.groupby('day_of_week').size()
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_activity = daily_activity.reindex(day_order)
            
            fig = px.bar(x=daily_activity.index, y=daily_activity.values,
                        title="User Activity by Day of Week")
            fig.update_xaxis(title="Day of Week")
            fig.update_yaxis(title="Number of Interactions")
            st.plotly_chart(fig, use_container_width=True)


def show_system_stats():
    """Show system statistics and health metrics."""
    st.header("⚙️ System Statistics")
    
    # Get system stats from API
    stats = call_api("stats")
    
    if stats:
        # Model performance
        if 'model_performance' in stats:
            st.subheader("🎯 Model Performance")
            perf = stats['model_performance']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision@10", f"{perf.get('precision_10', 0):.3f}")
            with col2:
                st.metric("Recall@10", f"{perf.get('recall_10', 0):.3f}")
            with col3:
                st.metric("NDCG@10", f"{perf.get('ndcg_10', 0):.3f}")
        
        # Cache performance
        if 'cache' in stats:
            st.subheader("💾 Cache Performance")
            cache = stats['cache']
            
            if 'keyspace_hits' in cache and 'keyspace_misses' in cache:
                hits = cache['keyspace_hits']
                misses = cache['keyspace_misses']
                hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cache Hit Rate", f"{hit_rate:.1%}")
                with col2:
                    st.metric("Cache Hits", f"{hits:,}")
                with col3:
                    st.metric("Cache Misses", f"{misses:,}")
            
            if 'used_memory' in cache:
                st.metric("Memory Usage", cache['used_memory'])
        
        # System metrics
        if 'system' in stats:
            st.subheader("🖥️ System Metrics")
            system = stats['system']
            
            col1, col2 = st.columns(2)
            with col1:
                if 'cpu_usage' in system:
                    st.metric("CPU Usage", f"{system['cpu_usage']:.1f}%")
                if 'memory_usage' in system:
                    st.metric("Memory Usage", f"{system['memory_usage']:.1f}%")
            
            with col2:
                if 'requests_per_second' in system:
                    st.metric("Requests/Second", f"{system['requests_per_second']:.1f}")
                if 'avg_response_time' in system:
                    st.metric("Avg Response Time", f"{system['avg_response_time']:.1f}ms")
    
    # Real-time monitoring
    st.subheader("📊 Real-time Monitoring")
    
    if st.checkbox("Enable Real-time Updates"):
        placeholder = st.empty()
        
        while True:
            current_stats = call_api("stats")
            if current_stats:
                with placeholder.container():
                    st.write(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Quick health check
                    health = call_api("health")
                    if health:
                        status = "🟢" if health['status'] == 'healthy' else "🔴"
                        st.write(f"**System Status:** {status} {health['status']}")
            
            time.sleep(5)  # Update every 5 seconds


if __name__ == "__main__":
    main()