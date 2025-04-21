"""
Analytics dashboard admin page UI for PharmInsight.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from ...auth.auth import is_admin
from ...config import DB_PATH
from ...qa.feedback import get_feedback_stats
from ...qa.history import get_popular_searches

def admin_analytics_page():
    """Admin analytics dashboard page"""
    if not is_admin():
        st.warning("Admin access required")
        return
        
    st.title("Analytics Dashboard")
    
    # Create tabs for different analytics sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Usage Statistics", 
        "💬 Feedback Analysis",
        "🔍 Search Patterns", 
        "📑 Document Analytics"
    ])
    
    # Tab 1: Usage Statistics
    with tab1:
        st.subheader("System Usage")
        
        conn = sqlite3.connect(DB_PATH)
        
        # Get counts from different tables
        query_counts = pd.read_sql_query(
            """
            SELECT 
                (SELECT COUNT(*) FROM users) as users_count,
                (SELECT COUNT(*) FROM documents) as documents_count,
                (SELECT COUNT(*) FROM chunks) as chunks_count,
                (SELECT COUNT(*) FROM qa_pairs) as qa_count,
                (SELECT COUNT(*) FROM search_history) as searches_count,
                (SELECT COUNT(*) FROM feedback) as feedback_count
            """,
            conn
        )
        
        # Get recent activity
        recent_activity = pd.read_sql_query(
            """
            SELECT action, user_id, timestamp, details
            FROM audit_log
            ORDER BY timestamp DESC
            LIMIT 50
            """,
            conn
        )
        
        # Get user activity over time
        user_activity = pd.read_sql_query(
            """
            SELECT 
                date(timestamp) as date,
                COUNT(*) as count,
                action
            FROM audit_log
            WHERE date(timestamp) > date('now', '-30 days')
            GROUP BY date(timestamp), action
            ORDER BY date
            """,
            conn
        )
        
        conn.close()
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", query_counts['users_count'].iloc[0])
            st.metric("Total Documents", query_counts['documents_count'].iloc[0])
            
        with col2:
            st.metric("Document Chunks", query_counts['chunks_count'].iloc[0])
            st.metric("Total Queries", query_counts['qa_count'].iloc[0])
            
        with col3:
            st.metric("Search Count", query_counts['searches_count'].iloc[0])
            st.metric("Feedback Collected", query_counts['feedback_count'].iloc[0])
        
        # Activity over time visualization
        if not user_activity.empty:
            st.subheader("Activity Over Time")
            fig = px.line(
                user_activity,
                x="date",
                y="count",
                color="action",
                title="User Activity (Last 30 Days)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Recent activity table
        st.subheader("Recent Activity")
        if not recent_activity.empty:
            # Format timestamp
            recent_activity['timestamp'] = pd.to_datetime(recent_activity['timestamp'])
            recent_activity['formatted_time'] = recent_activity['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                recent_activity[['formatted_time', 'user_id', 'action', 'details']],
                column_config={
                    "formatted_time": st.column_config.TextColumn("Time"),
                    "user_id": st.column_config.TextColumn("User"),
                    "action": st.column_config.TextColumn("Action"),
                    "details": st.column_config.TextColumn("Details")
                },
                use_container_width=True
            )
        else:
            st.info("No recent activity recorded")
    
    # Tab 2: Feedback Analysis
    with tab2:
        st.subheader("Feedback Analysis")
        
        stats_df, rating_dist, recent_feedback = get_feedback_stats()
        
        if not stats_df.empty:
            avg_rating = stats_df['avg_rating'].iloc[0]
            total_feedback = stats_df['total_feedback'].iloc[0]
            
            st.metric("Average Rating", f"{avg_rating:.2f}/3.00")
            st.metric("Total Feedback Collected", total_feedback)
            
            # Rating distribution visualization
            if not rating_dist.empty:
                st.subheader("Rating Distribution")
                
                # Convert ratings to descriptive labels
                rating_labels = {
                    1: "Not Helpful",
                    2: "Somewhat Helpful",
                    3: "Very Helpful"
                }
                
                rating_dist['rating_label'] = rating_dist['rating'].map(rating_labels)
                
                fig = px.pie(
                    rating_dist,
                    values='count',
                    names='rating_label',
                    title="Feedback Distribution",
                    color='rating_label',
                    color_discrete_map={
                        "Not Helpful": "#ff6961",
                        "Somewhat Helpful": "#fdfd96",
                        "Very Helpful": "#77dd77"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Recent feedback
            st.subheader("Recent Feedback")
            if not recent_feedback.empty:
                for _, row in recent_feedback.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        rating_emoji = {
                            1: "❌ Not Helpful",
                            2: "⚠️ Somewhat Helpful",
                            3: "✅ Very Helpful"
                        }
                        
                        with col1:
                            st.write(f"**Query:** {row['query']}")
                            if row['comment']:
                                st.write(f"**Comment:** {row['comment']}")
                        
                        with col2:
                            st.write(f"**Rating:** {rating_emoji.get(row['rating'])}")
                            st.caption(f"{row['timestamp'][:10]}")
                        
                        st.divider()
            else:
                st.info("No feedback collected yet")
        else:
            st.info("No feedback data available")
    
    # Tab 3: Search Patterns
    with tab3:
        st.subheader("Search Patterns")
        
        popular_searches = get_popular_searches(limit=20)
        
        # Get search success rate
        conn = sqlite3.connect(DB_PATH)
        search_success = pd.read_sql_query(
            """
            SELECT 
                date(timestamp) as date,
                COUNT(*) as total_searches,
                SUM(CASE WHEN num_results > 0 THEN 1 ELSE 0 END) as successful_searches
            FROM search_history
            WHERE date(timestamp) > date('now', '-30 days')
            GROUP BY date(timestamp)
            ORDER BY date
            """,
            conn
        )
        
        # Query length analysis
        query_length = pd.read_sql_query(
            """
            SELECT 
                length(query) as query_length,
                COUNT(*) as count
            FROM search_history
            GROUP BY length(query)
            ORDER BY length(query)
            """,
            conn
        )
        
        conn.close()
        
        # Popular searches
        st.subheader("Popular Searches")
        if not popular_searches.empty:
            fig = px.bar(
                popular_searches,
                x='count',
                y='query',
                title="Most Common Searches",
                orientation='h'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No search data available")
            
        # Search success rate
        if not search_success.empty:
            st.subheader("Search Success Rate")
            
            # Calculate success rate
            search_success['success_rate'] = (
                search_success['successful_searches'] / search_success['total_searches']
            ) * 100
            
            fig = px.line(
                search_success,
                x='date',
                y='success_rate',
                title="Search Success Rate (%)",
                markers=True
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
        # Query length distribution
        if not query_length.empty:
            st.subheader("Query Length Distribution")
            
            fig = px.histogram(
                query_length,
                x='query_length',
                y='count',
                title="Distribution of Query Lengths",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Document Analytics
    with tab4:
        st.subheader("Document Analytics")
        
        conn = sqlite3.connect(DB_PATH)
        
        # Document categories
        doc_categories = pd.read_sql_query(
            """
            SELECT 
                category,
                COUNT(*) as count
            FROM documents
            GROUP BY category
            ORDER BY count DESC
            """,
            conn
        )
        
        # Document usage (which documents are cited in answers)
        doc_usage = pd.read_sql_query(
            """
            WITH extracted_sources AS (
                SELECT 
                    question_id,
                    json_each.value as source
                FROM qa_pairs, json_each(json_extract(sources, '$'))
                WHERE sources IS NOT NULL
            )
            SELECT 
                s.source,
                COUNT(*) as usage_count
            FROM extracted_sources s
            GROUP BY s.source
            ORDER BY usage_count DESC
            LIMIT 15
            """,
            conn,
            params=()
        )
        
        # Document upload timeline
        doc_timeline = pd.read_sql_query(
            """
            SELECT 
                date(upload_date) as upload_date,
                COUNT(*) as count
            FROM documents
            GROUP BY date(upload_date)
            ORDER BY upload_date
            """,
            conn
        )
        
        conn.close()
        
        # Document categories
        st.subheader("Document Categories")
        if not doc_categories.empty:
            fig = px.pie(
                doc_categories,
                values='count',
                names='category',
                title="Documents by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No document category data available")
            
        # Document usage
        st.subheader("Most Cited Documents")
        if not doc_usage.empty:
            fig = px.bar(
                doc_usage,
                x='usage_count',
                y='source',
                title="Most Frequently Cited Documents",
                orientation='h'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No document citation data available")
            
        # Document upload timeline
        if not doc_timeline.empty:
            st.subheader("Document Upload Timeline")
            
            fig = px.line(
                doc_timeline,
                x='upload_date',
                y='count',
                title="Document Uploads Over Time",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
