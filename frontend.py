import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Configure the page
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with more styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.6rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2e86ab;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #3498db;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_connection():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        return True
    except:
        return False


def create_feature_explanation():
    """Create feature explanation section"""
    st.markdown('<div class="subsection-header">📋 Feature Descriptions</div>', unsafe_allow_html=True)

    features = {
        "Hours Studied": "Number of hours the student studies per day (1-10 hours)",
        "Previous Scores": "Average score from previous exams (40-100 marks)",
        "Attendance": "Percentage of classes attended (60-100%)",
        "Extracurricular": "Hours spent on extracurricular activities (0-5 hours)",
        "Parental Education": "Education level of parents (1=High School to 5=PhD)"
    }

    for feature, description in features.items():
        with st.expander(f"🎯 {feature}"):
            st.write(description)


def create_correlation_analysis(df):
    """Create correlation analysis section"""
    st.markdown('<div class="subsection-header">🔗 Feature Correlations</div>', unsafe_allow_html=True)

    # Calculate correlations
    corr_matrix = df.corr()

    # Create heatmap with updated configuration
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix Heatmap"
    )
    fig.update_layout(height=500)

    # Use config instead of deprecated kwargs
    st.plotly_chart(fig, use_container_width=True,
                    config={'displayModeBar': True, 'displaylogo': False})

    # Most correlated features with final score
    final_score_corr = corr_matrix['final_score'].sort_values(ascending=False)
    st.write("**Correlation with Final Score:**")
    for feature, corr in final_score_corr.items():
        if feature != 'final_score':
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"📊 {feature}")
            with col2:
                st.write(f"`{corr:.3f}`")


def create_feature_distributions(df):
    """Create feature distribution visualizations"""
    st.markdown('<div class="subsection-header">📈 Feature Distributions</div>', unsafe_allow_html=True)

    # Select features to visualize (exclude target for now)
    features = ['hours_studied', 'previous_scores', 'attendance', 'extracurricular', 'parental_education']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f.replace('_', ' ').title() for f in features],
        specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]]
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, feature in enumerate(features):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Histogram(x=df[feature], name=feature, marker_color=colors[i]),
            row=row, col=col
        )

    fig.update_layout(height=600, showlegend=False, title_text="Feature Distributions")

    # Use config instead of deprecated kwargs
    st.plotly_chart(fig, use_container_width=True,
                    config={'displayModeBar': True, 'displaylogo': False})


def create_model_insights():
    """Create model insights and explanations"""
    st.markdown('<div class="subsection-header">🤔 Model Insights</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Linear Regression:**
        - ✅ Simple and interpretable
        - ✅ Fast training and prediction
        - ✅ Works well with linear relationships
        - ❌ Cannot capture complex patterns
        - ❌ Assumes linear relationships
        """)

    with col2:
        st.markdown("""
        **Decision Tree:**
        - ✅ Captures non-linear relationships
        - ✅ Easy to visualize and understand
        - ✅ Handles mixed data types well
        - ❌ Can overfit easily
        - ❌ Sensitive to small data changes
        """)


def create_student_profile_analysis():
    """Create interactive student profile analysis"""
    st.markdown('<div class="subsection-header">👤 Student Profile Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        profile_type = st.selectbox(
            "Select Student Profile",
            ["High Performer", "Average Student", "Struggling Student", "Custom Profile"]
        )

    with col2:
        if profile_type == "High Performer":
            st.info("High performers typically have: High study hours, Good previous scores, High attendance")
        elif profile_type == "Average Student":
            st.info("Average students: Moderate study hours, Average previous scores, Good attendance")
        elif profile_type == "Struggling Student":
            st.warning("Struggling students: Low study hours, Poor previous scores, Low attendance")
        else:
            st.info("Adjust the sliders to create a custom student profile")


def main():
    # Header with project info
    st.markdown('<div class="main-header">🎓 Student Exam Score Prediction System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;'>
    Predict student performance using Machine Learning | Compare Linear Regression vs Decision Tree
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Check API connection
    if not check_api_connection():
        st.error("⚠️ Backend API is not running! Please start the FastAPI server first.")
        st.info("""
        **To fix this:**
        1. Open a terminal and run: `python app.py`
        2. Make sure the server is running on http://localhost:8000
        3. Refresh this page
        """)
        return

    # Enhanced sidebar
    with st.sidebar:
        st.markdown("![Logo](https://cdn-icons-png.flaticon.com/512/2997/2997892.png)")
        st.title("Navigation")
        page = st.radio(
            "Choose a section:",
            ["📊 Data Explorer", "🤖 Model Training", "🔮 Prediction Lab", "📈 Performance Analytics", "ℹ️ Project Info"]
        )

        st.markdown("---")
        st.markdown("### Quick Stats")
        try:
            response = requests.get(f"{API_BASE_URL}/data", params={"sample_size": 1})
            if response.status_code == 200:
                data = response.json()
                st.write(f"**Dataset Size:** {data['total_students']} students")
                st.write(f"**Features:** {len(data['columns']) - 1}")
        except:
            st.write("**Dataset:** Loading...")

        st.markdown("---")
        st.markdown("""
        **Project By:**  
        Gaurav Kumar  
        iitrprai_24102148
        """)

    # Data Explorer Page
    if page == "📊 Data Explorer":
        st.markdown('<div class="section-header">📊 Comprehensive Data Explorer</div>', unsafe_allow_html=True)

        # Create tabs for different data views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Overview", "🔗 Correlations", "📈 Distributions", "📋 Feature Info"])

        with tab1:
            col1, col2 = st.columns([3, 1])

            with col2:
                st.subheader("Data Controls")
                sample_size = st.slider("Sample size to display", 5, 100, 20)
                if st.button("🔄 Generate New Dataset", use_container_width=True):
                    with st.spinner("Generating new synthetic dataset..."):
                        response = requests.post(f"{API_BASE_URL}/generate_new_data", params={"num_students": 500})
                        if response.status_code == 200:
                            st.success("New dataset generated successfully!")
                            time.sleep(1)
                            st.rerun()

                # Quick statistics
                try:
                    response = requests.get(f"{API_BASE_URL}/data", params={"sample_size": 1})
                    if response.status_code == 200:
                        data = response.json()
                        st.metric("Total Students", data['total_students'])
                        st.metric("Number of Features", len(data['columns']) - 1)
                except:
                    pass

            with col1:
                try:
                    response = requests.get(f"{API_BASE_URL}/data", params={"sample_size": sample_size})
                    if response.status_code == 200:
                        data = response.json()
                        df = pd.DataFrame(data['sample_data'])

                        st.subheader("Student Data Sample")
                        st.dataframe(
                            df.style
                            .background_gradient(subset=['final_score'], cmap='YlOrBr')
                            .format("{:.1f}"),
                            use_container_width=True,
                            height=400
                        )

                        # Summary statistics
                        st.subheader("📊 Detailed Statistics")
                        stats_df = df.describe()
                        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

        with tab2:
            try:
                response = requests.get(f"{API_BASE_URL}/data", params={"sample_size": 500})
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data['sample_data'])
                    create_correlation_analysis(df)
            except Exception as e:
                st.error(f"Error in correlation analysis: {str(e)}")

        with tab3:
            try:
                response = requests.get(f"{API_BASE_URL}/data", params={"sample_size": 500})
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data['sample_data'])
                    create_feature_distributions(df)
            except Exception as e:
                st.error(f"Error creating distributions: {str(e)}")

        with tab4:
            create_feature_explanation()

    # Model Training Page
    elif page == "🤖 Model Training":
        st.markdown('<div class="section-header">🤖 Advanced Model Training</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🎯 Training Configuration</h4>
            Adjust the parameters below to see how they affect model performance. 
            Different settings can lead to better or worse predictions!
            </div>
            """, unsafe_allow_html=True)

            # Training parameters in expanders
            with st.expander("🔧 Basic Training Settings", expanded=True):
                col3, col4 = st.columns(2)
                with col3:
                    test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05,
                                          help="Proportion of data to use for testing")
                with col4:
                    random_state = st.number_input("Random State", 0, 100, 42,
                                                   help="Seed for reproducible results")

            with st.expander("🌳 Decision Tree Parameters", expanded=True):
                col5, col6 = st.columns(2)
                with col5:
                    max_depth = st.slider("Max Depth", 2, 20, 5,
                                          help="Maximum depth of the tree")
                with col6:
                    min_samples_split = st.slider("Min Samples Split", 2, 20, 2,
                                                  help="Minimum samples required to split a node")

        with col2:
            st.subheader("Training Control")
            if st.button("🚀 Train Both Models", type="primary", use_container_width=True):
                training_config = {
                    "test_size": test_size,
                    "random_state": random_state,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split
                }

                with st.spinner("Training models... This may take a few seconds."):
                    response = requests.post(f"{API_BASE_URL}/train", json=training_config)

                    if response.status_code == 200:
                        st.success("✅ Models trained successfully!")
                        results = response.json()

                        # Display results in nice cards
                        st.balloons()

                        col7, col8 = st.columns(2)

                        with col7:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>📈 Linear Regression</h3>
                                <h2>R²: {results['linear_regression_metrics']['r2_score']}</h2>
                                <h4>MAE: {results['linear_regression_metrics']['mae']}</h4>
                            </div>
                            """, unsafe_allow_html=True)

                        with col8:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>🌳 Decision Tree</h3>
                                <h2>R²: {results['decision_tree_metrics']['r2_score']}</h2>
                                <h4>MAE: {results['decision_tree_metrics']['mae']}</h4>
                            </div>
                            """, unsafe_allow_html=True)

                        st.write(f"**Training Data:** {results['training_samples']} students")
                        st.write(f"**Testing Data:** {results['testing_samples']} students")

                    else:
                        st.error("❌ Training failed! Check if the backend is running properly.")

        create_model_insights()

    # Prediction Lab Page
    elif page == "🔮 Prediction Lab":
        st.markdown('<div class="section-header">🔮 Interactive Prediction Lab</div>', unsafe_allow_html=True)

        create_student_profile_analysis()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Student Details")

            # Interactive input sliders with better formatting
            hours_studied = st.slider("Hours Studied per Day", 1.0, 10.0, 5.0, 0.5,
                                      help="Daily study hours (1-10 hours)")
            previous_scores = st.slider("Previous Exam Scores", 40.0, 100.0, 75.0, 1.0,
                                        help="Average of previous exam scores (40-100)")
            attendance = st.slider("Attendance Percentage", 60.0, 100.0, 85.0, 1.0,
                                   help="Class attendance rate (60-100%)")

        with col2:
            st.subheader("🎯 Additional Factors")

            extracurricular = st.slider("Extracurricular Hours", 0.0, 5.0, 2.0, 0.5,
                                        help="Time spent on extracurricular activities (0-5 hours)")
            parental_education = st.selectbox(
                "Parental Education Level",
                options=[1, 2, 3, 4, 5],
                index=2,
                format_func=lambda x: {
                    1: "🎓 High School",
                    2: "🎓 Some College",
                    3: "🎓 Bachelor's Degree",
                    4: "🎓 Master's Degree",
                    5: "🎓 PhD"
                }[x],
                help="Highest education level of parents"
            )

            st.markdown("---")
            if st.button("🎯 Predict Exam Score", type="primary", use_container_width=True):
                student_data = {
                    "hours_studied": hours_studied,
                    "previous_scores": previous_scores,
                    "attendance": attendance,
                    "extracurricular": extracurricular,
                    "parental_education": parental_education
                }

                try:
                    response = requests.post(f"{API_BASE_URL}/predict", json=student_data)

                    if response.status_code == 200:
                        prediction = response.json()

                        # Enhanced prediction display
                        col3, col4 = st.columns(2)

                        with col3:
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>📈 Linear Regression</h3>
                                <h1>{prediction['linear_regression_prediction']}</h1>
                                <p>Predicted Score</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>🌳 Decision Tree</h3>
                                <h1>{prediction['decision_tree_prediction']}</h1>
                                <p>Predicted Score</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Prediction comparison gauge
                        st.subheader("📊 Prediction Comparison")

                        fig = go.Figure()

                        fig.add_trace(go.Indicator(
                            mode="number+gauge+delta",
                            value=prediction['linear_regression_prediction'],
                            delta={'reference': prediction['decision_tree_prediction']},
                            domain={'x': [0, 0.45], 'y': [0, 1]},
                            title={'text': "Linear Regression"},
                            gauge={
                                'shape': "bullet",
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "blue"}
                            }
                        ))

                        fig.add_trace(go.Indicator(
                            mode="number+gauge+delta",
                            value=prediction['decision_tree_prediction'],
                            delta={'reference': prediction['linear_regression_prediction']},
                            domain={'x': [0.55, 1], 'y': [0, 1]},
                            title={'text': "Decision Tree"},
                            gauge={
                                'shape': "bullet",
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "orange"}
                            }
                        ))

                        fig.update_layout(height=250, margin=dict(t=50, b=10))

                        # Use config instead of deprecated kwargs
                        st.plotly_chart(fig, use_container_width=True,
                                        config={'displayModeBar': True, 'displaylogo': False})

                    else:
                        st.error("❌ Prediction failed! Please train the models first.")

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

    # Performance Analytics Page
    elif page == "📈 Performance Analytics":
        st.markdown('<div class="section-header">📈 Model Performance Analytics</div>', unsafe_allow_html=True)

        try:
            response = requests.get(f"{API_BASE_URL}/metrics")

            if response.status_code == 200:
                metrics = response.json()

                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Linear R² Score", f"{metrics['linear_regression']['r2_score']}")
                with col2:
                    st.metric("Linear MAE", f"{metrics['linear_regression']['mae']}")
                with col3:
                    st.metric("Tree R² Score", f"{metrics['decision_tree']['r2_score']}")
                with col4:
                    st.metric("Tree MAE", f"{metrics['decision_tree']['mae']}")

                # Performance comparison charts
                st.subheader("📊 Performance Comparison")

                models = ['Linear Regression', 'Decision Tree']
                r2_scores = [metrics['linear_regression']['r2_score'], metrics['decision_tree']['r2_score']]
                mae_scores = [metrics['linear_regression']['mae'], metrics['decision_tree']['mae']]

                # Create comparison charts
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('R² Score (Higher is better)', 'MAE Score (Lower is better)'),
                    specs=[[{"type": "bar"}, {"type": "bar"}]]
                )

                fig.add_trace(
                    go.Bar(name='R² Score', x=models, y=r2_scores,
                           marker_color=['#1f77b4', '#ff7f0e'],
                           text=r2_scores, textposition='auto'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(name='MAE', x=models, y=mae_scores,
                           marker_color=['#1f77b4', '#ff7f0e'],
                           text=mae_scores, textposition='auto'),
                    row=1, col=2
                )

                fig.update_layout(height=400, showlegend=False)

                # Use config instead of deprecated kwargs
                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': True, 'displaylogo': False})

                # Model recommendation
                st.subheader("🎯 Model Recommendation")

                if metrics['decision_tree']['r2_score'] > metrics['linear_regression']['r2_score']:
                    st.success("""
                    **🌟 Recommendation: Use Decision Tree Model**
                    - The Decision Tree shows better predictive performance (higher R²)
                    - It captures complex patterns in the data better
                    - Consider tuning parameters to prevent overfitting
                    """)
                else:
                    st.info("""
                    **💡 Recommendation: Use Linear Regression Model**
                    - Linear Regression provides good performance with simplicity
                    - Easier to interpret and explain
                    - More stable predictions
                    """)

            else:
                st.warning("📝 No model metrics available. Please train the models first on the 'Model Training' page.")

        except Exception as e:
            st.error(f"Error fetching metrics: {str(e)}")

    # Project Info Page
    elif page == "ℹ️ Project Info":
        st.markdown('<div class="section-header">ℹ️ Project Information</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ## 🎓 About This Project

            This Student Exam Score Prediction System demonstrates practical machine learning 
            application in educational analytics.

            ### 🔬 Key Features:
            - **Multiple Predictive Features**: Goes beyond just study hours
            - **Two ML Models**: Compare Linear Regression vs Decision Tree
            - **Interactive Interface**: Real-time predictions and training
            - **Comprehensive Analytics**: Detailed performance metrics and visualizations

            ### 🛠️ Technical Stack:
            - **Backend**: FastAPI (Python)
            - **Frontend**: Streamlit (Python)
            - **Machine Learning**: Scikit-learn
            - **Visualization**: Plotly
            - **Data**: Synthetic student dataset

            ### 📊 Models Implemented:
            1. **Linear Regression**: Baseline model for linear relationships
            2. **Decision Tree Regression**: Captures non-linear patterns

            ### 🎯 Learning Outcomes:
            - Understanding feature importance in predictions
            - Comparing different ML algorithms
            - Practical web deployment of ML models
            - Model performance evaluation
            """)

        with col2:
            st.markdown("""
            <div class="info-box">
            <h3>👨‍🎓 Student Developer</h3>
            <p><strong>Gaurav Kumar</strong><br>
            <strong>ID:</strong> iitrprai_24102148</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
            <h3>📚 Project Scope</h3>
            <p><strong>Domain:</strong> Educational Analytics<br>
            <strong>ML Task:</strong> Regression<br>
            <strong>Dataset:</strong> Synthetic (500 students)<br>
            <strong>Features:</strong> 5 input variables</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
            <h3>⚡ Quick Start</h3>
            <p>1. Go to <strong>Data Explorer</strong> to view data<br>
            2. Visit <strong>Model Training</strong> to train models<br>
            3. Use <strong>Prediction Lab</strong> for predictions<br>
            4. Check <strong>Performance Analytics</strong> for results</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🎓 Student Exam Score Prediction System</h3>
        <p>Developed by Gaurav Kumar (iitrprai_24102148) | IIT Ropar Final Sem Machine Learning Project</p>
        <p style='font-size: 0.8rem;'>Explore different sections using the sidebar navigation ↑</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()