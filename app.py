# -*- coding: utf-8 -*-
"""
Kristalball - AI Inventory Management & Forecasting App
Streamlit Application for Bar Inventory Analytics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import numpy as np
from datetime import timedelta
warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="Kristalball - Inventory Management",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔮 Kristalball Inventory Management</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Bar Inventory Analytics & Forecasting System")

# File upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load and process data
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        
        # Convert date and sort
        df['Date Time Served'] = pd.to_datetime(df['Date Time Served'])
        df = df.sort_values('Date Time Served')
        
        # Calculate consumed if missing or zero
        df['Calculated Consumed (ml)'] = df['Opening Balance (ml)'] + df['Purchase (ml)'] - df['Closing Balance (ml)']
        df['Consumed (ml)'] = df.apply(
            lambda row: row['Calculated Consumed (ml)'] if row['Consumed (ml)'] == 0 else row['Consumed (ml)'],
            axis=1
        )
        df.drop(columns=['Calculated Consumed (ml)'], inplace=True)
        
        # Set index
        df.set_index('Date Time Served', inplace=True)
        
        return df

    try:
        df = load_data(uploaded_file)
        
        # Sidebar for navigation
        st.sidebar.title("📊 Navigation")
        page = st.sidebar.radio("Choose a section:", [
            "📈 Data Overview", 
            "🔍 Interactive Analysis", 
            "🔮 Forecasting & Par Levels",
            "📋 Inventory Summary"
        ])
        
        # Data Overview Page
        if page == "📈 Data Overview":
            st.header("📈 Data Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
                st.metric("Date Range", f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            
            with col2:
                st.metric("Unique Bars", df['Bar Name'].nunique())
                st.metric("Unique Brands", df['Brand Name'].nunique())
            
            with col3:
                st.metric("Alcohol Types", df['Alcohol Type'].nunique())
                st.metric("Total Consumption", f"{df['Consumed (ml)'].sum():,.0f} ml")
            
            # Data quality check
            st.subheader("📋 Data Quality Summary")
            missing_data = df.isnull().sum()
            if missing_data.sum() == 0:
                st.success("✅ No missing values found in the dataset!")
            else:
                st.warning("⚠️ Missing values detected:")
                st.write(missing_data[missing_data > 0])
            
            # Show unique values
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🏪 Bar Names")
                bar_counts = df['Bar Name'].value_counts()
                st.write(bar_counts)
            
            with col2:
                st.subheader("🍷 Alcohol Types")
                alcohol_counts = df['Alcohol Type'].value_counts()
                st.write(alcohol_counts)
            
            with col3:
                st.subheader("🏷️ Top Brands")
                brand_counts = df['Brand Name'].value_counts().head(10)
                st.write(brand_counts)
            
            # Show sample data
            st.subheader("📊 Sample Data")
            st.dataframe(df.head(10))
        
        # Interactive Analysis Page
        elif page == "🔍 Interactive Analysis":
            st.header("🔍 Interactive Consumption Analysis")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_bar = st.selectbox("🏪 Select Bar:", df['Bar Name'].unique())
            
            # Filter alcohol types based on selected bar
            bar_alcohols = df[df['Bar Name'] == selected_bar]['Alcohol Type'].unique()
            with col2:
                selected_alcohol = st.selectbox("🍷 Select Alcohol Type:", bar_alcohols)
            
            # Filter brands based on selected bar and alcohol type
            bar_alcohol_brands = df[
                (df['Bar Name'] == selected_bar) & 
                (df['Alcohol Type'] == selected_alcohol)
            ]['Brand Name'].unique()
            with col3:
                selected_brand = st.selectbox("🏷️ Select Brand:", bar_alcohol_brands)
            
            # Filter data
            filtered_data = df[
                (df['Bar Name'] == selected_bar) &
                (df['Alcohol Type'] == selected_alcohol) &
                (df['Brand Name'] == selected_brand)
            ]
            
            if not filtered_data.empty:
                # Resample data
                daily = filtered_data['Consumed (ml)'].resample('D').sum()
                weekly = filtered_data['Consumed (ml)'].resample('W').sum()
                monthly = filtered_data['Consumed (ml)'].resample('M').sum()
                
                # Create interactive plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=daily.index, 
                    y=daily.values, 
                    mode='lines+markers',
                    name='Daily',
                    line=dict(color='#3498db', width=2),
                    marker=dict(size=4)
                ))
                
                fig.add_trace(go.Scatter(
                    x=weekly.index, 
                    y=weekly.values, 
                    mode='lines+markers',
                    name='Weekly',
                    line=dict(color='#e67e22', width=3),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=monthly.index, 
                    y=monthly.values, 
                    mode='lines+markers',
                    name='Monthly',
                    line=dict(color='#27ae60', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f'📊 Consumption Trends: {selected_brand} ({selected_alcohol}) at {selected_bar}',
                    xaxis_title='Date',
                    yaxis_title='Consumed (ml)',
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Daily Average", f"{daily.mean():.1f} ml")
                
                with col2:
                    st.metric("Weekly Average", f"{weekly.mean():.1f} ml")
                
                with col3:
                    st.metric("Monthly Average", f"{monthly.mean():.1f} ml")
                
                with col4:
                    st.metric("Peak Daily", f"{daily.max():.1f} ml")
                
            else:
                st.warning("⚠️ No data available for the selected combination.")
        
        # Forecasting Page
        elif page == "🔮 Forecasting & Par Levels":
            st.header("🔮 AI-Powered Forecasting & Par Level Recommendations")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_bar = st.selectbox("🏪 Select Bar:", df['Bar Name'].unique(), key="forecast_bar")
            
            bar_alcohols = df[df['Bar Name'] == selected_bar]['Alcohol Type'].unique()
            with col2:
                selected_alcohol = st.selectbox("🍷 Select Alcohol Type:", bar_alcohols, key="forecast_alcohol")
            
            bar_alcohol_brands = df[
                (df['Bar Name'] == selected_bar) & 
                (df['Alcohol Type'] == selected_alcohol)
            ]['Brand Name'].unique()
            with col3:
                selected_brand = st.selectbox("🏷️ Select Brand:", bar_alcohol_brands, key="forecast_brand")
            
            # Filter and process data
            filtered_data = df[
                (df['Bar Name'] == selected_bar) &
                (df['Alcohol Type'] == selected_alcohol) &
                (df['Brand Name'] == selected_brand)
            ]
            
            if not filtered_data.empty:
                # Resample to weekly
                ts = filtered_data['Consumed (ml)'].resample('W').sum()
                
                if len(ts.dropna()) >= 10:
                    try:
                        # Fit SARIMA model
                        with st.spinner("🔄 Building AI forecasting model..."):
                            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
                            model_fit = model.fit(disp=False)
                        
                        # Generate forecast
                        forecast = model_fit.forecast(steps=8)
                        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(weeks=1), periods=8, freq='W')
                        
                        # Calculate par level
                        total_forecast_demand = forecast.sum()
                        buffer_percentage = 0.20
                        par_level = total_forecast_demand * (1 + buffer_percentage)
                        
                        # Create forecast plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=ts.index, 
                            y=ts.values, 
                            mode='lines+markers',
                            name='Historical Data',
                            line=dict(color='#2c3e50', width=2),
                            marker=dict(size=6)
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_index, 
                            y=forecast, 
                            mode='lines+markers',
                            name='AI Forecast',
                            line=dict(color='#e67e22', dash='dot', width=3),
                            marker=dict(size=8, symbol='diamond')
                        ))
                        
                        # Par level line
                        fig.add_trace(go.Scatter(
                            x=[ts.index[0], forecast_index[-1]],
                            y=[par_level, par_level],
                            mode='lines',
                            name=f'Recommended Par Level',
                            line=dict(color='#27ae60', dash='dash', width=3)
                        ))
                        
                        # Add confidence interval
                        try:
                            forecast_ci = model_fit.get_forecast(steps=8).conf_int()
                            fig.add_trace(go.Scatter(
                                x=list(forecast_index) + list(forecast_index[::-1]),
                                y=list(forecast_ci.iloc[:, 1]) + list(forecast_ci.iloc[:, 0][::-1]),
                                fill='toself',
                                fillcolor='rgba(230, 126, 34, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='Forecast Confidence Interval'
                            ))
                        except:
                            pass
                        
                        fig.update_layout(
                            title=f'🔮 8-Week Forecast: {selected_brand} ({selected_alcohol}) at {selected_bar}',
                            xaxis_title='Date',
                            yaxis_title='Weekly Consumption (ml)',
                            height=600,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate comprehensive metrics
                        hist_mean = ts.mean()
                        hist_std = ts.std()
                        recent_4_weeks = ts.tail(4).mean() if len(ts) >= 4 else ts.mean()
                        older_4_weeks = ts.iloc[-8:-4].mean() if len(ts) >= 8 else ts.mean()
                        trend_change = ((recent_4_weeks - older_4_weeks) / older_4_weeks * 100) if older_4_weeks != 0 else 0
                        
                        forecast_mean = forecast.mean()
                        cv_forecast = (forecast.std() / forecast_mean * 100) if forecast_mean != 0 else 0
                        
                        safety_stock = par_level - total_forecast_demand
                        buffer_weeks = (safety_stock / forecast_mean) if forecast_mean != 0 else 0
                        
                        risk_level = "Low" if cv_forecast < 30 else "Medium" if cv_forecast < 60 else "High"
                        
                        # Display key metrics
                        st.subheader("📊 Key Metrics & Recommendations")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Recommended Par Level", f"{par_level:.0f} ml", 
                                    delta=f"{par_level - hist_mean * 8:.0f} ml vs 8-week avg")
                        
                        with col2:
                            st.metric("8-Week Forecast", f"{total_forecast_demand:.0f} ml",
                                    delta=f"{forecast_mean - hist_mean:.1f} ml/week")
                        
                        with col3:
                            st.metric("Safety Buffer", f"{safety_stock:.0f} ml",
                                    delta=f"{buffer_weeks:.1f} weeks coverage")
                        
                        with col4:
                            trend_delta_color = "normal" if abs(trend_change) < 5 else "inverse" if trend_change < 0 else "normal"
                            st.metric("Demand Trend", f"{trend_change:+.1f}%",
                                    delta_color=trend_delta_color)
                        
                        # Risk assessment cards
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if risk_level == "Low":
                                st.markdown("""
                                <div class="metric-card success-card style="color: black;">
                                    <h4>🟢 Risk Level: LOW</h4>
                                    <p>Stable demand pattern with low volatility. Safe to maintain regular ordering schedule.</p>
                                    <p><strong>Recommended Action:</strong> Monthly ordering, weekly monitoring</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif risk_level == "Medium":
                                st.markdown("""
                                <div class="metric-card warning-card style="color: black;">
                                    <h4>🟡 Risk Level: MEDIUM</h4>
                                    <p>Moderate demand volatility. Consider closer monitoring and more frequent orders.</p>
                                    <p><strong>Recommended Action:</strong> Bi-weekly ordering, daily monitoring</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="metric-card danger-card style="color: black;">
                                    <h4>🔴 Risk Level: HIGH</h4>
                                    <p>High demand volatility. Requires close monitoring and flexible inventory strategy.</p>
                                    <p><strong>Recommended Action:</strong> Weekly ordering, daily monitoring, consider alternative suppliers</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            if trend_change > 10:
                                st.markdown("""
                                <div class="metric-card success-card style="color: black;">
                                    <h4>📈 Growing Demand</h4>
                                    <p>Strong upward trend detected. Consider increasing par levels proactively.</p>
                                    <p><strong>Opportunity:</strong> Stock up before peak season</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif trend_change < -10:
                                st.markdown("""
                                <div class="metric-card warning-card style="color: black;">
                                    <h4>📉 Declining Demand</h4>
                                    <p>Downward trend detected. Consider reducing par levels to avoid overstocking.</p>
                                    <p><strong>Action:</strong> Review inventory strategy</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="metric-card success-card style="color: black;">
                                    <h4>➡️ Stable Demand</h4>
                                    <p>Steady demand pattern. Current inventory strategy appears appropriate.</p>
                                    <p><strong>Status:</strong> Continue current approach</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Detailed analysis table
                        st.subheader("📋 Detailed Analysis")
                        
                        analysis_data = {
                            "Metric": [
                                "Historical Weekly Average",
                                "Recent 4-Week Average", 
                                "Forecast Weekly Average",
                                "Demand Volatility",
                                "Par Level Recommendation",
                                "Safety Stock Coverage",
                                "Reorder Frequency",
                                "Stockout Risk"
                            ],
                            "Value": [
                                f"{hist_mean:.2f} ml",
                                f"{recent_4_weeks:.2f} ml",
                                f"{forecast_mean:.2f} ml",
                                f"{cv_forecast:.1f}%",
                                f"{par_level:.2f} ml",
                                f"{buffer_weeks:.1f} weeks",
                                "Weekly" if risk_level == "High" else "Bi-weekly" if risk_level == "Medium" else "Monthly",
                                "High" if buffer_weeks < 1 else "Medium" if buffer_weeks < 2 else "Low"
                            ],
                            "Status": [
                                "📊", "📈" if trend_change > 0 else "📉", "🔮", 
                                "🟢" if cv_forecast < 30 else "🟡" if cv_forecast < 60 else "🔴",
                                "💡", "🛡️", "⏰", 
                                "🔴" if buffer_weeks < 1 else "🟡" if buffer_weeks < 2 else "🟢"
                            ]
                        }
                        
                        analysis_df = pd.DataFrame(analysis_data)
                        st.dataframe(analysis_df, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"❌ Forecasting failed: {str(e)}")
                        st.info("This might be due to insufficient data variation or other data quality issues.")
                
                else:
                    st.error("❌ Insufficient data for forecasting. Need at least 10 weeks of historical data.")
            
            else:
                st.warning("⚠️ No data available for the selected combination.")
        
        # Summary Page
        elif page == "📋 Inventory Summary":
            st.header("📋 Complete Inventory Summary")
            
            # Overall statistics
            st.subheader("🏪 Bar Performance Overview")
            
            bar_summary = df.groupby('Bar Name').agg({
                'Consumed (ml)': ['sum', 'mean', 'count'],
                'Brand Name': 'nunique',
                'Alcohol Type': 'nunique'
            }).round(2)
            
            bar_summary.columns = ['Total Consumption (ml)', 'Avg Weekly (ml)', 'Records', 'Unique Brands', 'Alcohol Types']
            bar_summary = bar_summary.sort_values('Total Consumption (ml)', ascending=False)
            
            st.dataframe(bar_summary, use_container_width=True)
            
            # Top performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top 10 Brands by Volume")
                brand_consumption = df.groupby('Brand Name')['Consumed (ml)'].sum().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=brand_consumption.values,
                    y=brand_consumption.index,
                    orientation='h',
                    title="Brand Performance",
                    labels={'x': 'Total Consumption (ml)', 'y': 'Brand'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🍷 Consumption by Alcohol Type")
                alcohol_consumption = df.groupby('Alcohol Type')['Consumed (ml)'].sum().sort_values(ascending=False)
                
                fig = px.pie(
                    values=alcohol_consumption.values,
                    names=alcohol_consumption.index,
                    title="Distribution by Type"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Monthly trends
            st.subheader("📅 Monthly Consumption Trends")
            monthly_total = df['Consumed (ml)'].resample('M').sum()
            
            fig = px.line(
                x=monthly_total.index,
                y=monthly_total.values,
                title="Overall Monthly Consumption Trend",
                labels={'x': 'Month', 'y': 'Total Consumption (ml)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate downloadable report
            if st.button("📥 Generate Detailed Report"):
                # Create comprehensive summary
                report_data = []
                
                for bar in df['Bar Name'].unique():
                    for alcohol_type in df[df['Bar Name'] == bar]['Alcohol Type'].unique():
                        for brand in df[(df['Bar Name'] == bar) & (df['Alcohol Type'] == alcohol_type)]['Brand Name'].unique():
                            subset = df[
                                (df['Bar Name'] == bar) &
                                (df['Alcohol Type'] == alcohol_type) &
                                (df['Brand Name'] == brand)
                            ]
                            
                            if len(subset) > 0:
                                ts = subset['Consumed (ml)'].resample('W').sum()
                                
                                report_data.append({
                                    'Bar Name': bar,
                                    'Alcohol Type': alcohol_type,
                                    'Brand Name': brand,
                                    'Total Consumption (ml)': subset['Consumed (ml)'].sum(),
                                    'Average Weekly (ml)': ts.mean(),
                                    'Peak Weekly (ml)': ts.max(),
                                    'Weeks of Data': len(ts),
                                    'Volatility (%)': (ts.std() / ts.mean() * 100) if ts.mean() > 0 else 0
                                })
                
                report_df = pd.DataFrame(report_data)
                report_df = report_df.sort_values('Total Consumption (ml)', ascending=False)
                
                # Convert to CSV
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Complete Inventory Report (CSV)",
                    data=csv,
                    file_name="kristalball_inventory_report.csv",
                    mime="text/csv"
                )
                
                st.success("✅ Report generated successfully!")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please make sure your CSV file has the required columns: 'Date Time Served', 'Bar Name', 'Alcohol Type', 'Brand Name', 'Opening Balance (ml)', 'Purchase (ml)', 'Closing Balance (ml)', 'Consumed (ml)'")

else:
    # Welcome screen
    st.markdown("""
    
    This powerful AI-driven application helps you:
    
    - 📊 **Analyze** consumption patterns across your bars
    - 🔮 **Forecast** future demand using advanced AI models
    - 📦 **Optimize** inventory levels with smart par level recommendations
    - ⚠️ **Assess** risk levels and prevent stockouts
    - 📋 **Generate** comprehensive inventory reports
    
    ### 📁 Getting Started
    
    1. **Upload your CSV file** using the file uploader above
    2. **Explore your data** with interactive visualizations
    3. **Get AI-powered forecasts** for any bar-brand combination
    4. **Download detailed reports** for inventory planning
    
    ### 📋 Required CSV Columns
    
    Your CSV file should contain these columns:
    - `Date Time Served`
    - `Bar Name`
    - `Alcohol Type`
    - `Brand Name`
    - `Opening Balance (ml)`
    - `Purchase (ml)`
    - `Closing Balance (ml)`
    - `Consumed (ml)`
    
    ---
    
    **Ready to optimize your inventory?** Upload your data file to begin! 🎯
    """)
    
    # Add some sample insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🔮 **AI Forecasting**\n\nGet 8-week demand predictions with confidence intervals")
    
    with col2:
        st.info("📦 **Smart Par Levels**\n\nOptimized inventory recommendations with safety stock")
    
    with col3:
        st.info("⚠️ **Risk Assessment**\n\nIdentify high-risk items and prevent stockouts")

# Footer
st.markdown("---")
# st.markdown("Made with ❤️ using Streamlit | Kristalball Inventory Management System")