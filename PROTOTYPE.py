import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geoip2.database
import ipaddress
import numpy as np
import tempfile

# --- STREAMLIT APP ---
st.set_page_config(page_title="FunOlympics Dashboard", page_icon=":trophy:")
st.title("FunOlympics Web Traffic Analysis")

# File uploader for the GeoLite2 database
geoip_file = st.file_uploader("Upload GeoLite2 Database File", type="mmdb")

# File upload for the CSV log data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if geoip_file is not None and uploaded_file is not None:
    # Save the uploaded GeoLite2 database to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(geoip_file.read())
        tmp_file_path = tmp_file.name
    
    # Load GeoIP2 database
    reader = geoip2.database.Reader(tmp_file_path)
    
    # Load log data from the uploaded CSV file (skipping the first row)
    df = pd.read_csv(uploaded_file, names=["timestamp", "ip_address", "method", "endpoint", "status_code"], header=0)
    df['ip_address'] = df['ip_address'].astype(str)  

    # Convert timestamp to datetime for analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')

    # Extract relevant information from endpoints
    df['sport'] = df['endpoint'].str.extract(r'/(\w+)\.html', expand=False)
    df.dropna(subset=['sport'], inplace=True)  # Remove rows where 'sport' is NaN
    df['sport'] = df['sport'].fillna('other')  # Categorize non-sport endpoints as 'other'

    # Function to get country from IP
    def get_country(ip):
        if pd.isna(ip):  # Check if ip is NaN
            return 'Unknown'
        try:
            # Convert to IP address object
            ip_address = ipaddress.ip_address(ip)
            response = reader.country(ip_address)
            return response.country.name
        except (geoip2.errors.AddressNotFoundError, ValueError):
            return 'Unknown'

    # Apply the function to the DataFrame
    df['country'] = df['ip_address'].apply(get_country)

    # Display the first 5 records of the dataset
    st.subheader("First 5 Records of the Dataset")
    st.write(df.head())

    # Filters (Dropdowns and Checkboxes)
    st.sidebar.header("Filters")
    selected_sport = st.sidebar.selectbox("Select Sport", options=["All"] + df['sport'].unique().tolist())
    selected_country = st.sidebar.selectbox("Select Country", options=["All"] + df['country'].unique().tolist())
    selected_time_range = st.sidebar.slider("Select Time Range", min_value=0, max_value=23, value=(0, 23), step=1)

    # Apply filters
    filtered_df = df.copy()

    if selected_sport != "All":
        filtered_df = filtered_df[filtered_df['sport'] == selected_sport]

    if selected_country != "All":
        filtered_df = filtered_df[filtered_df['country'] == selected_country]

    filtered_df = filtered_df[(filtered_df['timestamp'].dt.hour >= selected_time_range[0]) & (filtered_df['timestamp'].dt.hour <= selected_time_range[1])]

    # KPIs
    st.subheader("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Visits", filtered_df.shape[0])

    with col2:
        st.metric("Unique Visitors", filtered_df["ip_address"].nunique())

    with col3:
        avg_visits_per_country = round(filtered_df.groupby('country')['ip_address'].nunique().mean(), 2) if len(filtered_df) > 0 else 0
        st.metric("Average Visits/Country", avg_visits_per_country)

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write("Here are the summary statistics for the filtered data:")
    total_visits = filtered_df.shape[0]
    unique_visitors = filtered_df['ip_address'].nunique()
    visits_by_country = filtered_df.groupby('country')['ip_address'].nunique()

    summary_data = {
        'Metric': ['Total Visits', 'Unique Visitors', 'Mean Visits/Country', 'Median Visits/Country', 'Std Dev Visits/Country'],
        'Value': [
            total_visits,
            unique_visitors,
            visits_by_country.mean() if not visits_by_country.empty else 0,
            visits_by_country.median() if not visits_by_country.empty else 0,
            visits_by_country.std() if not visits_by_country.empty else 0
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.write(summary_df)

    # Visualizations
    st.subheader("Visualizations")
   
    # 1. Most Popular Event (Bar Chart) with Filter
    st.sidebar.subheader("Visualization Filters")
    sport_counts = filtered_df['sport'].value_counts()
    selected_sports = st.sidebar.multiselect(
        "Select Sports", sport_counts.index.tolist(), default=sport_counts.index.tolist(), key='sports'
    )
 
    if not filtered_df.empty:
        sport_counts = sport_counts.loc[selected_sports]
        chart_data = pd.DataFrame({'Sport': sport_counts.index, 'Visits': sport_counts.values})
        chart_data = chart_data.sort_values(by='Visits', ascending=False)  # Sort by number of visits in descending order
 
        chart1 = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Sport:N', title='Sport', sort='-y', axis=alt.Axis(labelAngle=-45)),  # Sort by y-axis values
            y=alt.Y('Visits:Q', title='Number of Visits'),
            tooltip=['Sport', 'Visits']
        ).properties(
            title='Most Popular Event'
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.markdown("<p>No data available for the selected filters.</p>", unsafe_allow_html=True)
 
    # Filter data after the chart is displayed
    filtered_df = filtered_df[filtered_df['sport'].isin(selected_sports)]
 
    # 2. Traffic Over Time (Hourly Line Chart)
    filtered_df['hour'] = pd.to_datetime(filtered_df['timestamp'], format='%H:%M:%S').dt.hour
 
    # Aggregate visits by hour, filling missing hours with zeros
    hourly_visits = filtered_df['hour'].value_counts().reindex(range(24)).fillna(0)
    
    # Find the most popular sport for each hour
    hourly_popular_sport = filtered_df.groupby('hour')['sport'].agg(lambda x: x.value_counts().index[0])
 
    # Convert to DataFrame and rename the columns
    hourly_visits_df = pd.DataFrame({'Hour': hourly_visits.index, 'Visits': hourly_visits.values})  
 
    chart3 = alt.Chart(hourly_visits_df).mark_line(point=True).encode(
        x=alt.X('Hour:O', title='Hour', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Visits:Q', title='Visits'),
        tooltip=['Hour', 'Visits']
    ).properties(
        title='Traffic Over Time (Hourly)'
    ).interactive()
 
    st.altair_chart(chart3, use_container_width=True)
 
    # 3. User Engagement by Sport (Simulated Box Plot)
    # Simulate time spent data
    filtered_df['time_spent'] = np.random.randint(10, 301, size=len(filtered_df))  # Random values between 10 and 300 seconds
 
    chart4 = alt.Chart(filtered_df).mark_boxplot(extent='min-max').encode(
        x='sport:N',
        y='time_spent:Q',
        color='sport:N',
        tooltip=['sport:N', alt.Tooltip('time_spent:Q', title='Time Spent (seconds)')]
    ).properties(
        title='User Engagement by Sport (Simulated)'
    ).interactive()
 
    st.altair_chart(chart4, use_container_width=True)
 
    # 4. Traffic by Country (Choropleth Map)
    if not filtered_df.empty:  # Check if filtered_df is empty
        country_visits = filtered_df['country'].value_counts().reset_index(name='Visits').rename(columns={'index': 'country'})
        fig_map = px.choropleth(country_visits, locations='country',  # Use "Country" here instead of 'country'
                                locationmode='country names', color="Visits",
                                hover_name='country', color_continuous_scale=px.colors.sequential.Plasma)
        fig_map.update_layout(title_text='Visits by Country')
        st.plotly_chart(fig_map)
    else:
        st.markdown("<p>No data available for the selected filters.</p>", unsafe_allow_html=True)
 
    # Anomaly Detection
    st.subheader("Anomaly Detection")
 
    # Calculate Z-scores for visit counts per hour
    hourly_visits_df['Z-score'] = (hourly_visits_df['Visits'] - hourly_visits_df['Visits'].mean()) / hourly_visits_df['Visits'].std()
 
    # Flag anomalies based on Z-score threshold
    anomaly_threshold = 2  # Threshold for Z-score to flag an anomaly
    hourly_visits_df['Anomaly'] = hourly_visits_df['Z-score'].abs() > anomaly_threshold
 
    # Highlight anomalies in the chart
    chart_anomalies = alt.Chart(hourly_visits_df).mark_line(point=True).encode(
        x=alt.X('Hour:O', title='Hour'),
        y=alt.Y('Visits:Q', title='Visits'),
        color=alt.Color('Anomaly:N', scale=alt.Scale(domain=[True, False], range=['red', 'blue'])),
        tooltip=['Hour', 'Visits', 'Z-score']
    ).properties(
        title='Traffic Over Time with Anomalies'
    ).interactive()
   
    st.altair_chart(chart_anomalies, use_container_width=True)
   
    st.subheader("Detected Anomalies")
    anomalies = hourly_visits_df[hourly_visits_df['Anomaly']]
    st.write(anomalies)

else:
    st.write("Please upload both a CSV file and the GeoLite2 database file to see the dashboard.")
