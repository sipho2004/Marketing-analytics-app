"""
All-in-One Marketing Analytics App (Streamlit)

Features:
- Customers: RFM + KMeans segmentation
- Sales: dashboard + simple linear forecast
- Campaigns: ROI, CPA, top channels
- Social media: engagement rates, best posting days
- Automated CSV reports
- Sample data generators included for quick testing

Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Run locally: streamlit run streamlit_all_in_one_marketing_app.py
3. Use sidebar to toggle sample datasets or upload your own CSVs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title='All-in-One Marketing Analytics', layout='wide')
st.title('All-in-One Marketing Analytics App')

# ---------- Sample data generators ----------
def sample_customers(n=300, seed=42):
    rng = np.random.default_rng(seed)
    recency = np.clip(rng.exponential(scale=30, size=n).round(0), 0, None)
    frequency = np.clip(rng.poisson(lam=3, size=n) + 1, 1, None)
    monetary = np.clip((frequency * (rng.normal(200,80,size=n))).round(2), 5, None)
    age = np.clip((rng.normal(35,12,size=n)).round(0), 18, 90)
    df = pd.DataFrame({
        'customer_id': [f'C{1000+i}' for i in range(n)],
        'recency_days': recency,
        'frequency': frequency,
        'monetary': monetary,
        'age': age
    })
    return df

def sample_sales(start='2023-01-01', periods=365, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq='D')
    products = ['A','B','C','D']
    rows = []
    for d in dates:
        for p in products:
            units = int(max(0, rng.poisson(lam=5) + (0 if p=='D' else rng.integers(0,3))))
            price = {'A':50,'B':80,'C':30,'D':120}[p]
            revenue = units * price
            rows.append({'date': d, 'product': p, 'units': units, 'revenue': revenue})
    return pd.DataFrame(rows)

def sample_campaigns(seed=2):
    rng = np.random.default_rng(seed)
    campaigns = []
    channels = ['Facebook','Google','Email','Instagram']
    start = pd.to_datetime('2023-01-01')
    for i in range(12):
        s = start + pd.DateOffset(weeks=2*i)
        e = s + pd.DateOffset(days=13)
        channel = channels[i % len(channels)]
        spend = int(500 + rng.integers(0,2000))
        conversions = int(max(0, rng.poisson(lam=20) + rng.integers(0,10)))
        campaigns.append({'campaign_id': f'camp_{i+1}', 'date_start': s, 'date_end': e, 'channel': channel, 'spend': spend, 'conversions': conversions})
    return pd.DataFrame(campaigns)

def sample_social(start='2023-01-01', periods=180, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq='D')
    platforms = ['Facebook','Instagram','Twitter']
    rows = []
    for d in dates:
        for p in platforms:
            impressions = int(max(0, rng.poisson(lam=2000)))
            clicks = int(impressions * rng.uniform(0.01, 0.05))
            engagements = int(impressions * rng.uniform(0.02, 0.1))
            rows.append({'date': d, 'platform': p, 'impressions': impressions, 'clicks': clicks, 'engagements': engagements})
    return pd.DataFrame(rows)

# ---------- Sidebar ----------
st.sidebar.header('Data Options')
use_sample = st.sidebar.checkbox('Use sample datasets', value=True)

uploaded_customers = st.sidebar.file_uploader('Upload customers CSV', type=['csv'])
uploaded_sales = st.sidebar.file_uploader('Upload sales CSV', type=['csv'])
uploaded_campaigns = st.sidebar.file_uploader('Upload campaigns CSV', type=['csv'])
uploaded_social = st.sidebar.file_uploader('Upload social CSV', type=['csv'])

if use_sample:
    df_customers = sample_customers()
    df_sales = sample_sales()
    df_campaigns = sample_campaigns()
    df_social = sample_social()
else:
    df_customers = pd.read_csv(uploaded_customers) if uploaded_customers else None
    df_sales = pd.read_csv(uploaded_sales) if uploaded_sales else None
    df_campaigns = pd.read_csv(uploaded_campaigns) if uploaded_campaigns else None
    df_social = pd.read_csv(uploaded_social) if uploaded_social else None

st.markdown('---')

# ---------- Customers ----------
st.header('1) Customer Insights')
if df_customers is None:
    st.info('Upload a customers CSV or enable sample data in the sidebar')
else:
    st.subheader('Customer data preview')
    st.dataframe(df_customers.head(50))

    rfm_cols = set(['recency_days','frequency','monetary'])
    if rfm_cols.issubset(set(df_customers.columns)):
        st.subheader('RFM summary')
        rfm = df_customers[['recency_days','frequency','monetary']].describe().T
        st.table(rfm)

        features = ['recency_days','frequency','monetary']
        scaler = StandardScaler()
        X = scaler.fit_transform(df_customers[features].dropna())
        k = st.slider('Choose k for KMeans (customers)', 2, 8, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        df_customers_clean = df_customers.loc[df_customers[features].dropna().index].copy()
        df_customers_clean['segment'] = labels
        st.subheader('Segment sizes')
        st.table(df_customers_clean['segment'].value_counts().rename('count'))
        st.subheader('Segment profiles (means)')
        st.dataframe(df_customers_clean.groupby('segment')[features].mean().round(2))

        csv_buf = df_customers_clean.to_csv(index=False).encode('utf-8')
        st.download_button('Download customers with segment labels', data=csv_buf, file_name='customers_segmented.csv', mime='text/csv')
    else:
        st.warning('Customers CSV should include columns: recency_days, frequency, monetary')

st.markdown('---')

# ---------- Sales ----------
st.header('2) Sales Dashboard')
if df_sales is None:
    st.info('Upload a sales CSV or enable sample data in the sidebar')
else:
    df_sales['date'] = pd.to_datetime(df_sales['date'])
    st.subheader('Sales preview')
    st.dataframe(df_sales.head(50))

    st.subheader('Top products by revenue')
    prod = df_sales.groupby('product').agg({'revenue':'sum','units':'sum'}).sort_values('revenue', ascending=False)
    st.table(prod)

    st.subheader('Daily revenue timeseries')
    daily = df_sales.groupby('date')['revenue'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(daily['date'], daily['revenue'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')
    st.pyplot(fig)

    st.subheader('Simple linear trend forecast (next N days)')
    n_days = st.number_input('Forecast days', min_value=7, value=30)
    daily = daily.sort_values('date')
    daily['t'] = np.arange(len(daily))
    model = LinearRegression()
    model.fit(daily[['t']], daily['revenue'])
    future_t = np.arange(len(daily), len(daily) + n_days).reshape(-1,1)
    preds = model.predict(future_t)
    future_dates = [daily['date'].max() + timedelta(days=int(i)) for i in range(1, n_days+1)]
    forecast_df = pd.DataFrame({'date': future_dates, 'revenue_forecast': preds.round(2)})
    st.dataframe(forecast_df)

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(daily['date'], daily['revenue'], label='history')
    ax2.plot(forecast_df['date'], forecast_df['revenue_forecast'], label='forecast')
    ax2.legend()
    st.pyplot(fig2)

    csv_buf = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download sales forecast CSV', data=csv_buf, file_name='sales_forecast.csv', mime='text/csv')

st.markdown('---')

# ---------- Campaigns ----------
st.header('3) Campaign Performance')
if df_campaigns is None:
    st.info('Upload a campaigns CSV or enable sample data in the sidebar')
else:
    df_campaigns['date_start'] = pd.to_datetime(df_campaigns['date_start'])
    df_campaigns['date_end'] = pd.to_datetime(df_campaigns['date_end'])
    st.subheader('Campaigns preview')
    st.dataframe(df_campaigns)

    st.subheader('Campaign KPIs')
    df_campaigns['cpa'] = df_campaigns['spend'] / df_campaigns['conversions']
    st.dataframe(df_campaigns[['campaign_id','channel','spend','conversions','cpa']].sort_values('spend', ascending=False))

    ch = df_campaigns.groupby('channel').agg({'spend':'sum','conversions':'sum'})
    ch['cpa'] = (ch['spend']/ch['conversions']).round(2)
    ch['conversion_rate'] = (ch['conversions']/ch['conversions'].sum()).round(4)
    st.subheader('Channel summary')
    st.table(ch.reset_index())

    csv_buf = ch.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button('Download campaign channel summary', data=csv_buf, file_name='campaign_channel_summary.csv', mime='text/csv')

st.markdown('---')

# ---------- Social ----------
st.header('4) Social Media Analytics')
if df_social is None:
    st.info('Upload a social CSV or enable sample data in the sidebar')
else:
    df_social['date'] = pd.to_datetime(df_social['date'])
    st.subheader('Social preview')
    st.dataframe(df_social.head(50))

    st.subheader('Platform summary')
    plat = df_social.groupby('platform').agg({'impressions':'sum','clicks':'sum','engagements':'sum'})
    plat['ctr'] = (plat['clicks']/plat['impressions']).round(4)
    plat['engagement_rate'] = (plat['engagements']/plat['impressions']).round(4)
    st.table(plat)

    st.subheader('Best days to post (by engagement)')
    df_social['dayofweek'] = df_social['date'].dt.day_name()
    dow = df_social.groupby('dayofweek')['engagements'].sum().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0)
    st.bar_chart(dow)

    csv_buf = plat.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button('Download social platform summary', data=csv_buf, file_name='social_platform_summary.csv', mime='text/csv')

st.markdown('---')

# ---------- Automated report ----------
st.header('5) Generate Automated Report')
report_name = st.text_input('Report filename (CSV)', value='marketing_report.csv')
if st.button('Create report'):
    parts = {}  
    if df_customers is not None:
        parts['customers_count'] = len(df_customers)
    if df_sales is not None:

