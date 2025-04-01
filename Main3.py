import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
from pathlib import Path
import pandas_ta as ta
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# -----------------------------
# Additional Imports for News & Sentiment Analysis
# -----------------------------
import requests
import random
import time
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Additional import for interactive charts
import plotly.graph_objects as go
import plotly.express as px

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(
    page_title="Stock Analysis & Prediction",
    page_icon="favicon.ico",  # Ensure this file is in the same directory or update the path
    layout="wide"
)

# Hide Streamlit Default UI Elements
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set up directories (adjust BASE_DIR as needed)
BASE_DIR = r"C:\Users\saipr\OneDrive\Desktop\MAIN UPDATED PROJECT\DATASETS"
DAILY_DIR = os.path.join(BASE_DIR, "Daily_data")
WEEKLY_DIR = os.path.join(BASE_DIR, "Weekly_data")
MONTHLY_DIR = os.path.join(BASE_DIR, "Monthly_data")
SECTORS_FILE = os.path.join(BASE_DIR, "sectors with symbols.csv")

# -----------------------------
# --- GFS Analysis Functions --- (unchanged)
# -----------------------------
def get_latest_date():
    today = dt.date.today()
    return today.strftime("%Y-%m-%d")

def clean_and_save_data(data, filepath):
    data.reset_index(inplace=True)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data.to_csv(filepath, index=False)

def download_stock_data(interval, folder):
    base_path = BASE_DIR
    filepath = os.path.join(base_path, "indicesstocks.csv")
    start_date = "2020-01-01"
    end_date = get_latest_date()

    # Count total symbols
    total_symbols = 0
    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbols = line.split(",")
            total_symbols += len([s.strip() for s in symbols if s.strip()])

    if total_symbols == 0:
        st.warning("No symbols found in indicesstocks.csv")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    processed = 0

    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbols = line.split(",")
            for symbol in symbols:
                symbol = symbol.strip()
                if not symbol:
                    continue
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                    ticketfilename = symbol.replace(".", "_")
                    save_path = os.path.join(base_path, folder, f"{ticketfilename}.csv")
                    clean_and_save_data(data, save_path)
                    processed += 1
                    progress = processed / total_symbols
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading & Updating {folder} data: {processed}/{total_symbols} ({progress:.1%})")
                except Exception as e:
                    st.error(f"Error downloading & Updating {symbol}: {e}")
                    processed += 1
                    progress = processed / total_symbols
                    progress_bar.progress(progress)

    progress_bar.empty()
    status_text.empty()
    st.success(f"{folder.replace('_', ' ').title()} download & update completed!")

def fetch_vix():
    vix = yf.Ticker("^INDIAVIX")
    vix_data = vix.history(period="1d")
    return vix_data['Close'].iloc[0] if not vix_data.empty else None

def append_row(df, row):
    return pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True) if not row.isnull().all() else df

def getRSI14_and_BB(csvfilename):
    if Path(csvfilename).is_file():
        try:
            df = pd.read_csv(csvfilename)
            if df.empty or 'Close' not in df.columns:
                return 0.00, 0.00, 0.00, 0.00
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['rsi14'] = ta.rsi(df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20)
            if bb is None or df['rsi14'] is None:
                return 0.00, 0.00, 0.00, 0.00
            df['lowerband'] = bb['BBL_20_2.0']
            df['middleband'] = bb['BBM_20_2.0']
            rsival = df['rsi14'].iloc[-1].round(2)
            ltp = df['Close'].iloc[-1].round(2)
            lowerband = df['lowerband'].iloc[-1].round(2)
            middleband = df['middleband'].iloc[-1].round(2)
            return rsival, ltp, lowerband, middleband
        except Exception:
            return 0.00, 0.00, 0.00, 0.00
    else:
        return 0.00, 0.00, 0.00, 0.00

def dayweekmonth_datasets(symbol, symbolname, index_code):
    symbol_with_underscore = symbol.replace('.', '_')
    day_path = os.path.join(DAILY_DIR, f"{symbol_with_underscore}.csv")
    week_path = os.path.join(WEEKLY_DIR, f"{symbol_with_underscore}.csv")
    month_path = os.path.join(MONTHLY_DIR, f"{symbol_with_underscore}.csv")
    cday = dt.datetime.today().strftime('%d/%m/%Y')
    dayrsi14, dltp, daylowerband, daymiddleband = getRSI14_and_BB(day_path)
    weekrsi14, wltp, weeklowerband, weekmiddleband = getRSI14_and_BB(week_path)
    monthrsi14, mltp, monthlowerband, monthmiddleband = getRSI14_and_BB(month_path)
    new_row = pd.Series({
        'entrydate': cday,
        'indexcode': index_code,
        'indexname': symbolname,
        'dayrsi14': dayrsi14,
        'weekrsi14': weekrsi14,
        'monthrsi14': monthrsi14,
        'dltp': dltp,
        'daylowerband': daylowerband,
        'daymiddleband': daymiddleband,
        'weeklowerband': weeklowerband,
        'weekmiddleband': weekmiddleband,
        'monthlowerband': monthlowerband,
        'monthmiddleband': monthmiddleband
    })
    return new_row

def generateGFS(scripttype):
    indicesdf = pd.DataFrame(columns=[
        'entrydate', 'indexcode', 'indexname', 'dayrsi14', 
        'weekrsi14', 'monthrsi14', 'dltp', 'daylowerband', 
        'daymiddleband', 'weeklowerband', 'weekmiddleband', 
        'monthlowerband', 'monthmiddleband'
    ])
    try:
        with open(os.path.join(BASE_DIR, f"{scripttype}.csv")) as f:
            for line in f:
                if "," not in line:
                    continue
                symbol, symbolname = line.split(",")[0], line.split(",")[1]
                new_row = dayweekmonth_datasets(symbol.strip(), symbolname.strip(), symbol.strip())
                indicesdf = append_row(indicesdf, new_row)
    except Exception as e:
        st.error(f"Error generating GFS report: {e}")
    return indicesdf

def read_indicesstocks(file_path):
    indices_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    index_code = parts[0].strip()
                    indices_dict[index_code] = [stock.strip() for stock in parts[1:]]
    except Exception as e:
        st.error(f"Error reading indicesstocks.csv: {e}")
    return indices_dict

# -----------------------------
# --- LSTM Prediction Functions --- (with empty dataset check)
# -----------------------------
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def predict_future(model, last_sequence, scaler, days=5):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        next_pred = model.predict(current_sequence.reshape(1, -1, 1))
        predicted_value = next_pred[0, 0]
        predictions.append(predicted_value)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted_value
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_predicted_values(data, epochs=25, start_date=None, end_date=None):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    if start_date and end_date:
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    
    if len(df) < 50:
        st.warning("Not enough data in selected date range. Please select a wider range.")
        return None

    if 'High' in df.columns and 'Low' in df.columns:
        avg_high_gap = (df['High'] - df['Close']).mean()
        avg_low_gap = (df['Close'] - df['Low']).mean()
    else:
        avg_high_gap = 0
        avg_low_gap = 0
    
    close_data = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    
    time_step = 13
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # Check if training or testing data is empty
    if X_train.size == 0 or X_test.size == 0:
        st.error("Not enough data for training. Please select a wider date range.")
        return None

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  epochs=epochs, batch_size=32, verbose=1)
    
    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    train_dates = df['Date'].iloc[time_step + 1 : train_size]
    test_start = train_size + time_step
    test_end = test_start + len(y_test_actual)
    test_dates = df['Date'].iloc[test_start : test_end]
    
    last_sequence = scaled_data[-time_step:]
    future_preds = predict_future(model, last_sequence, scaler)
    
    future_high = future_preds + avg_high_gap
    future_low = future_preds - avg_low_gap
    
    return (
        train_r2, test_r2, future_preds, future_high, future_low,
        train_dates, y_train_actual, train_pred,
        test_dates, y_test_actual, test_pred
    )

# -----------------------------
# --- News Scraping & Sentiment Analysis Functions --- (unchanged except one addition)
# -----------------------------
def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session_news = get_session()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]

@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

def scrape_moneycontrol_news(company):
    search_url = f"https://www.moneycontrol.com/news/tags/{company.replace(' ', '-').lower()}.html"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session_news.get(search_url, headers=headers, timeout=10)
        if "captcha" in response.text.lower():
            raise Exception("CAPTCHA detected on Moneycontrol")
        response.raise_for_status()
        time.sleep(random.uniform(1, 3))
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')[:5]
        news = []
        for article in articles:
            if article.find('h2') and article.find('a'):
                headline = article.find('h2').text.strip()
                link = article.find('a')['href']
                news.append((headline, link))
        return news
    except Exception:
        return []

def scrape_bing_news(company):
    search_url = f"https://www.bing.com/news/search?q={company.replace(' ', '+')}"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session_news.get(search_url, headers=headers, timeout=10)
        if "captcha" in response.text.lower():
            raise Exception("CAPTCHA detected on Bing News")
        response.raise_for_status()
        time.sleep(random.uniform(1, 3))
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all("div", class_="news-card")[:5]
        results = []
        for a in articles:
            link_elem = a.find("a")
            if link_elem:
                headline = link_elem.get_text(strip=True)
                link = link_elem.get("href")
                results.append((headline, link))
        return results if results else []
    except Exception:
        return []

def fetch_news_newsapi(company):
    import requests.utils
    news_api_key = "063b1b2696c24c3a867c46c94cf9b810"
    keywords = (
        '"stocks" OR "market" OR "earnings" OR "finance" OR "shares" OR '
        '"dividends" OR "profit" OR "investment" OR "revenue" OR "IPO" OR '
        '"acquisition" OR "merger" OR "valuation" OR "forecast" OR "guidance" OR '
        '"liquidity" OR "debt" OR "equity" OR "yield" OR "expense" OR '
        '"gain" OR "growth" OR "upturn" OR "surge" OR "positive" OR "favorable" OR '
        '"improvement" OR "expansion" OR "strong" OR '
        '"loss" OR "decline" OR "drop" OR "bearish" OR "downturn" OR "negative" OR '
        '"weak" OR "recession" OR "crisis" OR "slump"'
    )
    query = f'"{company}" AND ({keywords})'
    encoded_query = requests.utils.quote(query)
    search_url = f"https://newsapi.org/v2/everything?qInTitle={encoded_query}&apiKey={news_api_key}"
    try:
        response = session_news.get(search_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        news = []
        for article in articles[:5]:
            headline = article.get("title", "")
            link = article.get("url", "")
            if headline and link:
                news.append((headline, link))
        return news
    except Exception:
        return []

def fetch_news(company):
    news = scrape_moneycontrol_news(company)
    if not news:
        time.sleep(random.uniform(1, 3))
        news = scrape_bing_news(company)
    return news

def analyze_sentiment(text, method):
    try:
        if method == "VADER":
            sia = load_vader()
            scores = sia.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                return "Positive"
            elif compound <= -0.05:
                return "Negative"
            else:
                return "Neutral"
        elif method == "FinBERT":
            finbert = load_finbert()
            result = finbert(text[:512], truncation=True)[0]
            return result['label'].capitalize()
    except Exception:
        return "Error"

def update_filtered_indices_by_sentiment(filepath, sentiment_method="VADER", use_newsapi=False, aggregate_sources=False):
    df = pd.read_csv(filepath)
    companies = df["Company Name"].unique()
    sentiment_summary = {}
    updated_companies = []
    news_data = {}

    progress_bar = st.progress(0)
    for idx, company in enumerate(companies):
        progress_bar.progress((idx + 1) / len(companies))
        if aggregate_sources:
            news_api_news = fetch_news_newsapi(company)
            default_news = fetch_news(company)
            combined_news = news_api_news + default_news
            unique_news = []
            seen = set()
            for headline, link in combined_news:
                key = (headline, link)
                if key not in seen:
                    unique_news.append((headline, link))
                    seen.add(key)
            news = unique_news
        elif use_newsapi:
            news = fetch_news_newsapi(company)
        else:
            news = fetch_news(company)
            
        company_news_details = []
        pos = neg = neu = 0
        
        if not news:
            verdict = "Neutral"
        else:
            for headline, link in news:
                sentiment = analyze_sentiment(headline, sentiment_method)
                company_news_details.append((headline, sentiment, link))
                if sentiment == "Positive":
                    pos += 1
                elif sentiment == "Negative":
                    neg += 1
                else:
                    neu += 1
            total = pos + neg + neu
            if total > 0:
                if (pos / total) > 0.4:
                    verdict = "Positive"
                elif (neg / total) > 0.4:
                    verdict = "Negative"
                else:
                    verdict = "Neutral"
            else:
                verdict = "Neutral"

        sentiment_summary[company] = {"Verdict": verdict, "Positive": pos, "Negative": neg, "Neutral": neu}
        news_data[company] = company_news_details

        if verdict != "Negative":
            updated_companies.append(company)

    progress_bar.empty()
    updated_df = df[df["Company Name"].isin(updated_companies)]
    updated_df.to_csv(filepath, index=False)

    return sentiment_summary, updated_df, news_data

# New helper to get a simple sentiment score for a stock (News Impact)
def get_stock_news_sentiment(stock):
    search_url = f"https://www.bing.com/news/search?q={stock.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all("a", {"class": "title"}, limit=5)
        sia = load_vader()
        sentiments = []
        for article in articles:
            headline = article.get_text(strip=True)
            sentiment_score = sia.polarity_scores(headline)["compound"]
            sentiments.append(sentiment_score)
        return np.mean(sentiments) if sentiments else 0
    except Exception:
        return 0

# -----------------------------
# --- Streamlit App UI ---
# -----------------------------
st.title("Stock Analysis & Prediction Dashboard 🗠")
st.markdown(
    "This app performs a three-step process: first it runs a GFS analysis to filter qualified indices/stocks, "
    "then it applies a news sentiment filter before running an LSTM model for future price predictions."
)

st.sidebar.header("Information")
st.sidebar.markdown(
    """
- **Volatility Index:** High VIX means more ups and downs, while a low VIX means a steadier market.   
- **GFS Analysis:** Download data, calculate technical indicators, and filter stocks.
- **News Sentiment Analysis:** Choose a sentiment model (VADER or FinBERT) and remove companies with negative sentiment.
- **Stock Prediction (LSTM):** Run future price predictions on the remaining companies.
"""
)

tab1, tab2 = st.tabs(["GFS Analysis & News Sentiment Analysis", "Stock Prediction (LSTM)"])

# ----------- Tab 1: GFS Analysis & News Sentiment ----------- 
with tab1:
    st.header("GFS Analysis")
    st.markdown(
        """
        **Overview:**  
        This section downloads stock data (Daily/Weekly/Monthly) for symbols listed in `indicesstocks.csv`, 
        calculates technical indicators (RSI, Bollinger Bands), and filters stocks based on multi-timeframe criteria.
        """
    )

    if st.button("Run Full GFS Analysis"):
        with st.spinner("Fetching VIX data..."):
            vix_value = fetch_vix()
            st.session_state.vix_value = vix_value

        if vix_value is None:
            st.error("Could not fetch VIX data. Please try again later.")
        else:
            st.session_state.show_data_choice = True
            if vix_value > 20:
                st.warning(
                    f"""
                    **High Volatility Detected (VIX: {vix_value:.2f})**  
                    Market conditions are volatile. Proceed with caution.  
                    Any analysis should be considered high-risk.
                    """
                )
            else:
                st.success(f"Market Volatility Normal (VIX: {vix_value:.2f})")

    if st.session_state.get('show_data_choice', False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update Data with Latest"):
                st.session_state.data_choice = 'update'
        with col2:
            if st.button("Continue with Existing Data"):
                st.session_state.data_choice = 'existing'

        if 'data_choice' in st.session_state:
            if st.session_state.data_choice == 'update':
                with st.spinner("Downloading and updating data..."):
                    os.makedirs(DAILY_DIR, exist_ok=True)
                    os.makedirs(WEEKLY_DIR, exist_ok=True)
                    os.makedirs(MONTHLY_DIR, exist_ok=True)
                    download_stock_data(interval='1d', folder='Daily_data')
                    download_stock_data(interval='1wk', folder='Weekly_data')
                    download_stock_data(interval='1mo', folder='Monthly_data')
                
                def generate_gfs_reports():
                    df3 = generateGFS('indicesdf')
                    df4 = df3.loc[
                        df3['monthrsi14'].between(40, 60) &
                        df3['weekrsi14'].between(40, 60) &
                        df3['dayrsi14'].between(40, 60)
                    ]
                    
                    st.markdown("### Qualified Indices")
                    if df4.empty:
                        st.warning("No indices meet GFS criteria.")
                    else:
                        st.dataframe(df4.style.format({
                            'dayrsi14': '{:.2f}',
                            'weekrsi14': '{:.2f}',
                            'monthrsi14': '{:.2f}',
                            'dltp': '{:.2f}'
                        }), use_container_width=True)
                        df4.to_csv(os.path.join(BASE_DIR, "filtered_indices.csv"), index=False)
                    
                    st.markdown("### Qualified Stocks")
                    indices_dict = read_indicesstocks(os.path.join(BASE_DIR, "indicesstocks.csv"))
                    results_df = pd.DataFrame(columns=df3.columns)
                    for index in df4['indexcode']:
                        if index in indices_dict:
                            for stock in indices_dict[index]:
                                if stock:
                                    new_row = dayweekmonth_datasets(stock, stock, index)
                                    results_df = append_row(results_df, new_row)
                    
                    results_df = results_df.loc[
                        results_df['monthrsi14'].between(40, 60) &
                        results_df['weekrsi14'].between(40, 60) &
                        results_df['dayrsi14'].between(40, 60)
                    ]
                    
                    sectors_df = pd.read_csv(SECTORS_FILE)
                    results_df = results_df.merge(
                        sectors_df[['Index Name', 'Company Name']],
                        left_on='indexname',
                        right_on='Index Name',
                        how='left'
                    )
                    results_df.drop(columns=['Index Name'], inplace=True, errors='ignore')
                    results_df['Company Name'] = results_df['Company Name'].fillna('N/A')
                    
                    if results_df.empty:
                        st.warning("No stocks meet GFS criteria.")
                    else:
                        st.dataframe(results_df.style.format({
                            'dayrsi14': '{:.2f}',
                            'weekrsi14': '{:.2f}',
                            'monthrsi14': '{:.2f}',
                            'dltp': '{:.2f}'
                        }), use_container_width=True)
                        results_df.to_csv(os.path.join(BASE_DIR, "filtered_indices_output.csv"), index=False)
                    
                    st.success("GFS Analysis completed!")
                
                generate_gfs_reports()
                del st.session_state.data_choice
                st.session_state.show_data_choice = False

            elif st.session_state.data_choice == 'existing':
                with st.spinner("Using existing data..."):
                    def generate_gfs_reports():
                        df3 = generateGFS('indicesdf')
                        df4 = df3.loc[
                            df3['monthrsi14'].between(40, 60) &
                            df3['weekrsi14'].between(40, 60) &
                            df3['dayrsi14'].between(40, 60)
                        ]
                        
                        st.markdown("### Qualified Indices")
                        if df4.empty:
                            st.warning("No indices meet GFS criteria.")
                        else:
                            st.dataframe(df4.style.format({
                                'dayrsi14': '{:.2f}',
                                'weekrsi14': '{:.2f}',
                                'monthrsi14': '{:.2f}',
                                'dltp': '{:.2f}'
                            }), use_container_width=True)
                            df4.to_csv(os.path.join(BASE_DIR, "filtered_indices.csv"), index=False)
                        
                        st.markdown("### Qualified Stocks")
                        indices_dict = read_indicesstocks(os.path.join(BASE_DIR, "indicesstocks.csv"))
                        results_df = pd.DataFrame(columns=df3.columns)
                        for index in df4['indexcode']:
                            if index in indices_dict:
                                for stock in indices_dict[index]:
                                    if stock:
                                        new_row = dayweekmonth_datasets(stock, stock, index)
                                        results_df = append_row(results_df, new_row)
                        
                        results_df = results_df.loc[
                            results_df['monthrsi14'].between(40, 60) &
                            results_df['weekrsi14'].between(40, 60) &
                            results_df['dayrsi14'].between(40, 60)
                        ]
                        
                        sectors_df = pd.read_csv(SECTORS_FILE)
                        results_df = results_df.merge(
                            sectors_df[['Index Name', 'Company Name']],
                            left_on='indexname',
                            right_on='Index Name',
                            how='left'
                        )
                        results_df.drop(columns=['Index Name'], inplace=True, errors='ignore')
                        results_df['Company Name'] = results_df['Company Name'].fillna('N/A')
                        
                        if results_df.empty:
                            st.warning("No stocks meet GFS criteria.")
                        else:
                            st.dataframe(results_df.style.format({
                                'dayrsi14': '{:.2f}',
                                'weekrsi14': '{:.2f}',
                                'monthrsi14': '{:.2f}',
                                'dltp': '{:.2f}'
                            }), use_container_width=True)
                            results_df.to_csv(os.path.join(BASE_DIR, "filtered_indices_output.csv"), index=False)
                        
                        st.success("GFS Analysis completed!")
                    
                    generate_gfs_reports()
                    del st.session_state.data_choice
                    st.session_state.show_data_choice = False

    st.markdown("## News Sentiment Analysis")
    st.markdown(
        """
        The filtered stocks from the GFS analysis are now evaluated based on recent news sentiment.  
        Companies with an overall negative sentiment (based on scraped news headlines) will be removed.
        """
    )
    filtered_file = os.path.join(BASE_DIR, "filtered_indices_output.csv")
    if os.path.exists(filtered_file):
        sentiment_method = st.radio(
            "Select Sentiment Analysis Model",
            ("VADER", "FinBERT"),
            index=0
        )
        use_newsapi = st.checkbox("Keyword Specific (NewsAPI.org)", value=False)
        aggregate_sources = st.checkbox("Aggregate Keywords & Multi-Sources", value=False)
        if st.button("Run News Sentiment Analysis"):
            with st.spinner("Analyzing news sentiment for each company..."):
                sentiment_summary, updated_df, news_data = update_filtered_indices_by_sentiment(
                    filtered_file,
                    sentiment_method,
                    use_newsapi,
                    aggregate_sources
                )
            st.success("News Sentiment Analysis completed!")

            st.markdown("### Sentiment Summary")
            summary_df = pd.DataFrame.from_dict(sentiment_summary, orient='index')
            st.dataframe(summary_df)

            st.markdown("### Updated Filtered Indices")
            st.dataframe(updated_df.style.format({
                'dayrsi14': '{:.2f}',
                'weekrsi14': '{:.2f}',
                'monthrsi14': '{:.2f}',
                'dltp': '{:.2f}'
            }), use_container_width=True)

            st.markdown("### Detailed News Analysis")
            for company, articles in news_data.items():
                if articles:
                    with st.expander(f"{company} ({len(articles)} articles)"):
                        for headline, sentiment, link in articles:
                            st.markdown(f"**{headline}**  \nSentiment: `{sentiment}`  \n[Read Article]({link})")
    else:
        st.info("GFS analysis output not found. Please run the GFS Analysis first.")

# ----------- Tab 2: LSTM Stock Prediction -----------
with tab2:
    st.header("Stock Prediction using LSTM")
    st.markdown(
        """
        **Overview:**  
        This section loads the filtered stocks (output from the GFS and News Sentiment Analysis) 
        and trains an LSTM model on their daily data to predict future prices.
        """
    )
    filtered_indices_path = os.path.join(BASE_DIR, "filtered_indices_output.csv")
    if os.path.exists(filtered_indices_path):
        selected_indices = pd.read_csv(filtered_indices_path)
        st.success("Loaded filtered indices from GFS & News Sentiment Analysis.")
    else:
        st.error("Filtered indices file not found. Please run the GFS and News Sentiment Analysis first.")
        st.stop()

    if os.path.exists(SECTORS_FILE):
        sectors_df = pd.read_csv(SECTORS_FILE)
    else:
        st.error("sectors with symbols.csv file not found at the specified path.")
        st.stop()

    daily_data = {}
    daily_files_list = [f for f in os.listdir(DAILY_DIR) if f.endswith('.csv')]
    if not daily_files_list:
        st.error("No daily data files found in the Daily_data folder.")
        st.stop()

    for file in daily_files_list:
        file_path = os.path.join(DAILY_DIR, file)
        name = os.path.splitext(file)[0].replace('_', '.')
        try:
            df_daily = pd.read_csv(file_path)
            daily_data[name] = df_daily
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    st.sidebar.header("LSTM Configuration")
    epochs_input = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=500, value=25)
    start_date = st.sidebar.date_input("Select Start Date", value=dt.date(2020, 1, 1))
    end_date = st.sidebar.date_input("Select End Date", value=dt.date.today())
    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
        st.stop()

    # NEW: Radio option to choose display mode
    display_mode = st.radio("Display Mode", ("Single Stock Analysis", "Multi-Stock Comparison"))

    if display_mode == "Multi-Stock Comparison":
        # Multi-stock comparison code (omitted here for brevity; remains as before)
        available_stocks = selected_indices['indexname'].unique().tolist()
        selected_stocks = st.multiselect("Select stocks for comparison", options=available_stocks, default=available_stocks)
        
        def get_future_predictions_for_stock(index_name):
            if index_name in daily_data:
                result = get_predicted_values(
                    daily_data[index_name],
                    epochs=epochs_input,
                    start_date=start_date,
                    end_date=end_date
                )
                if result is None:
                    return None
                (train_r2, test_r2, future_preds, future_high, future_low,
                 train_dates, y_train_actual, train_pred,
                 test_dates, y_test_actual, test_pred) = result
                last_date_in_data = pd.to_datetime(daily_data[index_name]['Date']).max()
                future_dates = pd.date_range(last_date_in_data + pd.Timedelta(days=1), periods=5, freq='B')[:5]
                return future_dates, future_preds, future_high, future_low, test_r2
            else:
                return None

        combined_fig = go.Figure()
        results_list = []
        sentiment_summary = {}
        for stock in selected_stocks:
            data = get_future_predictions_for_stock(stock)
            if data is not None:
                future_dates, future_preds, future_high, future_low, test_r2 = data
                sentiment = get_stock_news_sentiment(stock)
                sentiment_label = "Positive" if sentiment > 0.05 else ("Negative" if sentiment < -0.05 else "Neutral")
                sentiment_summary[stock] = sentiment_label
                combined_fig.add_trace(go.Scatter(
                    x=future_dates, y=future_preds,
                    mode="lines+markers",
                    name=f"{stock} (Sentiment: {sentiment_label}, R2: {test_r2:.2f})"
                ))
                combined_fig.add_trace(go.Scatter(
                    x=future_dates, y=future_high,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                combined_fig.add_trace(go.Scatter(
                    x=future_dates, y=future_low,
                    mode="lines",
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                forecast_category = "Strong Forecast" if test_r2 >= 0.9 else ("Moderate Forecast" if test_r2 >= 0.8 else "Weak Forecast")
                results_list.append({
                    'Stock': stock,
                    'Test R2 Score': test_r2,
                    'Close Day 1': future_preds[0],
                    'Close Day 2': future_preds[1],
                    'Close Day 3': future_preds[2],
                    'Close Day 4': future_preds[3],
                    'Close Day 5': future_preds[4],
                    'High Day 1': future_high[0],
                    'High Day 2': future_high[1],
                    'High Day 3': future_high[2],
                    'High Day 4': future_high[3],
                    'High Day 5': future_high[4],
                    'Low Day 1': future_low[0],
                    'Low Day 2': future_low[1],
                    'Low Day 3': future_low[2],
                    'Low Day 4': future_low[3],
                    'Low Day 5': future_low[4],
                    'Forecast': forecast_category,
                    'Sentiment': sentiment_label
                })
        combined_fig.update_layout(title="Multi-Stock Future Predictions Comparison",
                                   xaxis_title="Date",
                                   yaxis_title="Predicted Price",
                                   hovermode="x unified",
                                   template="plotly_dark")
        st.plotly_chart(combined_fig, use_container_width=True)
        
        st.markdown("### News Sentiment Summary for Compared Stocks")
        sentiment_df = pd.DataFrame(list(sentiment_summary.items()), columns=["Stock", "Sentiment"])
        st.dataframe(sentiment_df)
        
        if results_list:
            results_df = pd.DataFrame(results_list)
            st.markdown("### Detailed Prediction Summary")
            st.dataframe(results_df.style.format({
                'Test R2 Score': '{:.4f}',
                'Close Day 1': '{:.2f}',
                'Close Day 2': '{:.2f}',
                'Close Day 3': '{:.2f}',
                'Close Day 4': '{:.2f}',
                'Close Day 5': '{:.2f}',
                'High Day 1': '{:.2f}',
                'High Day 2': '{:.2f}',
                'High Day 3': '{:.2f}',
                'High Day 4': '{:.2f}',
                'High Day 5': '{:.2f}',
                'Low Day 1': '{:.2f}',
                'Low Day 2': '{:.2f}',
                'Low Day 3': '{:.2f}',
                'Low Day 4': '{:.2f}',
                'Low Day 5': '{:.2f}'
            }))
            
            summary_df = results_df.groupby('Forecast').agg({
                'Close Day 1': 'mean',
                'Close Day 2': 'mean',
                'Close Day 3': 'mean',
                'Close Day 4': 'mean',
                'Close Day 5': 'mean'
            }).reset_index()
            st.markdown("### Forecast Category Summary (Average Predicted Close)")
            st.dataframe(summary_df.style.format({
                'Close Day 1': '{:.2f}',
                'Close Day 2': '{:.2f}',
                'Close Day 3': '{:.2f}',
                'Close Day 4': '{:.2f}',
                'Close Day 5': '{:.2f}'
            }))
    else:
        # Single Stock Analysis
        for _, row in selected_indices.iterrows():
            index_name = row['indexname']
            if index_name in daily_data:
                matching_rows = sectors_df[sectors_df['Index Name'] == index_name]
                if not matching_rows.empty:
                    company_name = matching_rows['Company Name'].iloc[0]
                else:
                    company_name = index_name

                col_header, col_button = st.columns([4, 1])
                with col_header:
                    st.subheader(f"Processing {index_name} ({company_name})")
                with col_button:
                    yahoo_url = f"https://finance.yahoo.com/chart/{index_name}"
                    st.markdown(f"[🗠 View Current Charts on Yahoo Finance]({yahoo_url})")

                result = get_predicted_values(
                    daily_data[index_name],
                    epochs=epochs_input,
                    start_date=start_date,
                    end_date=end_date
                )
                if result is None:
                    st.warning(f"Not enough data for {index_name}. Skipping...")
                    continue

                (
                    train_r2, test_r2, future_preds, future_high, future_low,
                    train_dates, y_train_actual, train_pred,
                    test_dates, y_test_actual, test_pred
                ) = result

                # Prepare Plotly charts for Training and Test Data
                train_plot_df = pd.DataFrame({
                    'Date': pd.to_datetime(train_dates),
                    'Actual': y_train_actual.flatten(),
                    'Predicted': train_pred.flatten()
                })
                test_plot_df = pd.DataFrame({
                    'Date': pd.to_datetime(test_dates),
                    'Actual': y_test_actual.flatten(),
                    'Predicted': test_pred.flatten()
                })

                fig_train = go.Figure()
                fig_train.add_trace(go.Scatter(x=train_plot_df['Date'], y=train_plot_df['Actual'], mode='lines', name='Actual'))
                fig_train.add_trace(go.Scatter(x=train_plot_df['Date'], y=train_plot_df['Predicted'], mode='lines', name='Predicted'))
                fig_train.update_layout(title=f"Training Data (R² = {train_r2:.4f})", xaxis_title="Date", yaxis_title="Price")
                
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(x=test_plot_df['Date'], y=test_plot_df['Actual'], mode='lines', name='Actual'))
                fig_test.add_trace(go.Scatter(x=test_plot_df['Date'], y=test_plot_df['Predicted'], mode='lines', name='Predicted'))
                fig_test.update_layout(title=f"Test Data (R² = {test_r2:.4f})", xaxis_title="Date", yaxis_title="Price")

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_train, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_test, use_container_width=True)

                last_date_in_data = pd.to_datetime(daily_data[index_name]['Date']).max()
                future_dates = pd.date_range(last_date_in_data + pd.Timedelta(days=1), periods=5, freq='B')[:5]
                future_plot_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Close': future_preds,
                    'Predicted High': future_high,
                    'Predicted Low': future_low
                })

                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(x=future_plot_df['Date'], y=future_plot_df['Predicted Close'], mode='lines+markers', name='Predicted Close'))
                fig_future.add_trace(go.Scatter(x=future_plot_df['Date'], y=future_plot_df['Predicted High'], mode='lines', name='Predicted High', line=dict(dash='dot')))
                fig_future.add_trace(go.Scatter(x=future_plot_df['Date'], y=future_plot_df['Predicted Low'], mode='lines', name='Predicted Low', line=dict(dash='dot')))
                fig_future.update_layout(title="Future Predictions", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
                st.plotly_chart(fig_future, use_container_width=True)
                
                # New: Create a prediction summary table for this single stock
                forecast_category = "Strong Forecast" if test_r2 >= 0.9 else ("Moderate Forecast" if test_r2 >= 0.8 else "Weak Forecast")
                st.markdown(f"**Forecast Category:** {forecast_category} (Test R² = {test_r2:.4f})")
                future_summary_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Close": future_preds,
                    "Predicted High": future_high,
                    "Predicted Low": future_low
                })
                st.markdown("### Prediction Summary for This Stock")
                st.dataframe(future_summary_df.style.format({
                    "Predicted Close": "{:.2f}",
                    "Predicted High": "{:.2f}",
                    "Predicted Low": "{:.2f}"
                }))
    # End of Tab 2
