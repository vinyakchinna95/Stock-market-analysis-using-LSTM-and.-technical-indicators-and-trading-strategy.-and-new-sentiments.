import os
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
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

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# -----------------------------
# Configuration & Paths
# -----------------------------
# (Adjust BASE_DIR as needed; this is your local datasets folder)
BASE_DIR = r"C:\Users\saipr\OneDrive\Desktop\MAIN UPDATED PROJECT\DATASETS"
DAILY_DIR = os.path.join(BASE_DIR, "Daily_data")
WEEKLY_DIR = os.path.join(BASE_DIR, "Weekly_data")
MONTHLY_DIR = os.path.join(BASE_DIR, "Monthly_data")
SECTORS_FILE = os.path.join(BASE_DIR, "sectors with symbols.csv")

# -----------------------------
# --- GFS Analysis Functions ---
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

    total_symbols = 0
    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbols = line.split(",")
            total_symbols += len([s.strip() for s in symbols if s.strip()])

    if total_symbols == 0:
        return "No symbols found in indicesstocks.csv"

    processed = 0
    messages = []
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
                    messages.append(f"Downloaded {symbol}: {processed}/{total_symbols}")
                except Exception as e:
                    processed += 1
                    messages.append(f"Error downloading {symbol}: {e}")
    return "\n".join(messages)

def fetch_vix():
    vix = yf.Ticker("^INDIAVIX")
    vix_data = vix.history(period="1d")
    if not vix_data.empty:
        return vix_data['Close'].iloc[0]
    return None

def append_row(df, row):
    if row.isnull().all():
        return df
    return pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True)

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
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                symbol = parts[0]
                symbolname = parts[1]
                new_row = dayweekmonth_datasets(symbol.strip(), symbolname.strip(), symbol.strip())
                indicesdf = append_row(indicesdf, new_row)
    except Exception as e:
        return None, f"Error generating GFS report: {e}"
    return indicesdf, "GFS report generated successfully."

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
        return None, f"Error reading indicesstocks.csv: {e}"
    return indices_dict, "Indices stocks read successfully."

# -----------------------------
# --- LSTM Prediction Functions ---
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
        return None, "Not enough data in selected date range. Please select a wider range."

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
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=epochs, batch_size=32, verbose=0)
    
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
    
    return (train_r2, test_r2, future_preds, future_high, future_low,
            train_dates, y_train_actual, train_pred,
            test_dates, y_test_actual, test_pred), "LSTM prediction completed."

# -----------------------------
# --- News Scraping & Sentiment Analysis Functions ---
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

def load_vader():
    return SentimentIntensityAnalyzer()

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

def update_filtered_indices_by_sentiment(filepath, sentiment_method="VADER"):
    df = pd.read_csv(filepath)
    companies = df["Company Name"].unique()
    sentiment_summary = {}
    updated_companies = []
    news_data = {}
    
    for idx, company in enumerate(companies):
        news = fetch_news(company)
        company_news_details = []
        
        if not news:
            verdict = "Neutral"
        else:
            pos = neg = neu = 0
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
        
        sentiment_summary[company] = verdict
        news_data[company] = company_news_details
        
        if verdict != "Negative":
            updated_companies.append(company)
    
    updated_df = df[df["Company Name"].isin(updated_companies)]
    updated_df.to_csv(filepath, index=False)
    
    return sentiment_summary, updated_df, news_data

# -----------------------------
# --- Gradio App UI ---
# -----------------------------
import gradio as gr
import matplotlib.pyplot as plt

def run_full_gfs_analysis(data_choice):
    # data_choice: "update" or "existing"
    messages = []
    vix_value = fetch_vix()
    if vix_value is None:
        return "Could not fetch VIX data. Please try again later.", ""
    messages.append(f"VIX: {vix_value:.2f}")
    if vix_value > 20:
        messages.append("High Volatility Detected. Proceed with caution.")
    else:
        messages.append("Market Volatility Normal.")
    
    if data_choice == "update":
        os.makedirs(DAILY_DIR, exist_ok=True)
        os.makedirs(WEEKLY_DIR, exist_ok=True)
        os.makedirs(MONTHLY_DIR, exist_ok=True)
        messages.append(download_stock_data(interval='1d', folder='Daily_data'))
        messages.append(download_stock_data(interval='1wk', folder='Weekly_data'))
        messages.append(download_stock_data(interval='1mo', folder='Monthly_data'))
    
    df3, msg = generateGFS('indicesdf')
    messages.append(msg)
    if df3 is None or df3.empty:
        return "\n".join(messages), "No indices meet GFS criteria."
    
    # Qualified Indices
    df4 = df3.loc[
        (df3['monthrsi14'].between(40, 60)) &
        (df3['weekrsi14'].between(40, 60)) &
        (df3['dayrsi14'].between(40, 60))
    ]
    qualified_indices = df4.copy()
    if qualified_indices.empty:
        indices_msg = "No indices meet GFS criteria."
    else:
        indices_msg = qualified_indices.to_string(index=False)
        qualified_indices.to_csv(os.path.join(BASE_DIR, "filtered_indices.csv"), index=False)
    
    # Qualified Stocks
    indices_dict, msg_read = read_indicesstocks(os.path.join(BASE_DIR, "indicesstocks.csv"))
    messages.append(msg_read)
    results_df = pd.DataFrame(columns=df3.columns)
    if indices_dict is not None:
        for index in df4['indexcode']:
            if index in indices_dict:
                for stock in indices_dict[index]:
                    if stock:
                        new_row = dayweekmonth_datasets(stock, stock, index)
                        results_df = append_row(results_df, new_row)
    results_df = results_df.loc[
        (results_df['monthrsi14'].between(40, 60)) &
        (results_df['weekrsi14'].between(40, 60)) &
        (results_df['dayrsi14'].between(40, 60))
    ]
    if results_df.empty:
        stocks_msg = "No stocks meet GFS criteria."
    else:
        sectors_df = pd.read_csv(SECTORS_FILE)
        results_df = results_df.merge(
            sectors_df[['Index Name', 'Company Name']],
            left_on='indexname',
            right_on='Index Name',
            how='left'
        )
        results_df.drop(columns=['Index Name'], inplace=True, errors='ignore')
        results_df['Company Name'] = results_df['Company Name'].fillna('N/A')
        stocks_msg = results_df.to_string(index=False)
        results_df.to_csv(os.path.join(BASE_DIR, "filtered_indices_output.csv"), index=False)
    
    messages.append("GFS Analysis completed!")
    full_msg = "\n".join(messages)
    return full_msg, f"Qualified Indices:\n{indices_msg}\n\nQualified Stocks:\n{stocks_msg}"

def run_news_sentiment_analysis(sentiment_method):
    filtered_file = os.path.join(BASE_DIR, "filtered_indices_output.csv")
    if not os.path.exists(filtered_file):
        return "Filtered indices file not found. Please run the GFS Analysis first.", "", ""
    sentiment_summary, updated_df, news_data = update_filtered_indices_by_sentiment(filtered_file, sentiment_method)
    summary_df = pd.DataFrame.from_dict(sentiment_summary, orient='index', columns=["Verdict"])
    summary_msg = summary_df.to_string()
    updated_msg = updated_df.to_string(index=False)
    detailed_msg = ""
    for company, articles in news_data.items():
        if articles:
            detailed_msg += f"\n{company} ({len(articles)} articles):\n"
            for headline, sentiment, link in articles:
                detailed_msg += f"Headline: {headline}\nSentiment: {sentiment}\nLink: {link}\n\n"
    return "News Sentiment Analysis completed!", summary_msg, f"Detailed News Analysis:\n{detailed_msg}"

def run_lstm_analysis(epochs_input, start_date, end_date):
    filtered_indices_path = os.path.join(BASE_DIR, "filtered_indices_output.csv")
    if not os.path.exists(filtered_indices_path):
        return "Filtered indices file not found. Please run the GFS and News Sentiment Analysis first.", ""
    selected_indices = pd.read_csv(filtered_indices_path)
    sectors_df = pd.read_csv(SECTORS_FILE) if os.path.exists(SECTORS_FILE) else None
    if sectors_df is None:
        return "Sectors file not found.", ""
    daily_files_list = [f for f in os.listdir(DAILY_DIR) if f.endswith('.csv')]
    if not daily_files_list:
        return "No daily data files found in Daily_data folder.", ""
    daily_data = {}
    for file in daily_files_list:
        file_path = os.path.join(DAILY_DIR, file)
        name = os.path.splitext(file)[0].replace('_', '.')
        try:
            df_daily = pd.read_csv(file_path)
            daily_data[name] = df_daily
        except Exception as e:
            continue
    
    results = []
    summary_text = ""
    for _, row in selected_indices.iterrows():
        index_name = row['indexname']
        if index_name in daily_data:
            matching_rows = sectors_df[sectors_df['Index Name'] == index_name]
            company_name = matching_rows['Company Name'].iloc[0] if not matching_rows.empty else index_name
            result, msg = get_predicted_values(daily_data[index_name], epochs=epochs_input, start_date=start_date, end_date=end_date)
            if result is None:
                summary_text += f"Not enough data for {index_name}. Skipping...\n"
                continue
            (train_r2, test_r2, future_preds, future_high, future_low,
             train_dates, y_train_actual, train_pred,
             test_dates, y_test_actual, test_pred) = result
            summary_text += (f"Index: {index_name} ({company_name})\n"
                             f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}\n"
                             f"Future Predictions (Close): {future_preds}\n\n")
            results.append({
                'Index Name': index_name,
                'Company Name': company_name,
                'Test R2 Score': test_r2,
                'Verdict': "Strong Forecast" if test_r2 >= 0.9 else ("Moderate Forecast" if test_r2 >= 0.8 else "Weak Forecast")
            })
    if results:
        result_df = pd.DataFrame(results)
        verdict_msg = result_df.to_string(index=False)
    else:
        verdict_msg = "No valid data found for prediction."
    return summary_text, verdict_msg

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Stock Analysis & Prediction Dashboard 🗠")
    gr.Markdown("This app performs a three-step process: GFS Analysis, News Sentiment Analysis, and LSTM-based Stock Prediction.")
    
    with gr.Tabs():
        with gr.TabItem("GFS Analysis & News Sentiment Analysis"):
            with gr.Row():
                data_choice_radio = gr.Radio(choices=["update", "existing"], label="Select Data Option", value="update")
                gfs_btn = gr.Button("Run Full GFS Analysis")
            gfs_output = gr.Textbox(label="GFS Analysis Log", lines=10)
            gfs_results = gr.Textbox(label="GFS Analysis Results", lines=10)
            
            gfs_btn.click(fn=run_full_gfs_analysis, inputs=data_choice_radio, outputs=[gfs_output, gfs_results])
            
            gr.Markdown("## News Sentiment Analysis")
            sentiment_radio = gr.Radio(choices=["VADER", "FinBERT"], label="Select Sentiment Analysis Model", value="VADER")
            news_btn = gr.Button("Run News Sentiment Analysis")
            news_log = gr.Textbox(label="News Sentiment Log", lines=5)
            sentiment_summary_box = gr.Textbox(label="Sentiment Summary", lines=5)
            detailed_news_box = gr.Textbox(label="Detailed News Analysis", lines=10)
            
            news_btn.click(fn=run_news_sentiment_analysis, inputs=sentiment_radio, outputs=[news_log, sentiment_summary_box, detailed_news_box])
        
        with gr.TabItem("Stock Prediction (LSTM)"):
            gr.Markdown("## LSTM Stock Prediction")
            epochs_slider = gr.Slider(minimum=5, maximum=500, step=5, label="Number of Epochs", value=25)
            start_date_input = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2020-01-01")
            end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)", value=dt.date.today().strftime("%Y-%m-%d"))
            lstm_btn = gr.Button("Run LSTM Analysis")
            lstm_summary = gr.Textbox(label="LSTM Analysis Summary", lines=10)
            lstm_verdict = gr.Textbox(label="Prediction Verdict", lines=5)
            
            lstm_btn.click(fn=run_lstm_analysis, inputs=[epochs_slider, start_date_input, end_date_input], outputs=[lstm_summary, lstm_verdict])
    
demo.launch()
