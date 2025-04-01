import requests
import streamlit as st
import random
import time
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Download NLTK resources
nltk.download('vader_lexicon')

# List of rotating User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]

def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = get_session()

# Initialize sentiment analyzers
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
        response = session.get(search_url, headers=headers, timeout=10)
        if "captcha" in response.text.lower():
            raise Exception("CAPTCHA detected on Moneycontrol")
        response.raise_for_status()
        time.sleep(random.uniform(1, 3))
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')[:5]
        news = [(article.find('h2').text.strip(), article.find('a')['href'])
                for article in articles if article.find('h2') and article.find('a')]
        return news
    except requests.HTTPError as e:
        return None if e.response.status_code == 403 else []
    except Exception as e:
        return None if "CAPTCHA" in str(e) else []

def scrape_bing_news(company):
    search_url = f"https://www.bing.com/news/search?q={company.replace(' ', '+')}"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session.get(search_url, headers=headers, timeout=10)
        if "captcha" in response.text.lower():
            raise Exception("CAPTCHA detected on Bing News")
        response.raise_for_status()
        time.sleep(random.uniform(1, 3))
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all("div", class_="news-card")[:5]
        results = [(a.find("a").get_text(strip=True), a.find("a").get("href"))
                   for a in articles if a.find("a")]
        return results if results else []
    except Exception as e:
        return []

def fetch_news(company):
    news = scrape_moneycontrol_news(company)
    # If no news found from Moneycontrol (None or empty list), fallback to Bing News
    if not news:
        time.sleep(random.uniform(1, 3))
        news = scrape_bing_news(company)
    return news

def analyze_sentiment(text, method):
    try:
        if method == "VADER":
            sia = load_vader()
            scores = sia.polarity_scores(text)
            return "Positive" if scores['compound'] >= 0.05 else "Negative" if scores['compound'] <= -0.05 else "Neutral"
        elif method == "FinBERT":
            finbert = load_finbert()
            result = finbert(text[:512], truncation=True)[0]
            return result['label'].capitalize()
    except Exception:
        return "Error"

st.title("News Scraper & Sentiment Analyzer")
st.markdown("**Enter company names or stock tickers (e.g., 'Apple' or 'AAPL'):**")
companies_input = st.text_area("Separate multiple entries with commas", placeholder="Apple, AAPL, Tata Motors")

method = st.radio("Sentiment Analysis Model:", ("VADER (General Purpose)", "FinBERT (Financial Specific)"), index=1).split()[0]

if st.button("Fetch News & Analyze"):
    companies = [c.strip() for c in companies_input.split(",") if c.strip()]
    if not companies:
        st.error("Please enter at least one company name/ticker")
    else:
        progress_bar = st.progress(0)
        news_data = {}
        sentiment_summary = {}

        for idx, company in enumerate(companies):
            progress_bar.progress((idx + 1) / len(companies))
            st.subheader(f"Analyzing: {company}")
            news = fetch_news(company)
            if news is None or not news:
                st.warning(f"No news found for {company}")
                continue

            company_news = []
            pos = neg = neu = 0
            for headline, link in news:
                sentiment = analyze_sentiment(headline, method)
                company_news.append((headline, sentiment, link))
                pos += (sentiment == "Positive")
                neg += (sentiment == "Negative")
                neu += (sentiment == "Neutral")

            total = pos + neg + neu
            verdict = "Positive" if pos / total > 0.4 else "Negative" if neg / total > 0.4 else "Neutral"

            news_data[company] = company_news
            sentiment_summary[company] = {"Positive": pos, "Negative": neg, "Neutral": neu, "Verdict": verdict}
            st.success(f"Analyzed {len(news)} articles for {company} - Overall Sentiment: {verdict}")
            time.sleep(1)

            st.subheader("Final Analysis Report")
        if news_data:
            st.write("### Sentiment Summary")
            summary_df = pd.DataFrame.from_dict(sentiment_summary, orient='index')
            
            # Custom styling function
            def color_verdict(val):
                return 'background-color: #ff0000' if val == "Negative" else 'background-color: #21bc24'
            
            # Apply styling
            styled_df = (summary_df.style
                         .highlight_max(axis=0, color='#21bc24', subset=['Positive', 'Negative', 'Neutral'])
                         .map(color_verdict, subset=['Verdict']))
            
            st.dataframe(styled_df)
            st.write("### Detailed News Analysis")
            for company, articles in news_data.items():
                with st.expander(f"{company} ({len(articles)} articles)"):
                    for headline, sentiment, link in articles:
                        st.markdown(f"""
                        **{headline}**  
                        Sentiment: `{sentiment}`  
                        [Read Article]({link})  
                        """)
        else:
            st.error("No news found across all sources for the given companies")
        progress_bar.empty()
