import os
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import requests
import pandas as pd 
import numpy as np
import plotly.express as px
from datetime import datetime


# Load API keys
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")  # Hugging Face API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")  # Groq API Key

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize FAISS vector store (empty by default)
vectorstore = None

# Initialize ChatGroq with Llama3
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

# Streamlit app
st.title("📈 AI-Powered Financial Research Assistant")
st.sidebar.header("🔍 Select Analysis Mode")

# Sidebar Navigation
mode = st.sidebar.radio(
    "Choose an option",
    ["📂 Upload Financial Report", "📊 Real-Time Stock Analysis", "💡 AI-Powered Investment Insights", "📊 Portfolio Tracker", "📊 ETF & Sector Analysis"]
)


# Define the prompt template for Q&A
prompt = ChatPromptTemplate.from_template(
    """
    You are a financial assistant that will help the company.
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question:
    <context>
    {context}
    <context>
    Question: {input}
    """
)

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.express as px

# Define major ETFs for different sectors
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE"
}

# Define top stocks for each sector manually (fallback)
TOP_STOCKS_BY_SECTOR = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Healthcare": ["UNH", "JNJ", "PFE", "LLY", "ABBV"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "Industrials": ["HON", "GE", "CAT", "UPS", "BA"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "NEM"],
    "Utilities": ["NEE", "DUK", "SO", "EXC", "AEP"],
    "Real Estate": ["PLD", "AMT", "SPG", "EQIX", "O"]
}

@st.cache_data
def fetch_etf_data(etf_symbol, start_date, end_date):
    """Fetch historical ETF data with caching."""
    etf = yf.Ticker(etf_symbol)
    return etf.history(start=start_date, end=end_date)


def fetch_sector_news(sector):
    """Fetch the latest news headlines related to a given sector using NewsAPI."""
    news_api_url = f'https://newsapi.org/v2/everything?q={sector}&apiKey=70c61b55871b4ee0b8d0f857d6b1fc89'
    response = requests.get(news_api_url)
    
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        news_list = [f"{idx+1}. {article['title']}  \n  [link]({article['url']})   (Source: {article['source']['name']})"
                     for idx, article in enumerate(articles[:5])]
        return "\n".join(news_list)
    
    return "No recent news available."

def plot_sector_performance():
    """Plots the performance of different sector ETFs."""
    st.title("📊 Sector Performance Analysis")
    
    start_date = st.date_input("Start Date", datetime.datetime(2023, 1, 1).date())
    end_date = st.date_input("End Date", datetime.datetime(2024, 1, 1).date())
    selected_sectors = st.multiselect("Choose Sectors", list(SECTOR_ETFS.keys()), default=["Technology", "Healthcare", "Energy"])
    
    if not selected_sectors:
        st.warning("Please select at least one sector.")
        return
    
    data = []
    for sector in selected_sectors:
        etf_symbol = SECTOR_ETFS[sector]
        df = fetch_etf_data(etf_symbol, start_date, end_date)
        if df.empty:
            st.error(f"Could not fetch data for {sector} ({etf_symbol}).")
            continue
        
        df = df.reset_index()
        df["Sector"] = sector
        df["50-Day MA"] = df["Close"].rolling(window=50).mean()
        df["200-Day MA"] = df["Close"].rolling(window=200).mean()
        data.append(df)
    
    if data:
        combined_df = pd.concat(data, ignore_index=True)
        
        # Plot Sector ETF Performance
        fig = px.line(combined_df, x="Date", y="Close", color="Sector", title="Sector ETF Performance")
        fig.update_layout(xaxis_title="Date", yaxis_title="Closing Price", legend_title="Sector")
        st.plotly_chart(fig)
        
        # Display Performance Summary
        performance_df = combined_df.groupby("Sector")["Close"].agg(["first", "last"])
        performance_df["% Change"] = ((performance_df["last"] - performance_df["first"]) / performance_df["first"]) * 100
        st.write("### Sector Performance Summary")
        st.dataframe(performance_df.rename(columns={"first": "Start Price", "last": "End Price"}))
        
        # Display Moving Averages
        st.write("### Moving Averages")
        for sector in selected_sectors:
            sector_df = combined_df[combined_df["Sector"] == sector]
            fig_ma = px.line(sector_df, x="Date", y=["Close", "50-Day MA", "200-Day MA"], title=f"{sector} - Moving Averages")
            st.plotly_chart(fig_ma)
        
        # Display Correlation Matrix
        pivot_df = combined_df.pivot(index="Date", columns="Sector", values="Close")
        correlation_matrix = pivot_df.corr()
        st.write("### Sector Correlation Matrix")
        st.dataframe(correlation_matrix)
        
        # Display Trading Volume Trends
        st.write("### Trading Volume Trends")
        fig_volume = px.line(combined_df, x="Date", y="Volume", color="Sector", title="Sector ETF Volume Trends")
        st.plotly_chart(fig_volume)
        
        # Display News
        st.write("### News in Selected Sectors")
        for sector in selected_sectors:
            stock_news = fetch_sector_news(sector)
            st.write(f"#### {sector} - News")
            st.write(stock_news)

# Run Sector Analysis if selected
if mode == "📊 ETF & Sector Analysis":
    plot_sector_performance()


# Function to save uploaded files
def save_uploaded_file(uploaded_file):
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to initialize the vectorstore
def create_vector_embedding(file_paths):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.docs = []
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)  # Assuming PDFs are uploaded
            st.session_state.docs.extend(loader.load())  # Load all documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


# Function to load and process documents
def process_financial_report(file):
    global vectorstore
    temp_filename = "uploaded_report.txt"
    with open(temp_filename, "wb") as f:
        f.write(file.getvalue())

    loader = TextLoader(temp_filename)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)

    return "Financial report successfully processed and stored for AI-based analysis!"

# Upload financial report option
if mode == "📂 Upload Financial Report":
    uploaded_file = st.file_uploader("Upload a financial report (TXT or PDF format)", type=["txt", "pdf"])
    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)  # Save the file to a directory
        st.success("File uploaded successfully!")

        # Initialize vector embeddings and create vectorstore
        create_vector_embedding([file_path])
        st.success("Vector database initialized with the uploaded files!")

        # User input for Q&A from the uploaded report
        user_question = st.text_input("Ask a question about the uploaded financial report:")
        if user_question:
            if "vectors" in st.session_state:
                retriever = st.session_state.vectors.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="map_reduce")
                start = time.process_time()
                response = qa_chain.run(user_question)
                elapsed_time = time.process_time() - start
                st.write(f"Response Time: {elapsed_time:.2f} seconds")
                st.write(response)

            else:
                st.error("Please upload and initialize the vector database first.")

# Function to generate AI insights
def generate_insights(query):
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")
    return llm.invoke(query)

# Function to fetch real-time stock data
def get_stock_1data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo")  # Last 1 month data
    return data

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="6mo")  # Last 6 months data
    info = stock.info
    return data, info

def fetch_stock_news(ticker_symbol):
    """Fetch the latest news headlines for a given stock using yfinance."""
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news

    if news:
        news_list = []
        for idx, article in enumerate(news[:5], start=1):  # Get the top 5 articles
            title = article.get('content', {}).get('title', 'No title available')
            link = article.get('content', {}).get('canonicalUrl', {}).get('url', 'No link available')
            news_list.append(f"{idx}. Title: {title}\n   Link: {link}\n")
        return news_list
    return ["No news available for this stock."]

# Function to analyze and compare stocks
def compare_stocks(ticker1, ticker2):
    st.write(f"### 📈 Comparing {ticker1} vs {ticker2}")

    # Fetch stock data
    stock1_data, stock1_info = get_stock_data(ticker1)
    stock2_data, stock2_info = get_stock_data(ticker2)

    # Extract key financial metrics
    metrics = ["marketCap", "trailingPE", "forwardPE", "dividendYield", "beta", "52WeekChange"]
    stock1_metrics = {metric: stock1_info.get(metric, "N/A") for metric in metrics}
    stock2_metrics = {metric: stock2_info.get(metric, "N/A") for metric in metrics}

    # Display key financial metrics
    st.write("#### 📊 Key Financial Metrics")
    comparison_table = f"""
    | Metric          | {ticker1} | {ticker2} |
    |----------------|----------|----------|
    | Market Cap ($B) | {round(stock1_metrics['marketCap'] / 1e9, 2) if stock1_metrics['marketCap'] != "N/A" else "N/A"} | {round(stock2_metrics['marketCap'] / 1e9, 2) if stock2_metrics['marketCap'] != "N/A" else "N/A"} |
    | P/E Ratio       | {stock1_metrics['trailingPE']} | {stock2_metrics['trailingPE']} |
    | Forward P/E     | {stock1_metrics['forwardPE']} | {stock2_metrics['forwardPE']} |
    | Dividend Yield  | {stock1_metrics['dividendYield']} | {stock2_metrics['dividendYield']} |
    | Beta (Volatility) | {stock1_metrics['beta']} | {stock2_metrics['beta']} |
    | 52-Week Change | {round(stock1_metrics['52WeekChange'] * 100, 2) if stock1_metrics['52WeekChange'] != "N/A" else "N/A"}% | {round(stock2_metrics['52WeekChange'] * 100, 2) if stock2_metrics['52WeekChange'] != "N/A" else "N/A"}% |
    """
    st.markdown(comparison_table)

    # Plot stock trends
    st.write("#### 📉 Stock Performance (Last 6 Months)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock1_data.index, stock1_data["Close"], label=ticker1, color="blue")
    ax.plot(stock2_data.index, stock2_data["Close"], label=ticker2, color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price ($)")
    ax.set_title(f"{ticker1} vs {ticker2} - Stock Trend")
    ax.legend()
    st.pyplot(fig)

    # AI-generated insights
    analysis_prompt = f"""
    Compare the following two stocks based on their market performance, financial health, and investment potential:
    - {ticker1}: Market Cap ${stock1_metrics['marketCap']}, P/E Ratio {stock1_metrics['trailingPE']}, Forward P/E {stock1_metrics['forwardPE']}, Beta {stock1_metrics['beta']}, 52-Week Change {stock1_metrics['52WeekChange']}
    - {ticker2}: Market Cap ${stock2_metrics['marketCap']}, P/E Ratio {stock2_metrics['trailingPE']}, Forward P/E {stock2_metrics['forwardPE']}, Beta {stock2_metrics['beta']}, 52-Week Change {stock2_metrics['52WeekChange']}

    Provide a detailed analysis of which stock is a better investment in terms of risk, growth potential, and market performance.
    """
    ai_insight = generate_insights(analysis_prompt)
    insight_text = ai_insight.content if hasattr(ai_insight, "content") else str(ai_insight)


    st.write("# 🤖 AI-Powered Investment Analysis")
    st.write(insight_text)


# Real-time stock analysis option
if mode == "📊 Real-Time Stock Analysis":
    st.write("### 🔍 Enter stock ticker(s) for analysis:")
    ticker1 = st.text_input("Stock 1 (e.g., AAPL, TSLA)")
    ticker2 = st.text_input("Stock 2 (optional, for comparison)")

    if st.button("Analyze Stocks"):
        if not ticker1:
            st.warning("Please enter at least one stock ticker.")
        else:
            stock1_data = get_stock_1data(ticker1)
            st.write(f"### 📰 {ticker1} News")
            news_headlines = fetch_stock_news(ticker1)
            for headline in news_headlines:
                st.write(f"- {headline}")
            st.write(f"### 📊 {ticker1} Stock Performance (Last 1 Month)")
            st.line_chart(stock1_data["Close"])

            if ticker2:
                stock2_data = get_stock_1data(ticker2)
                st.write(f"### 📰 {ticker2} News")
                news_headlines = fetch_stock_news(ticker2)
                for headline in news_headlines:
                    st.write(f"- {headline}")
                st.write(f"### 📊 {ticker2} Stock Performance (Last 1 Month)")
                st.line_chart(stock2_data["Close"])
                st.write("## Here is the Comparison Of The two stocks")
                compare_stocks(ticker1, ticker2)


if mode =="💡 AI-Powered Investment Insights":
    st.write("### 💡 AI-Powered Investment Insights")
    query_text = st.text_input("Ask about market trends, investment strategies, or stock insights:")
    if query_text:
        response = generate_insights(query_text)
        response_text = response.content if hasattr(response, "content") else str(response)
        st.write(response_text)


# Initialize the portfolio in session state if not already initialized
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# Portfolio Tracker
if mode == "📊 Portfolio Tracker":
    st.write("### 🛠️ Manage Your Portfolio")
    
    # Input stock ticker and purchase details
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)")
    shares = st.number_input("Enter Number of Shares", min_value=1, step=1)
    purchase_price = st.number_input("Enter Purchase Price per Share ($)", min_value=0.01, step=0.01)
    add_button = st.button("Add Stock to Portfolio")
    
    if add_button:
        if ticker and shares and purchase_price:
            stock_data, _ = get_stock_data(ticker)
            current_price = stock_data["Close"].iloc[-1]
            st.session_state.portfolio.append({
                "ticker": ticker, 
                "shares": shares, 
                "purchase_price": purchase_price, 
                "current_price": current_price
            })
            st.success(f"Added {shares} shares of {ticker} at {purchase_price} per share to your portfolio!")

    # Display portfolio if it contains stocks
    if st.session_state.portfolio:
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        
        # Calculate current value and performance
        portfolio_df['current_value'] = portfolio_df['shares'] * portfolio_df['current_price']
        portfolio_df['purchase_value'] = portfolio_df['shares'] * portfolio_df['purchase_price']
        portfolio_df['growth'] = (portfolio_df['current_value'] - portfolio_df['purchase_value']).round(2)
        portfolio_df['performance'] = (portfolio_df['growth'] / portfolio_df['purchase_value']) * 100

        st.write("### 📊 Portfolio Overview")
        st.dataframe(portfolio_df)

        # Calculate total portfolio value and total growth
        total_value = portfolio_df['current_value'].sum()
        total_purchase_value = portfolio_df['purchase_value'].sum()
        total_growth = total_value - total_purchase_value
        total_performance = (total_growth / total_purchase_value) * 100

        st.write(f"### Total Portfolio Value: ${total_value:,.2f}")
        st.write(f"### Total Growth: ${total_growth:,.2f}")
        st.write(f"### Total Performance: {total_performance:.2f}%")

        # Visualization: Portfolio Distribution by Ticker
        st.write("### 📊 Portfolio Distribution by Ticker")
        portfolio_pie = portfolio_df.groupby('ticker')['current_value'].sum()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(portfolio_pie, labels=portfolio_pie.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.set_title("Portfolio Allocation")
        st.pyplot(fig)

        # Visualization: Performance of Individual Stocks
        st.write("### 📊 Performance of Individual Stocks")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(portfolio_df['ticker'], portfolio_df['performance'], color=plt.cm.Paired.colors)
        ax.set_xlabel("Stock Ticker")
        ax.set_ylabel("Performance (%)")
        ax.set_title("Stock Performance in Portfolio")
        st.pyplot(fig)
        
    else:
        st.write("No stocks in your portfolio yet.")