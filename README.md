# 🚀 AI-Powered Financial Assistant

This project is an AI-powered financial assistant built using **Streamlit, FAISS, LangChain, and Yahoo Finance (yFinance)**. It helps users get stock market insights, financial data, and perform similarity-based searches on financial documents.

---

## 📌 Features

- **💰 Stock Market Analysis**: Fetches real-time and historical financial data using Yahoo Finance (yFinance).
- **🔍 Semantic Search**: Uses FAISS (Facebook AI Similarity Search) for efficient and fast retrieval of relevant financial documents.
- **🧠 AI-Powered Q&A**: Integrates LangChain to enhance question-answering capabilities on financial data.
- **📊 Interactive UI**: Built with Streamlit for a user-friendly web interface.
- **📂 Document Processing**: Allows uploading and processing financial documents for intelligent search and insights.
- **⚡ Fast Performance**: Optimized indexing and query retrieval with FAISS for quick results.

---

## 🛠️ Technologies Used

- **Python** - The core programming language.
- **Streamlit** - For building the interactive web interface.
- **FAISS** - For efficient similarity search.
- **LangChain** - For AI-driven question answering.
- **yFinance** - For fetching stock market data.
- **Pickle** - For serializing and storing FAISS indexes.

---

## 🚀 Getting Started

### 1️⃣ Installation

Make sure you have Python installed. Then, clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/financial-assistant.git
cd financial-assistant

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Running the Application

Once dependencies are installed, run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📈 Usage Guide

1️⃣ **Search Stock Data**
   - Enter a stock ticker symbol (e.g., `AAPL`, `GOOGL`).
   - View real-time and historical stock data.
   
2️⃣ **Upload Financial Documents**
   - Upload a PDF or text file.
   - AI processes and indexes the file using FAISS.
   
3️⃣ **Ask AI Financial Questions**
   - Type a financial query.
   - The AI retrieves relevant insights from indexed documents.

---

