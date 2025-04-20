
# 🔍 NewsSureAI

**NewsSureAI** is an AI-powered application that automatically discovers and analyzes news articles on **Climate Risk**, **Insurance**, and **Policy** from trusted sources. It summarizes articles and generates human-readable insights to answer:
- ✅ What changed?
- ✅ Who is affected?
- ✅ Why it matters?

Built with 💡 Hugging Face Transformers, 🧠 spaCy, and 🖥️ Streamlit.

---

## 🚀 Features

- Fetches articles from **NewsData.io** and **SerpAPI**.
- Filters trusted domains using NLP and zero-shot classification.
- Summarizes articles with Hugging Face’s `facebook/bart-large-cnn`.
- Generates key insights using `mistralai/Mixtral-8x7B-Instruct-v0.1`.
- Evaluates summarization quality using **ROUGE metrics**.
- Clean and intuitive **Streamlit UI**.

---

## ⚙️ Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/swapna-d-2698/NewsSureAI.git
cd NewsSureAI
```

### 2. Create `.env` file with your API keys

```dotenv
SERPAPI_KEY=your_serpapi_key
NEWSDATA_API_KEY=your_newsdata_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

### 3. Install dependencies

Install python 3.11.0

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

### 4. Launch the application

```bash
streamlit run news_sure_ai.py
```

---

## 🧪 Example Usage

1. Launch the app.
2. Enter a topic like `"Climate Risk"` or `"InsurTech Policy"`.
3. Choose number of days (e.g. last 2 days).
4. Click **Fetch Latest Articles**.
5. Explore:
   - Summarized articles
   - Human-readable insights
   - ROUGE metrics in sidebar

---

## 📊 Key Performance Metrics

| Metric | Description |
|--------|-------------|
| ✅ Summary Success Rate | % of articles with successful summaries |
| ⚡ Avg Execution Time | Time to fetch + summarize each article |
| 📈 ROUGE-1 / ROUGE-2 / ROUGE-L | Summary quality vs. full text |
| 🎯 Relevance Score | Confidence of classification |

---

## 📎 Tech Stack

- `Streamlit` for UI
- `Hugging Face Transformers API`
- `spaCy` for NER
- `nltk` for sentence tokenization
- `Rouge-score` for evaluation
- `newspaper3k`, `trafilatura`, `bs4` for extraction

---

## 📃 License

MIT License © 2025

---

## 👋 Contact

Built by [Swapna D]. For questions, reach out via [swapnadev2698@gmail.com/GitHub Issues(https://github.com/swapna-d-2698/NewsSureAI/issues)].
