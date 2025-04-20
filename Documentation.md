
# 🧾 NewsSureAI - Documentation

## 📌 Overview

**NewsSureAI** is an intelligent, topic-aware summarization and insight-generation platform designed to fetch, filter, and analyze recent articles related to Climate Risk, InsurTech, and Policy domains. It uses a combination of keyword filtering, NLP techniques, and transformer-based models to produce concise summaries and actionable insights from long-form news articles.

---

## 🧠 Model Architecture & Approach

### Step-by-Step Workflow

1. **User Input**: A topic query (e.g., “climate risk”) and a time range (e.g., past 3 days).
2. **Article Collection**: News articles are fetched from NewsData.io and SerpAPI APIs.
3. **Domain Filtering**:
   - Only articles from **trusted sources** are kept.
   - Each article is checked for **relevance** using:
     - Keyword matching
     - Named Entity Recognition (NER) via `spaCy`
     - Zero-Shot classification using `facebook/bart-large-mnli`
4. **Content Extraction**:
   - Primary: `newspaper3k`
   - Fallbacks: `trafilatura`, BeautifulSoup-based scraping
5. **Summarization**:
   - Model: `facebook/bart-large-cnn` (via Hugging Face API)
   - Content is chunked if too long and summarized in parts
6. **Insight Generation**:
   - Model: `mistralai/Mixtral-8x7B-Instruct-v0.1`
   - Outputs 3 bullet points answering:
     - What changed?
     - Who is affected?
     - Why it matters?
7. **Evaluation**:
   - Summaries are evaluated using **ROUGE-1**, **ROUGE-2**, and **ROUGE-L**.
   - Stats and failed samples are displayed in the sidebar.

---

## 🔧 Instructions

### 1. Setup

- Create a `.env` file with your API keys:

```dotenv
SERPAPI_KEY=your_serpapi_key
NEWSDATA_API_KEY=your_newsdata_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

- Install dependencies:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

### 2. Run the Application

```bash
streamlit run app.py
```

---

## 💡 Example Use Case

- **Topic**: "InsurTech Policy"
- **Days**: 3

### Output:

- 🎯 Filtered 12 relevant articles from trusted domains
- 📃 Summarized each article in under 7 lines
- 💡 Insights Generated:
  - What changed?
  - Who is affected?
  - Why it matters?
- 📈 ROUGE metrics evaluated and displayed in sidebar

---

## 📊 Key Performance Indicators (KPIs)

| Metric | Description |
|--------|-------------|
| ✅ **Summary Success Rate** | % of successfully summarized articles |
| ⏱️ **Avg Execution Time per Article** | Time taken from fetching to insight generation |
| 📈 **ROUGE Scores** | ROUGE-1, ROUGE-2, ROUGE-L to compare summaries vs original text |
| 🎯 **Relevance Detection Accuracy** | Effectiveness of zero-shot filtering |
| 🔍 **Insight Quality** | Subjectively assessed for clarity, depth, and alignment with summary |

---

## 🛠️ Models Used

| Model | Purpose |
|-------|---------|
| `facebook/bart-large-cnn` | Summarization |
| `facebook/bart-large-mnli` | Zero-shot classification |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Insight generation |

---

## 📁 Output Example
```
📌 [Sample Headline About a Critical Global Issue]

🔗 Read Full Article

📅 Published: YYYY-MM-DD

📝 Summary:  
A major issue is unfolding that has significant implications for the region and potentially the world. The event is being driven by a combination of environmental, political, and economic factors. Experts warn that if the situation continues unchecked, it could lead to severe outcomes affecting millions of people. Authorities are urging for immediate action to mitigate the crisis and reduce potential damage.

💡 Key Insights:

- **What changed?** A critical situation is escalating due to growing pressures and insufficient intervention.  
- **Who is affected?** Populations in high-risk or vulnerable areas are most at risk.  
- **Why it matters?** The consequences of inaction could be catastrophic, highlighting the urgency of informed decision-making.
```

---

## 📬 Contact

For collaboration, feedback, or queries, reach out via [GitHub Issues](https://github.com/swapna-d-2698/NewsSureAI/issues) or [swapnadev2698@gmail.com].
