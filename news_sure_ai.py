import requests
import time
from newspaper import Article
import feedparser
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urlparse
from dateutil import parser
from serpapi import GoogleSearch
from transformers import pipeline
import nltk
import trafilatura
import spacy
# Make sure you have this
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import os
from huggingface_hub import InferenceClient
from nltk.tokenize import sent_tokenize
import re
from rouge_score import rouge_scorer
import random
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Load spaCy and zero-shot model
nlp = spacy.load("en_core_web_sm")

# Access the keys
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize API URLs for Hugging Face Inference
SUMMARIZATION_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
ZERO_SHOT_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

INSIGHT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

KEYWORDS = [
    "climate", "climate change", "global warming", "carbon", "emissions",
    "extreme weather", "natural disasters", "flood", "drought", "wildfire", "hurricane",
    "climate risk", "climate impact", "environment", "greenhouse gas", "sustainability",
    "resilience", "adaptation", "climate resilience", "insurance", "insurtech", "reinsurance",
    "coverage", "policy", "premium", "claim", "risk", "risk assessment", "risk exposure", 
    "underwriting", "loss", "regulation", "policy", "compliance", "climate law", "climate policy",
    "financial risk", "climate finance", "transition risk", "net zero",
    "InsurTech startup", "insurance technology", "digital underwriting", "blockchain insurance", 
    "AI insurance", "IoT insurance", "claims automation", "cyber insurance", "peer-to-peer insurance", 
    "insurance policy", "insurance market", "reinsurance market", "underwriting model", "reinsurance policy", 
    "catastrophic risk", "risk pool", "reinsurance treaty", "regulatory compliance in insurance",
    "green insurance", "sustainable finance in insurance", "climate risk investment", "impact investment insurance", 
    "ESG insurance", "green bonds in insurance", "climate change and reinsurance"
]

# Expanded Relevant Labels
RELEVANT_LABELS = [
    "climate risk", "climate change impact", "insurance and risk", "exposure to climate disasters",
    "climate policy", "sustainability", "natural disaster coverage", "financial risk management",
    "climate insurance", "green finance", "regulatory frameworks", "climate adaptation strategies",
    "risk mitigation", "weather-related insurance claims", "environmental risk",
    "InsurTech innovation", "Digital insurance transformation", "Insurance policy development", 
    "Reinsurance industry trends", "Sustainable finance in insurance", "Impact investing in insurance", 
    "Climate risk insurance products", "Insurance regulatory framework"
]

GENERIC_TERMS = {"risk", "policy", "impact", "loss", "coverage", "event", "exposure"}

# Expanded Trusted Domains for Insurance & InsurTech
TRUSTED_DOMAINS = [
    "forbes.com", "reuters.com", "climatechangenews.com", "sciencedirect.com", "jstor.org", "link.springer.com",
    "theguardian.com", "ft.com", "techcrunch.com", "carriermanagement.com", "businessinsurance.com",
    "intelligentinsurer.com", "insurtechdigital.com", "unepfi.org", "oecd.org", "weforum.org", "worldbank.org",
    "bloomberg.com", "insurancejournal.com", "bbc.com", "cnn.com", "onlinelibrary.wiley.com", "ssrn.com",
    "nature.com", "carbonbrief.org", "grist.org", "climate.gov", "climatepolicyinitiative.org", "earthobservatory.nasa.gov",
    "aon.com", "marsh.com", "reinsurance.news", "insurancebusinessmag.com", "cnbc.com", "insuranceinnovations.com",
    "fsb.org", "unfccc.int", "ec.europa.eu", "gov.uk", "cftc.gov", "rand.org", "brookings.edu", "chathamhouse.org",
    "carnegieendowment.org", "msci.com", "spglobal.com", "bnef.com", "sciencedaily.com", "phys.org", "scientificamerican.com",
    "insurtechnews.com", "techcrunch.com", "insuranceinnovators.com", "carrier-management.com", "finextra.com", 
    "insurancenews.com.au", "insurancetech.com", "finextra.com", "insurtechdigital.com", "straitstimes.com",
    "insuranceeurope.eu", "insuranceinsider.com", "insuranceage.co.uk", "iii.org", "naic.org", "insurance.ca.gov",
    "fca.org.uk", "insuranceERM.com", "digitalinsurance.com", "insurancepost.com", "theactuary.com", "verisk.com",
    "climateactiontracker.org", "climateanalytics.org", "climatebonds.net", "ipcc.ch", "un.org", "who.int", "e3g.org",
    "cdp.net", "seia.org", "irena.org", "nrel.gov", "energy.gov", "climatecentral.org", "resilience.org", "adelphi.de",
    "rff.org", "edf.org", "bis.org", "imf.org", "eba.europa.eu", "bankofengland.co.uk", "nber.org", "nasdaq.com",
    "worldeconomicforum.org", "ecb.europa.eu", "ngfs.net", "cambridge.org", "nytimes.com", "washingtonpost.com", "apnews.com",
    "npr.org", "axios.com", "latimes.com", "economist.com", "time.com", "newsweek.com", "usatoday.com", "abcnews.go.com",
    "cbc.ca", "globalnews.ca", "aljazeera.com", "dw.com", "thehindu.com", "indiatoday.in", "hindustantimes.com", "japantimes.co.jp",
    "prnewswire.com", "benzinga.com", "nbcphiladelphia.com", "chicagotribune.com", "globenewswire.com"
]

error_strings = [
    "error", "503", "service unavailable", "unavailable", "timeout",
    "failed", "invalid", "not found", "null", "none", "exception",
    "bad gateway", "internal server", "empty summary", "could not generate",
    "rate limit", "unauthorized", "forbidden", "connection refused"
]

def classify_zero_shot(text, candidate_labels):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"
    }
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": candidate_labels
        }
    }
    try:
        response = requests.post(ZERO_SHOT_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Extract highest score + label
            if "scores" in result and len(result["scores"]) > 0:
                return {
                    "label": result["labels"][0],
                    "score": result["scores"][0],
                    "raw": result
                }
        else:
            print(f"[Zero-Shot Error] {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[Zero-Shot Exception] {e}")
    
    return None

# Function to check if URL belongs to a trusted domain
def is_trusted_url(url):
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return any(trusted in domain for trusted in TRUSTED_DOMAINS)
    except:
        return False

# Fetch articles from NewsData.io
def fetch_newsdata_articles(query, num_days, max_pages=5):
    print(f"Fetching articles from NewsData.io for query: {query}")
    
    base_url = "https://newsdata.io/api/1/news"
    articles = []
    query = query.strip()

    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": "en",
        "country": "us",
        "category": "business,science,technology"
    }

    page_count = 0
    next_page_token = None

    while page_count < max_pages:
        if next_page_token:
            params["page"] = next_page_token

        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print("❌ NewsData API error:", response.status_code, response.text)
            return []

        data = response.json()
        results = data.get("results", [])
        print(f"✅ Page {page_count + 1}: {len(results)} articles")

        for item in results:
            title = item.get("title")
            link = item.get("link")
            pub_date_str = item.get("pubDate")

            if not link or not pub_date_str:
                print(f"⚠️ Skipping (missing link/date): {title}")
                continue

            if not is_trusted_url(link):
                print(f"🚫 Skipping (untrusted domain): {link}")
                continue

            try:
                pub_date = parser.parse(pub_date_str)
                age = datetime.now() - pub_date
                if age < timedelta(days=num_days):
                    articles.append({
                        "title": title,
                        "link": link,
                        "pubDate": pub_date.strftime("%Y-%m-%d")
                    })
                else:
                    print(f"📅 Skipping (too old): {title} | Date: {pub_date_str}")
            except Exception as e:
                print(f"⚠️ Error parsing date for '{title}': {e}")

        next_page_token = data.get("nextPage")
        if not next_page_token:
            break

        page_count += 1

    print(f"✅ Total articles collected: {len(articles)}")
    return articles
    
# Fetch articles from SerpAPI
def fetch_serpapi_articles(query, num_days):
    print(f"Fetching articles from SerpAPI for query: {query}")
    params = {
        "engine": "google",
        "q": query,
        "tbm": "nws",
        "api_key": SERPAPI_KEY,
        "num": 100
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    articles = []
    if "news_results" in results:
        for item in results["news_results"]:
            link = item.get("link")
            title = item.get("title")
            date_str = item.get("date")

            if not link or not title or not date_str:
                continue

            if is_trusted_url(link):
                try:
                    # Try to parse the date using dateutil.parser, which can handle relative dates
                    pub_date = parser.parse(date_str, fuzzy=True)
                    
                    # Check if the date is within the desired range
                    if datetime.now() - pub_date < timedelta(days=num_days):
                        articles.append({
                            "title": title,
                            "link": link,
                            "pubDate": pub_date.strftime("%Y-%m-%d")
                        })
                except Exception as e:
                    print(f"Error parsing SerpAPI date for '{title}': {e}")
    else:
        print("No news_results found in SerpAPI response or query failed.")

    return articles

def extract_with_custom_user_agent(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        article = Article(url)
        article.download(input_html=response.text)
        article.parse()
        return article.title, article.text
    except Exception as e:
        print(f"Custom UA failed: {e}")
        return None, None

def extract_content_trafilatura(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            result = trafilatura.extract(downloaded, include_title=True, include_comments=False)
            if result:
                title = url.split("/")[-1].replace("-", " ").title()
                return title, result
        return None, None
    except Exception as e:
        print(f"Trafilatura failed: {e}")
        return None, None
    
def extract_article_content(url):
    try:
        # --- 1. Try Newspaper3k ---
        article = Article(url)
        article.download()
        article.parse()
        title = article.title
        text = article.text

        if len(text.split()) >= 50:
            return title.strip(), text.strip()

        # --- 2. Fallback to Newspaper3k with Custom User-Agent ---
        print(f"Fallback to Newspaper3k with custom UA: {url}")
        title, text = extract_with_custom_user_agent(url)
        if text and len(text.split()) >= 50:
            return title.strip(), text.strip()

        # --- 3. Final Fallback to Trafilatura ---
        print(f"Final fallback to Trafilatura: {url}")
        title, text = extract_content_trafilatura(url)
        if text and len(text.split()) >= 50:
            return title.strip(), text.strip()

        print(f"All extraction methods failed or content too short: {url}")
        return None, None

    except Exception as e:
        print(f"Unexpected error for {url}: {e}, trying manual scraping...")
        # Try fallback
        fallback_title, fallback_text = manual_scrape_fallback(url)
        if fallback_title and fallback_text:
            print(f"Successfully extracted using fallback for {url}")
            return fallback_title, fallback_text
        else:
            print(f"⚠️ Failed to extract content for article: {url}")
            return None, None
    
def manual_scrape_fallback(url):
    print(f"Fallback to manual scraping for: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
                  "image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Failed to fetch page content: {response.status_code}")
            return None, None

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        else:
            print(f"⚠️ No paragraphs found for {url}")
            return None, None

        title = soup.title.string if soup.title and soup.title.string else "No Title"

        if len(text.split()) < 50:
            print(f"⚠️ Content is too short for summarization: {url}")
            return None, None

        return title.strip(), text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None, None
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None, None

def is_relevant_article(text, user_query=None, use_zero_shot=True):
    text_lower = text.lower()
    query_keywords = user_query.lower().split() if user_query else []

    contextual_keywords = set(KEYWORDS + query_keywords) - GENERIC_TERMS

    # Phase 1: Exact keyword co-occurrence (contextual AND domain-specific)
    if all(kw in text_lower for kw in query_keywords if len(kw) > 3):
        if any(domain_kw in text_lower for domain_kw in KEYWORDS):
            return True

    # Phase 2: Named Entity-based strong match
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"ORG", "GPE", "EVENT", "PRODUCT", "LAW"}:
            if any(kw in ent.text.lower() for kw in contextual_keywords):
                return True

    # Phase 3: Zero-shot classification with stricter threshold and filter
    if use_zero_shot:
        candidate_labels = list(set(RELEVANT_LABELS + query_keywords))
        result = classify_zero_shot(text, candidate_labels)

        if result and isinstance(result, dict) and result.get("score", 0) > 0.75:
            top_label = result["label"].lower()
            if any(kw in top_label for kw in query_keywords):
                return True

    return False

# Function to call Hugging Face API for Summarization
def summarize_using_api(text, max_len=100, min_len=60):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"
    }
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_len,
            "min_length": min_len,
            "do_sample": False
        }
    } 
    try:
        # Make API call for summarization
        response = requests.post(SUMMARIZATION_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            summary = response.json()[0]['summary_text']
            return summary
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        print(f"Error in summarization: {e}")
        return "Error in summarization."

def summarize_using_api_with_retry(chunk, max_len, min_len, retries=5, delay=1):
    for attempt in range(retries):
        try:
            summary = summarize_using_api(chunk, max_len, min_len)
            return summary
        except Exception as e:
            if "503" in str(e):
                # Exponential backoff: wait longer on each retry attempt
                time.sleep(delay * (2 ** attempt) + random.uniform(0, 1))  # random jitter
            else:
                raise e
    return "Error: Failed to summarize after multiple attempts."

# Modify the summarize_article function to use this retry logic:
def summarize_article(content):
    if not content or len(content.strip()) == 0:
        return "Content is too short to summarize."
    try:
        max_chunk_words = 500
        sentences = sent_tokenize(content)
        chunks = []
        chunk = ""
        
        # Split content into chunks with <= max_chunk_words
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= max_chunk_words:
                chunk += " " + sentence
            else:
                chunks.append(chunk.strip())
                chunk = sentence
        if chunk:
            chunks.append(chunk.strip())

        # Estimate how much each chunk should summarize to stay within the number of lines
        target_total_sentences = 7  # You can adjust this target if needed
        summaries = []
        sentences_per_chunk = max(1, target_total_sentences // len(chunks))

        # Loop through each chunk and summarize it
        for chunk in chunks:
            input_len = len(chunk.split())
            max_len = min(100, int(input_len * 0.4))
            min_len = min(60, int(input_len * 0.2))
            max_len = max(30, max_len)
            min_len = max(10, min_len)

            # Call the Hugging Face API for summarization with retry logic
            summary = summarize_using_api_with_retry(chunk, max_len, min_len)
            truncated_summary = " ".join(sent_tokenize(summary)[:sentences_per_chunk])
            summaries.append(truncated_summary)

        return " ".join(summaries)

    except Exception as e:
        print(f"Error summarizing article: {e}")
        return "Error during summarization."

def query_hf_model(payload, model):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_insights(summary):
    prompt = f"""
            Given the summary below, generate exactly 3 bullet points that provide:
            1. What changed?  
            2. Who is affected?  
            3. Why it matters.

            Each bullet must be:
            - within 25 words.
            - Clear and free of numbering or question repetition.
            - Easy to understand (avoid technical jargon unless necessary).

            Only return the 3 bullet points.

            Summary:
            {summary}

            Insights:
            """
    try:
        output = query_hf_model({"inputs": prompt}, INSIGHT_MODEL)

        # Validate output type
        if not isinstance(output, list) or not output or 'generated_text' not in output[0]:
            print(f"⚠️ Unexpected model response: {output}")
            return ["Insight generation failed."]

        generated = output[0]['generated_text']
        if "Insights:" in generated:
            generated = generated.split("Insights:")[1]

        # Extract bullet points cleanly
        lines = [line.strip("-• ").strip() for line in generated.split("\n") if line.strip()]
        return lines[:3] if lines else ["No insights generated."]

    except Exception as e:
        print(f"⚠️ Error in generate_insights: {e}")
        return ["Insight generation failed due to exception."]

# Helper function to check if summary is valid (i.e., not an error or invalid summary)
def is_summary_valid(summary):
    return summary and not any(err in summary.lower() for err in error_strings)

# Method to evaluate and display ROUGE scores
def evaluate_and_display_rouge(enriched_articles):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_scores = []
    extraction_success = 0
    total_time = 0
    invalid_summaries = []  # To track failed summaries for debugging

    # Iterate over each article and calculate ROUGE scores
    for enriched_article in enriched_articles:
        start_time = time.time()

        # Extract reference summary and full text
        reference = enriched_article.get("summary", "").strip()
        extracted_text = enriched_article.get("full_text", "").strip()

        # Check if both summary and full text are valid and not containing errors
        if extracted_text and is_summary_valid(reference):
            scores = scorer.score(reference, extracted_text)  # Compare generated vs extracted text
            rouge_scores.append(scores)
            extraction_success += 1
        else:
            # Collect failed summaries for debugging
            if reference:
                invalid_summaries.append({
                    "title": enriched_article.get("title"),
                    "summary": reference
                })

        end_time = time.time()
        total_time += (end_time - start_time)

    # Calculate averages
    avg_execution_time = total_time / len(enriched_articles) if enriched_articles else 0
    success_rate = (extraction_success / len(enriched_articles)) * 100 if enriched_articles else 0

    # Displaying results in Streamlit's sidebar
    st.sidebar.markdown(f"<div style='line-height: 1.3; font-size: 14px;'>"
                        f"<strong>Articles Processed :<strong> {len(enriched_articles)} <br> "
                        f"<strong>Success Rate :<strong> {success_rate:.2f}% <br> "
                        f"<strong>Avg Execution Time per Article :<strong> {avg_execution_time:.2f} sec"
                        "</div> <br> ", unsafe_allow_html=True)

    # Compute & Display ROUGE Scores using Markdown
    if rouge_scores:
        avg_rouge1 = sum(score["rouge1"].fmeasure for score in rouge_scores) / len(rouge_scores)
        avg_rouge2 = sum(score["rouge2"].fmeasure for score in rouge_scores) / len(rouge_scores)
        avg_rougeL = sum(score["rougeL"].fmeasure for score in rouge_scores) / len(rouge_scores)

        st.sidebar.markdown(f"<div style='line-height: 1.3; font-size: 14px;'>"
                            f"<strong>ROUGE Scores :<strong> <br>"
                            f"ROUGE-1 : {avg_rouge1:.4f} <br> "
                            f"ROUGE-2 : {avg_rouge2:.4f}  <br> "
                            f"ROUGE-L : {avg_rougeL:.4f}"
                            "</div>", unsafe_allow_html=True)
    else:
        st.sidebar.write("⚠️ No valid articles processed for ROUGE evaluation.")

    # If any summaries failed, display them
    if invalid_summaries:
        st.sidebar.markdown("<br><strong>⚠️ Failed Summaries:</strong>", unsafe_allow_html=True)
        for article in invalid_summaries:
            st.sidebar.markdown(f"<div style='font-size: 12px;'>"
                                f"<strong>Title:</strong> {article['title']} <br>"
                                f"<strong>Summary:</strong> {article['summary'][:200]}...</div> <br>", unsafe_allow_html=True)
    
# Streamlit UI Setup
st.title("NewsSureAI")
st.write("Browse the latest insights on Climate Risk, Insurance, and Policies from trusted sources.")

# User input for topic
query = st.text_input("Enter your topic (e.g., Climate Risk, InsureTech Policy)", "Climate Risk")
# Sidebar input for number of days
col1, col2 = st.columns(2)
with col1:
    num_days = st.number_input("Select the number of days for fetching articles:", min_value=1, max_value=30, value=2)

# Streamlit UI code for fetching articles
if st.button("Fetch Latest Articles"):
    with st.spinner(f"Fetching articles related to '{query}'..."):
        newsdata_articles = fetch_newsdata_articles(query, num_days)
        serpapi_articles = []

        print(f"Fetched {len(newsdata_articles)} from NewsData.io, and {len(serpapi_articles)} from SerpAPI.")
        
        # Combine and deduplicate by link
        seen_links = set()
        all_articles = []

        # Loop through articles and deduplicate by link
        for source_articles in [newsdata_articles, serpapi_articles]:
            for article in source_articles:
                if article['link'] not in seen_links:
                    seen_links.add(article['link'])
                    all_articles.append(article)

        enriched_articles = []

        # Process each article and summarize
        for article in all_articles:
            title, content = extract_article_content(article['link'])
            
            if content:
                print(f"🔎 Processing: {title}")
                try:
                    if is_relevant_article(title, content, query):
                        print(f"✅ Relevant article found: {title}")
                        # Summarize the content
                        summary = summarize_article(content)
                        print(f"📝 Summary: {summary}")
                        insights = generate_insights(summary)
                        print(f"💡 Insights: {insights}")
                        enriched_articles.append({
                            "title": title,
                            "link": article['link'],
                            "pubDate": article['pubDate'],
                            "full_text": content,
                            "summary": summary,
                            "insights": insights
                        })
                    else:
                        print(f"Skipped irrelevant article:: {title}")
                except Exception as e:
                    print(f"⚠️ Failed to summarize article: {e}")
            else:
                print(f"⚠️ Failed to extract content for article: {article['title']}")

        if not enriched_articles:
            st.warning(f"No readable content found for '{query}'.")
        else:
            st.success(f"Found {len(enriched_articles)} articles with readable content.")

        # Display all article titles first with expandable details
        for article in enriched_articles:
            with st.expander(f"📌 {article['title']}"):
                st.markdown(f"🔗 [Read Full Article]({article['link']})")
                st.markdown(f"📅 **Published:** {article['pubDate']}")
                st.markdown(f"📝 **Summary:** {article['summary']}")

                insights = article.get('insights', [])
                insights = [re.sub(r'^\d+\.\s*', '', insight) for insight in insights]

                if insights:
                    st.markdown("💡 **Key Insights:**")
                    html_output = "<div style='line-height: 1.3;'><ul>"

                    if len(insights) > 0:
                        html_output += f"<li><strong>What changed?</strong> {insights[0]}</li>"
                    if len(insights) > 1:
                        html_output += f"<li><strong>Who is affected?</strong> {insights[1]}</li>"
                    if len(insights) > 2:
                        html_output += f"<li><strong>Why it matters?</strong> {insights[2]}</li>"

                    # If more than 3 insights exist, list the rest normally
                    if len(insights) > 3:
                        for i in insights[3:]:
                            html_output += f"<li>{i}</li>"

                    html_output += "</ul></div>"
                    st.markdown(html_output, unsafe_allow_html=True)

        # Sidebar evaluation
        with st.sidebar:
            st.write("**ROUGE Evaluation for News Article Summaries**")
            evaluate_and_display_rouge(enriched_articles)
                