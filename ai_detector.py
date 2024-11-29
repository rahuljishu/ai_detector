# app.py
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import numpy as np
from collections import Counter
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def analyze_text(text):
    # Basic text cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Word count check
    words = word_tokenize(text)
    if len(words) > 1000:
        return {"error": "Text exceeds 1000 words limit"}
    
    # Text analysis
    sentences = sent_tokenize(text)
    blob = TextBlob(text)
    
    # Feature extraction
    avg_sentence_length = len(words) / len(sentences)
    unique_words_ratio = len(set(words)) / len(words)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Sentiment consistency
    sentiment_scores = [TextBlob(sent).sentiment.polarity for sent in sentences]
    sentiment_variance = np.var(sentiment_scores)
    
    # Calculate repeated phrases
    phrase_length = 3
    phrases = [' '.join(words[i:i+phrase_length]) for i in range(len(words)-phrase_length+1)]
    phrase_counts = Counter(phrases)
    repetition_ratio = len([p for p in phrase_counts.values() if p > 1]) / len(phrases) if phrases else 0
    
    # AI detection metrics
    metrics = {
        "sentence_length_score": min(100, (avg_sentence_length / 15) * 100),
        "vocabulary_diversity": unique_words_ratio * 100,
        "word_complexity": min(100, (avg_word_length / 7) * 100),
        "sentiment_consistency": min(100, (1 - sentiment_variance) * 100),
        "repetition_patterns": (1 - repetition_ratio) * 100
    }
    
    # Calculate overall AI probability
    weights = {
        "sentence_length_score": 0.2,
        "vocabulary_diversity": 0.25,
        "word_complexity": 0.2,
        "sentiment_consistency": 0.15,
        "repetition_patterns": 0.2
    }
    
    ai_probability = sum(metrics[key] * weights[key] for key in weights)
    
    return {
        "metrics": metrics,
        "ai_probability": ai_probability,
        "word_count": len(words)
    }

def main():
    st.set_page_config(page_title="AI Text Detector", layout="wide")
    
    st.title("AI Text Detector")
    st.markdown("Upload text (max 1000 words) to analyze if it's AI-generated")
    
    text_input = st.text_area("Enter your text:", height=200)
    
    if st.button("Analyze"):
        if not text_input:
            st.warning("Please enter some text to analyze")
            return
            
        results = analyze_text(text_input)
        
        if "error" in results:
            st.error(results["error"])
            return
            
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("AI Generation Probability", f"{results['ai_probability']:.1f}%")
            st.metric("Word Count", results['word_count'])
        
        with col2:
            st.markdown("### Detailed Metrics")
            metrics = results['metrics']
            for metric, value in metrics.items():
                st.progress(value/100)
                st.caption(f"{metric.replace('_', ' ').title()}: {value:.1f}%")

if __name__ == "__main__":
    main()
