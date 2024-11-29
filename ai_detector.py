# app.py
import streamlit as st
from textblob import TextBlob
import numpy as np
from collections import Counter
import re

def get_sentences(text):
    return re.split('[.!?]+', text)

def analyze_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    
    if len(words) > 1000:
        return {"error": "Text exceeds 1000 words limit"}
    
    sentences = [s.strip() for s in get_sentences(text) if s.strip()]
    blob = TextBlob(text)
    
    avg_sentence_length = len(words) / max(len(sentences), 1)
    unique_words_ratio = len(set(words)) / max(len(words), 1)
    avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
    
    sentiment_scores = [TextBlob(sent).sentiment.polarity for sent in sentences]
    sentiment_variance = np.var(sentiment_scores) if sentiment_scores else 0
    
    phrase_length = 3
    phrases = [' '.join(words[i:i+phrase_length]) for i in range(len(words)-phrase_length+1)]
    phrase_counts = Counter(phrases)
    repetition_ratio = len([p for p in phrase_counts.values() if p > 1]) / max(len(phrases), 1) if phrases else 0
    
    metrics = {
        "sentence_length_score": min(100, (avg_sentence_length / 15) * 100),
        "vocabulary_diversity": unique_words_ratio * 100,
        "word_complexity": min(100, (avg_word_length / 7) * 100),
        "sentiment_consistency": min(100, (1 - sentiment_variance) * 100),
        "repetition_patterns": (1 - repetition_ratio) * 100
    }
    
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
