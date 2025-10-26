# final_sentiment.py
"""
Sentiment analysis for comment files in NDJSON format.
Adds VADER/TextBlob metrics plus trust/distrust flags, and writes full results
and a per-year summary including positive/neutral/negative counts and trust metrics.
Also outputs counts of how many times each trust/distrust keyword was flagged,
and generates a 1000-comment subsample per year for manual review.
"""
import glob
import json
import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Ensure VADER lexicon is available
nltk.download('vader_lexicon', quiet=True)

# Keywords for trust/distrust filtering (original order)
distrust_terms = [
    'lying', 'lies', "don’t trust", "don't trust", 'corrupt',
    'propaganda', 'misleading', 'brainwash', 'cover-up', 'fake',
    'hoax', 'scam', 'deceive'
]
trust_terms = [
    'i trust', 'honest', 'truthful', 'reliable', 'credible',
    'transparent', 'always tell the truth', 'i believe them',
    'believe it', 'trust', 'genuine'
]


def load_all_comments(patterns=('comments_*.ndjson',), max_per_year=10000, removed_path='removed_comments.ndjson'):
    # Load raw comments and attach year
    records = []
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    for path in sorted(set(paths)):
        year = int(re.search(r'comments_(\d{4})\.ndjson', path).group(1))
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                obj['year'] = year
                records.append(obj)
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Identify and separate removed comments (no letter characters)
    mask = df['text'].str.contains(r"[^\W\d_]", na=False, regex=True)
    df_removed = df[~mask].copy()
    df_kept = df[mask].copy()

    # Analyze sentiment on removed set so metrics are included
    df_removed = analyze_sentiment(df_removed)

    # Write removed comments with full metrics to NDJSON
    with open(removed_path, 'w', encoding='utf-8') as rf:
        for rec in df_removed.to_dict(orient='records'):
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"🗑️ Wrote {len(df_removed)} removed comments (with metrics) to {removed_path}", flush=True)

    # Cap to max_per_year per year on kept comments
    df_capped = (
        df_kept
        .sort_values(['year', 'published_at'], na_position='last')
        .groupby('year', group_keys=False)
        .head(max_per_year)
    )
    return df_capped


def analyze_sentiment(df):
    """
    Add VADER, TextBlob, and trust/distrust flags to DataFrame.
    """
    if df.empty:
        print("No comments to analyze.")
        return df
    vader = SentimentIntensityAnalyzer()
    scores = df['text'].apply(vader.polarity_scores)
    df['vader_neg']      = scores.map(lambda s: s['neg'])
    df['vader_neu']      = scores.map(lambda s: s['neu'])
    df['vader_pos']      = scores.map(lambda s: s['pos'])
    df['vader_compound'] = scores.map(lambda s: s['compound'])
    df['tb_polarity']     = df['text'].apply(lambda t: TextBlob(t).sentiment.polarity)
    df['tb_subjectivity'] = df['text'].apply(lambda t: TextBlob(t).sentiment.subjectivity)
    df['text_lower'] = df['text'].str.lower().fillna('')
    df['is_distrust'] = df['text_lower'].apply(lambda t: any(term in t for term in distrust_terms))
    df['is_trust']    = df['text_lower'].apply(lambda t: any(term in t for term in trust_terms))
    return df


def generate_summary(df, summary_path='sentiment_summary_per_year.csv'):
    def label_sentiment(c):
        if c >= 0.05: return 'positive'
        if c <= -0.05: return 'negative'
        return 'neutral'
    df['sent_label'] = df['vader_compound'].apply(label_sentiment)
    summary = df.groupby('year').agg(
        count=('text','size'),
        pos_count=('sent_label', lambda s: (s=='positive').sum()),
        neu_count=('sent_label', lambda s: (s=='neutral').sum()),
        neg_count=('sent_label', lambda s: (s=='negative').sum()),
        distrust_count=('is_distrust','sum'),
        trust_count=('is_trust','sum'),
        avg_vader_neg=('vader_neg','mean'),
        avg_vader_neu=('vader_neu','mean'),
        avg_vader_pos=('vader_pos','mean'),
        avg_vader_compound=('vader_compound','mean'),
        avg_tb_polarity=('tb_polarity','mean'),
        avg_tb_subjectivity=('tb_subjectivity','mean')
    ).reset_index()
    summary.to_csv(summary_path, index=False)
    print(f"Per-year sentiment summary saved to {summary_path}")
    return summary


def count_terms(df, terms, column='text_lower'):
    counts=[]
    for term in terms:
        total=df[column].str.count(re.escape(term)).sum()
        counts.append({'term':term,'count':int(total)})
    return pd.DataFrame(counts)


def main():
    print("Loading comments...")
    df = load_all_comments()
    if df.empty:
        print("No comments loaded; check your comment files.")
        return
    print(f"Loaded {len(df)} comments across years {df['year'].min()}–{df['year'].max()}.")
    print("Analyzing sentiment on full sample...")
    df_sent = analyze_sentiment(df)
    full_out = 'comments_sentiment_analysis.csv'
    df_sent.to_csv(full_out, index=False)
    print(f"✅ Full sentiment analysis saved to {full_out}.")
    print("Generating 1,000-comment subsamples per year for review...")
    for yr, grp in df_sent.groupby('year'):
        subsample = grp.sample(n=min(1000,len(grp)),random_state=42)
        sample_fn=f"sample_comments_{yr}.ndjson"
        with open(sample_fn,'w',encoding='utf-8') as sf:
            for rec in subsample.to_dict(orient='records'):
                sf.write(json.dumps(rec,ensure_ascii=False)+"\n")
        print(f"📋 Wrote {len(subsample)}-comment sample for {yr} to {sample_fn}")
    print("Generating summary...")
    generate_summary(df_sent)
    count_terms(df_sent, distrust_terms).to_csv('distrust_term_counts.csv', index=False)
    count_terms(df_sent, trust_terms).to_csv('trust_term_counts.csv', index=False)

if __name__ == '__main__':
    main()
