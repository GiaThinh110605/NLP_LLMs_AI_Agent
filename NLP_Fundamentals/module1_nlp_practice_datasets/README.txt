MODULE 1 - NLP FUNDAMENTALS: 3 PRACTICE DATASETS
==================================================

Files
-----
1. exercise1_raw_text.csv
   Columns: id, text
   Goal: practice lowercase, whitespace cleanup, punctuation handling, tokenization,
   stopword removal, stemming/lemmatization, and compare before/after text.

2. exercise2_sms_spam.csv
   Columns: id, label, message
   Labels: spam / ham
   Goal: train/test split, TF-IDF, MultinomialNB, Accuracy, Precision, Recall, F1,
   then predict 5 custom messages.
   NOTE: this is a small educational sample (40 rows), not a production dataset.
   For a larger real-world dataset, use the UCI SMS Spam Collection:
   https://archive.ics.uci.edu/dataset/228/sms+spam+collection

3. exercise3_sentiment.csv
   Columns: id, label, text
   Labels: positive / negative
   Goal: compare CountVectorizer vs TfidfVectorizer, unigram vs bigram,
   MultinomialNB vs LogisticRegression, and inspect mistakes.
   NOTE: small educational sample (40 rows); results will be unstable due to size.

Suggested environment
---------------------
Google Colab: https://colab.research.google.com/
Install if needed:
!pip install nltk scikit-learn pandas

Important practice rules
------------------------
- Split train/test BEFORE fitting the vectorizer to avoid data leakage.
- Fit the vectorizer only on training text; use transform on test text.
- Use random_state=42 for reproducibility.
- Small datasets are for learning the pipeline, not for claiming strong model quality.
