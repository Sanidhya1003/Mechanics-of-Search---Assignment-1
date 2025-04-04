import json
import re
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''# Download stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')'''

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tokenize(text):
    """Lowercase, remove punctuation, and split into words."""
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return [lemmatizer.lemmatize(word) for word in words if word not in stop_words]


def compute_bm25_index(documents, k1=1.5, b=0.75):
    """Compute BM25 index for a collection of documents."""
    N = len(documents)
    avg_doc_length = sum(len(tokenize(doc.get("text", ""))) for doc in documents) / N
    df = defaultdict(int)
    doc_lengths = {}
    tf_index = {}

    for doc in documents:
        doc_id = str(doc.get("docno", "UNKNOWN"))
        words = tokenize(doc.get("text", ""))
        doc_lengths[doc_id] = len(words)
        term_counts = defaultdict(int)
        for word in words:
            term_counts[word] += 1
        tf_index[doc_id] = term_counts
        for term in set(words):
            df[term] += 1

    idf_index = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in df}
    bm25_index = {}

    for doc_id, term_counts in tf_index.items():
        bm25_index[doc_id] = {}
        for term, tf in term_counts.items():
            idf = idf_index.get(term, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_lengths[doc_id] / avg_doc_length))
            bm25_index[doc_id][term] = idf * (numerator / denominator)

    return bm25_index, idf_index


# Load documents
with open("D:\\Practicum\\Mechanics of Search\\parsed_documents.json", "r") as f:
    documents = json.load(f)

bm25_index, idf_index = compute_bm25_index(documents)


def compute_query_bm25(query, idf_index, bm25_index, k1=1.5, b=0.75):
    """Compute BM25 scores for a given query across all documents."""
    query_terms = tokenize(query)
    scores = defaultdict(float)

    for doc_id, term_scores in bm25_index.items():
        for term in query_terms:
            if term in term_scores:
                scores[doc_id] += term_scores[term]

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Load queries
with open("D:\\Practicum\\Mechanics of Search\\parsed_queries.json", "r") as f:
    queries = json.load(f)

query_results = {}
for query_obj in queries:
    query_id = str(query_obj.get("num", "UNKNOWN"))
    query_text = query_obj.get("title", "")
    query_results[query_id] = compute_query_bm25(query_text, idf_index, bm25_index)[:100]

output_file = "bm25_results.txt"
with open(output_file, "w") as f:
    for query_id, results in query_results.items():
        for rank, (doc_id, score) in enumerate(results, start=1):
            f.write(f"{query_id} 0 {doc_id} {rank} {score:.4f} BM25_Model\n")

# Print sample results
for query_id, results in list(query_results.items())[:3]:
    print(f"Query ID: {query_id}")
    print("Ranked Documents:")
    for rank, (doc_id, score) in enumerate(results[:100], start=1):
        print(f"  Rank {rank}: Doc {doc_id}, Score: {score:.4f}")
    print("-" * 50)

print("BM25 Query Ranking Computation Complete!")
