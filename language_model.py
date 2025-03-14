import json
import re
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''# Download stopwords and wordnet if not already downloaded
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


def compute_language_model(documents):
    """Compute term probabilities for each document and the entire collection."""
    term_frequencies = defaultdict(int)
    doc_term_probs = {}

    total_terms = 0
    for doc in documents:
        doc_id = str(doc.get("docno", "UNKNOWN"))
        words = tokenize(doc.get("text", ""))
        total_terms += len(words)

        doc_term_counts = defaultdict(int)
        for word in words:
            doc_term_counts[word] += 1
            term_frequencies[word] += 1

        doc_length = sum(doc_term_counts.values())
        doc_term_probs[doc_id] = {word: count / doc_length for word, count in doc_term_counts.items()}

    # Compute collection-wide probabilities
    collection_probs = {word: count / total_terms for word, count in term_frequencies.items()}

    return doc_term_probs, collection_probs


def compute_query_lm_jm(query, documents, doc_term_probs, collection_probs, lambda_param=0.1):
    """Compute query likelihood scores using Jelinek-Mercer smoothing."""
    query_terms = tokenize(query)
    scores = defaultdict(float)

    for doc_id, term_probs in doc_term_probs.items():
        score = 0
        for term in query_terms:
            # Interpolated probability
            p_term_given_doc = term_probs.get(term, 0)
            p_term_given_collection = collection_probs.get(term, 0)
            smoothed_prob = (1 - lambda_param) * p_term_given_doc + lambda_param * p_term_given_collection

            if smoothed_prob > 0:
                score += math.log(smoothed_prob)

        scores[doc_id] = score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Load documents
with open("D:\\Practicum\\Mechanics of Search\\parsed_documents.json", "r") as f:
    documents = json.load(f)

# Load queries
with open("D:\\Practicum\\Mechanics of Search\\parsed_queries.json", "r") as f:
    queries = json.load(f)

doc_term_probs, collection_probs = compute_language_model(documents)

query_results_lm = {}
for query_obj in queries:
    query_id = str(query_obj.get("num", "UNKNOWN"))
    query_text = query_obj.get("title", "")
    query_results_lm[query_id] = compute_query_lm_jm(query_text, documents, doc_term_probs, collection_probs)[:100]

output_file = "unigram_results.txt"
with open(output_file, "w") as f:
    for query_id, results in query_results_lm.items():
        for rank, (doc_id, score) in enumerate(results, start=1):
            f.write(f"{query_id} 0 {doc_id} {rank} {score:.4f} Unigram_Model\n")

print("Language Model Query Ranking Computation Complete!")

print("Language Model Query Ranking Computation Complete!\n")

# Display results for the first few queries (e.g., first 3 queries)
num_queries_to_display = 3
for i, (query_id, results) in enumerate(query_results_lm.items()):
    if i >= num_queries_to_display:
        break
    print(f"Query ID: {query_id}")
    for rank, (doc_id, score) in enumerate(results[:100], start=1):  # Show top 10 documents per query
        print(f"  Rank {rank}: Doc {doc_id} | Score: {score:.4f}")
    print("\n" + "-" * 50 + "\n")


