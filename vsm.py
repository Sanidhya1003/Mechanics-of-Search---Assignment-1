import json
import re
import math
from collections import defaultdict
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords and wordnet if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tokenize(text):
    """Lowercase, remove punctuation, and split into words."""
    if not isinstance(text, str):  # Ensure text is a string
        print("Warning: Found non-string text, replacing with empty string.")
        text = ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    words = text.split() # Split into words

    # Remove stopwords and apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words #Return processed list of words

    #return text.split()




def compute_tf(documents):
    """Compute Term Frequency (TF) for each document."""
    tf_index = {}

    for doc in documents:
        doc_id = str(doc.get("docno", "UNKNOWN"))  # Ensure ID is always a string
        text = doc.get("text", "")  # Ensure text is a string

        if text is None:  # Explicitly handle NoneType case
            print(f"Warning: Document ID {doc_id} has None as text! Replacing with empty string.")
            text = ""

        text = str(text)  # Force text to be a string
        if not text.strip():  # Skip empty documents
            print(f"Warning: Document ID {doc_id} has no valid text!")
            continue

        words = tokenize(text)
        total_terms = len(words)

        if total_terms == 0:  # Skip documents with no words after tokenization
            print(f"Warning: Document ID {doc_id} has no valid words after tokenization!")
            continue

        term_counts = defaultdict(int)
        for word in words:
            term_counts[word] += 1

        # Normalize term counts by total terms in the document
        tf_index[doc_id] = {term: 1 + math.log(count) if count > 0 else 0 for term, count in term_counts.items()}

    return tf_index


# Load parsed document JSON file
with open("D:\Practicum\Mechanics of Search\parsed_documents.json", "r") as f:
    documents = json.load(f)

'''# Debugging: Check first few documents to verify structure
print("Sample documents (first 3) for verification:")
for i in range(min(3, len(documents))):  # Prevents index error if less than 3 docs
    print(documents[i])'''

# Compute TF index
tf_index = compute_tf(documents)

# Print sample TF values for the first few documents
sample_docs = list(tf_index.keys())[:5]  # Get first 5 document IDs
for doc_id in sample_docs:
    print(f"Document ID: {doc_id}")
    print(f"Sample TF values: {dict(list(tf_index[doc_id].items())[:10])}")  # Print first 10 terms
    print("-" * 50)


'''# Save TF index to JSON
with open("tf_index.json", "w") as f:
    json.dump(tf_index, f, indent=4)

print("TF index computed and saved successfully!")'''


# Step 3: Compute Inverse Document Frequency (IDF)
def compute_idf(documents):
    """Compute Inverse Document Frequency (IDF) for each term in the corpus."""
    N = len(documents)  # Total number of documents
    df = defaultdict(int)  # Document frequency for each term

    # Count the number of documents each term appears in
    for doc in documents:
        text = doc.get("text", "")  # Ensure text is a string
        if not isinstance(text, str):
            print(
                f"Warning: Document ID {doc.get('docno', 'UNKNOWN')} has non-string text, replacing with empty string.")
            text = ""

        terms = set(tokenize(text))  # Use set to avoid duplicate terms
        for term in terms:
            df[term] += 1

    # Calculate IDF for each term
    idf = {}
    for term, frequency in df.items():
        idf[term] = math.log((N + 1) / (frequency + 1)) + 1   # IDF formula

    return idf

idf_index = compute_idf(documents)

# Print sample IDF values for verification
sample_terms = list(idf_index.keys())[:5]  # Print first 10 terms as a sample
for term in sample_terms:
    print(f"Term: {term}, IDF: {idf_index[term]}")

print("IDF computation complete! Now you can verify the output.")

def compute_tf_idf(tf_index, idf_index):
    """Compute TF-IDF scores for each term in each document."""
    tf_idf_index = {}

    for doc_id, term_frequencies in tf_index.items():
        tf_idf_index[doc_id] = {term: tf * idf_index.get(term, 0) for term, tf in term_frequencies.items()}

    return tf_idf_index

# Compute TF-IDF index
tf_idf_index = compute_tf_idf(tf_index, idf_index)

# Print sample TF-IDF values for verification
sample_docs = list(tf_idf_index.keys())[:5]  # Get first 5 document IDs
for doc_id in sample_docs:
    print(f"Document ID: {doc_id}")
    print(f"Sample TF-IDF values: {dict(list(tf_idf_index[doc_id].items())[:10])}")  # Print first 10 terms
    print("-" * 50)


def compute_query_tf_idf(query, idf_index):
    """Compute TF-IDF for a given query."""
    query_tf_idf = {}

    words = tokenize(query)
    total_terms = len(words)

    if total_terms == 0:
        return query_tf_idf  # Return empty if query is empty

    term_counts = defaultdict(int)
    for word in words:
        term_counts[word] += 1

    # Compute TF-IDF for the query
    query_tf_idf = {term: (count / total_terms) * idf_index.get(term, 0) for term, count in term_counts.items()}

    return query_tf_idf



import numpy as np

def cosine_similarity(query_vector, doc_vector):
    """Compute cosine similarity between query and document vectors."""
    query_terms = set(query_vector.keys())
    doc_terms = set(doc_vector.keys())

    # Make sure both query and doc vectors have the same terms for comparison
    all_terms = query_terms.union(doc_terms)

    # Compute dot product and magnitudes using all terms
    dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in all_terms)
    query_norm = np.sqrt(sum(value ** 2 for value in query_vector.values()))
    doc_norm = np.sqrt(sum(value ** 2 for value in doc_vector.values()))

    '''common_terms = query_terms.intersection(doc_terms)

    # Compute dot product
    dot_product = sum(query_vector[term] * doc_vector[term] for term in common_terms)

    # Compute magnitudes
    query_norm = np.sqrt(sum(value ** 2 for value in query_vector.values()))
    doc_norm = np.sqrt(sum(value ** 2 for value in doc_vector.values()))'''

    # Avoid division by zero
    if query_norm == 0 or doc_norm == 0:
        return 0.0

    return dot_product / (query_norm * doc_norm)



# Load queries
with open("D:\Practicum\Mechanics of Search\parsed_queries.json", "r") as f:
    queries = json.load(f)

query_results = {}

for query_obj in queries:
    query_id = str(query_obj.get("num", "UNKNOWN"))  # Ensure query ID is a string
    query_text = query_obj.get("title", "")

    query_vector = compute_query_tf_idf(query_text, idf_index)  # Compute TF-IDF for query

    similarities = {}
    for doc_id, doc_vector in tf_idf_index.items():
        similarities[doc_id] = cosine_similarity(query_vector, doc_vector)

    # Sort documents by similarity score (higher is better)
    ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    query_results[query_id] = ranked_docs[:100]  # Top 10 results for each query

# Print sample results for verification
for query_id, results in list(query_results.items())[:3]:  # First 3 queries
    print(f"Query ID: {query_id}")
    for doc_id, score in results:
        print(f"  Doc {doc_id}: {score:.4f}")
    print("-" * 50)

print("Cosine similarity computation complete!")


output_file = "vsm_results.txt"  # Name of the output file for trec_eval

with open(output_file, "w") as f:
    for query_id, results in query_results.items():
        for rank, (doc_id, similarity) in enumerate(results[:100], start=1):  # Top 100 docs
            f.write(f"{query_id} 0 {doc_id} {rank} {similarity:.4f} VSM_Model\n")  # Formatting line
