'''import json
# Load the JSON file
with open("parsed_documents.json", "r", encoding="utf-8") as json_file:
    loaded_documents = json.load(json_file)

print(f"Loaded {len(loaded_documents)} documents from JSON!")
'''

import json
import re
from collections import defaultdict

def tokenize(text):
    """Tokenizes text: Converts to lowercase and removes punctuation."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    return text.split()  # Split into words

def build_inverted_index(documents):
    """Builds an inverted index from a list of documents."""
    inverted_index = defaultdict(set)

    for doc in documents:
        doc_id = doc["docno"]  # Get document number
        text = doc["text"]  # Get document text

        if text:  # Ensure text is not empty
            words = tokenize(text)  # Tokenize the text
            for word in words:
                inverted_index[word].add(doc_id)  # Add doc_id to the word's postings

    # Convert sets to sorted lists for easy searching
    for word in inverted_index:
        inverted_index[word] = sorted(inverted_index[word])

    return inverted_index

# Load parsed documents
with open("D:\Practicum\Mechanics of Search\parsed_documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# Build inverted index
inverted_index = build_inverted_index(documents)


# Print a small sample to check correctness
print("Sample Inverted Index:")
sample_words = list(inverted_index.keys())[:10]  # Get first 10 words
for word in sample_words:
    print(f"{word}: {inverted_index[word]}")

# Save inverted index to JSON
with open("inverted_index.json", "w", encoding="utf-8") as json_file:
    json.dump(inverted_index, json_file, indent=4, ensure_ascii=False)

print(f"Inverted index built and saved to inverted_index.json!")
