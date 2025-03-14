import xml.etree.ElementTree as ET


def parse_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        xml_content = file.read()

    # Add a root element dynamically if missing
    xml_content = f"<root>{xml_content}</root>"

    # Parse the modified XML content
    root = ET.fromstring(xml_content)
    documents = []

    for doc in root.findall('doc'):
        docno = doc.find('docno').text if doc.find('docno') is not None else "Unknown"
        title = doc.find('title').text if doc.find('title') is not None else "No Title"
        author = doc.find('author').text if doc.find('author') is not None else "Unknown Author"
        bib = doc.find('bib').text if doc.find('bib') is not None else "No Bib Info"
        text = doc.find('text').text if doc.find('text') is not None else ""

        # Handle None before slicing
        text_preview = text[:100] + "..." if text else "No Content"

        # Print some details for debugging
        print(f"Document ID: {docno}")
        print(f"Title: {title}")
        print(f"Author: {author}")
        print(f"Bib: {bib}")
        print(f"Text: {text_preview}\n")

        documents.append({'docno': docno, 'title': title, 'author': author, 'bib': bib, 'text': text})

    return documents


# File path
documents_file = 'D:\Practicum\Mechanics of Search\Dataset\cranfield-trec-dataset-main\cran.all.1400.xml'  # Update with actual path

# Parse
documents = parse_documents(documents_file)

print(f"\nTotal Documents Parsed: {len(documents)}")
import json

# Save parsed documents to JSON
with open("parsed_documents.json", "w", encoding="utf-8") as json_file:
    json.dump(documents, json_file, indent=4, ensure_ascii=False)

print("Documents saved to parsed_documents.json")
