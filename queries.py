import xml.etree.ElementTree as ET
import json

def parse_queries(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    queries = []
    for i, top in enumerate(root.findall("top"), start=1):  # Start numbering from 1
        num_element = top.find("num")
        num = str(i)  # Update 'num' to match 1-225 numbering
        num_element.text = num  # Update 'num' element in the XML
        title = top.find("title").text.strip()

        # Print some details for debugging
        print(f"Num: {num}")
        print(f"Title: {title}")

        queries.append({"num": num, "title": title})



    return queries

# File path (Change if needed)
queries_file = "D:\Practicum\Mechanics of Search\Dataset\cranfield-trec-dataset-main\cran.qry.xml"

# Parse Queries
queries = parse_queries(queries_file)

# Save Queries to JSON
with open("parsed_queries.json", "w", encoding="utf-8") as json_file:
    json.dump(queries, json_file, indent=4, ensure_ascii=False)

print(f"Parsed {len(queries)} queries and saved to parsed_queries.json!")
