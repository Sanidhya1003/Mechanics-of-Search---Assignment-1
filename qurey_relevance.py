import json

def parse_qrels(file_path):
    qrels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                topic, iteration, docno, relevance = parts

                # Print some details for debugging
                print(f"Topic: {topic}")
                print(f"Doc No: {docno}")
                print(f"Relevance: {relevance}")


                qrels.append({
                    "topic": int(topic),
                    "docno": int(docno),
                    "relevance": int(relevance)
                })

    return qrels

# File path (Change if needed)
qrels_file = "D:\Practicum\Mechanics of Search\Dataset\cranfield-trec-dataset-main\cranqrel.trec.txt"

# Parse Qrels
qrels = parse_qrels(qrels_file)

# Save Qrels to JSON
with open("parsed_qrels.json", "w", encoding="utf-8") as json_file:
    json.dump(qrels, json_file, indent=4, ensure_ascii=False)

print(f"Parsed {len(qrels)} query relevance judgments and saved to parsed_qrels.json!")
