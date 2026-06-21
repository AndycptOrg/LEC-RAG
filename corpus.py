def get_docs():
    import csv

    with open('synthetic-it-related-knowledge-items\synthetic_knowledge_items.csv', 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        first_column = [row['ki_text']for row in reader]
    return first_column
