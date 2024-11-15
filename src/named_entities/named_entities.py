import xml.etree.ElementTree as ET
import gzip
import os
from collections import defaultdict
from tqdm import tqdm_gui as tqdm
import multiprocessing
import pandas as pd


CORENLP_PATH = "MovieSummaries/morethan100MB/corenlp_plot_summaries"
CSV_OUTPUT_PATH = "named_entities/named_entities.csv"


def main():

    gz_files = inputs()
    # extracting named entities from all the gz files
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(extract_named_entities, gz_files),
                desc="extracting all named entities from all gz files",
                total=len(gz_files)
            )
        )
    # saving the results
    pd.DataFrame(results).to_csv(CSV_OUTPUT_PATH, index=False)


def inputs():

    gz_files = []
    # getting all the .gz files' paths
    for root, _, files in os.walk(CORENLP_PATH):
        for file in files:
            if file.endswith('.gz'):
                gz_files.append(os.path.join(root, file))
    return gz_files


def extract_named_entities(xml_file):
    """extract_named_entities

    Args:
        xml_file: xml_file where to extract named entities

    Returns:
        dict: dictionary with the named entities
    """
    # extracting named entities from the xml file
    with gzip.open(xml_file, 'rt', encoding='utf-8') as f:
        tree = ET.parse(f)
        root = tree.getroot()

        entities = []
        for sentence in root.iter('sentence'):
            for token in sentence.iter('token'):
                ner = token.find('NER').text
                word = token.find('word').text
                if ner != 'O':  # Skip non-entities
                    entities.append((word, ner))
        
    # creating a dictionary with the named entities
    entity_dict = defaultdict(list)

    # taking Wikipedia ID from the filename
    id, _ = os.path.splitext(os.path.splitext(os.path.basename(xml_file))[0])
    entity_dict["ID"] = id

    for name, entity_type in entities:
        entity_dict[entity_type].append(name)

    return dict(entity_dict)


if __name__ == "__main__":
    main()