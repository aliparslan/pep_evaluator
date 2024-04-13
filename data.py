import csv
import json
import time
from GEOparse import GEOparse
from Bio import Entrez

csv_file_path = "dataset.csv"
gse_column_name = "GSE"
pmid_column_name = "PMID"
Entrez.email = "aliparslan@outlook.com"


def process_geo_dataset(geo_accession: str):
    try:
        print(f"\033[1;32mProcessing {geo_accession}... \033[0m")
        geo_dataset = GEOparse.get_GEO(geo_accession, destdir=None)
        metadata = geo_dataset.metadata

        with open(f"{geo_accession}.txt", "w") as file:
            file.write(f"Title: {metadata['title'][0]}\n")
            file.write(f"Summary: {metadata['summary'][0]}\n")
            file.write(f"Overall design: {metadata['overall_design'][0]}\n")
            file.write(f"Type: {metadata['type'][0]}\n")
            file.write(f"Contributor(s): {','.join(metadata['contributor'])}")

            # file.write(json.dumps(metadata))
    except Exception as e:
        print(f"Error processing dataset: {geo_dataset}: {str(e)}")


def get_mesh_terms(pmid):
    with Entrez.efetch(db="pubmed", id=pmid, retmode="xml") as handle:
        record = Entrez.read(handle)

    mesh_terms = set()

    mesh_heading_list = record["PubmedArticle"][0]["MedlineCitation"]["MeshHeadingList"]

    for mesh_heading in mesh_heading_list:
        if "QualifierName" in mesh_heading:
            for qualifer_name in mesh_heading["QualifierName"]:
                mesh_terms.add(str(qualifer_name))

        if "DescriptorName" in mesh_heading:
            mesh_terms.add(str(mesh_heading["DescriptorName"]))

    return list(mesh_terms)


start = time.time()
with open(csv_file_path, "r") as file:
    csv_reader = csv.DictReader(file)
    count = 0
    for row in csv_reader:
        if count >= 3:
            break
        gse_number = row[gse_column_name]
        pmid = row[pmid_column_name]

        process_geo_dataset(gse_number)
        mesh_terms = get_mesh_terms(pmid)
        print(f"Mesh terms for {pmid}: {', '.join(mesh_terms)}")

        count += 1
end = time.time()

print(f"Time taken: {(end - start):.2f} seconds")
