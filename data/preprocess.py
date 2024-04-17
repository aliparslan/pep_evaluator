import csv
from GEOparse import GEOparse
from Bio import Entrez
from tqdm import tqdm

csv_file_path = "data.csv"
new_csv_file_path = "NEW_processed_data.csv"
gse_column_name = "gse"
pmid_column_name = "pmid"
Entrez.email = "aliparslan@outlook.com"


def process_geo_dataset(geo_accession: str):
    try:
        print(f"\033[1;32m\nProcessing {geo_accession}... \033[0m")
        geo_dataset = GEOparse.get_GEO(geo_accession, destdir=None)
        metadata = geo_dataset.metadata

        metadata_dict = {
            "title": metadata["title"][0],
            "summary": metadata["summary"][0],
            "overall_design": metadata.get("overall_design", [""])[0],
            "type": metadata["type"][0],
            "contributor": ", ".join(metadata.get("contributor", []))
        }

    except Exception as e:
        print(f"Error processing dataset: {geo_dataset}: {str(e)}")
        return {}

    return metadata_dict


def get_mesh_terms(pmid):
    mesh_terms = set()

    with Entrez.efetch(db="pubmed", id=pmid, retmode="xml") as handle:
        record = Entrez.read(handle)

    mesh_heading_list = record["PubmedArticle"][0]["MedlineCitation"]["MeshHeadingList"]

    for mesh_heading in mesh_heading_list:
        if "QualifierName" in mesh_heading:
            for qualifer_name in mesh_heading["QualifierName"]:
                mesh_terms.add(str(qualifer_name))

        if "DescriptorName" in mesh_heading:
            mesh_terms.add(str(mesh_heading["DescriptorName"]))

    return ", ".join(mesh_terms)


with open(csv_file_path, "r") as file:
    csv_reader = csv.DictReader(file)
    rows = list(csv_reader)

processed_rows = []
for row in tqdm(rows, desc="Processing datasets", unit="row"):
    gse_number = row[gse_column_name]
    pmid = row[pmid_column_name]
    metadata_dict = process_geo_dataset(gse_number)
    mesh_terms = get_mesh_terms(pmid)
    processed_row = {
        "topic": row["topic"],
        "gse": gse_number,
        "pmid": pmid,
        "title": metadata_dict.get("title", ""),
        "summary": metadata_dict.get("summary", ""),
        "overall_design": metadata_dict.get("overall_design", ""),
        "type": metadata_dict.get("type", ""),
        "contributor": metadata_dict.get("contributor", ""),
        "mesh_terms": mesh_terms
    }
    processed_rows.append(processed_row)

field_names = ["topic", "gse", "pmid", "title", "summary", "overall_design", "type", "contributor", "mesh_terms"]
with open(new_csv_file_path, "w", newline="") as file:
    csv_writer = csv.DictWriter(file, fieldnames=field_names)
    csv_writer.writeheader()
    csv_writer.writerows(processed_rows)