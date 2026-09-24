from sql_utils import SQLConnection
from styling import Options, Readability

db_full = SQLConnection("all_data_aug_2024")
db_publication = SQLConnection("publication")
readability_obj = Readability()
options_obj = Options()


def database_for(dataset_key: str) -> SQLConnection:
    if dataset_key == "publication":
        return db_publication
    if dataset_key == "full":
        return db_full
    raise ValueError(f"Invalid dataset key: {dataset_key}")
