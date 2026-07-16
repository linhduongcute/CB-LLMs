from datasets import load_dataset as hf_load_dataset


UCI_DRUG = "Duyacquy/UCI_drug"
ECOMMERCE_TEXT = "Duyacquy/Ecommerce_text"

DATASET_PATH_NAMES = {
    "SetFit_sst2": "SetFit/sst2",
    "Duyacquy_UCI_drug": UCI_DRUG,
    "Duyacquy_Ecommerce_text": ECOMMERCE_TEXT,
}


def dataset_from_path_name(path_name):
    """Restore Hub dataset IDs that were made filesystem-safe with '/' -> '_'."""
    return DATASET_PATH_NAMES.get(path_name, path_name)


def load_dataset(dataset_name, *args, **kwargs):
    """Load a dataset and adapt its label column to the project's 0-based schema."""
    dataset = hf_load_dataset(dataset_name, *args, **kwargs)

    if dataset_name == UCI_DRUG:
        rating_to_label = {1: 0, 5: 1, 10: 2}

        def normalize_rating(example):
            rating = example["rating"]
            if rating not in rating_to_label:
                raise ValueError(
                    f"Unexpected rating {rating!r} in {UCI_DRUG}; expected 1, 5, or 10."
                )
            example["label"] = rating_to_label[rating]
            return example

        dataset = dataset.map(normalize_rating)

    if dataset_name == ECOMMERCE_TEXT:
        category_to_label = {
            "Household": 0,
            "Electronics": 1,
            "Clothing & Accessories": 2,
            "Books": 3,
        }

        def normalize_category(example):
            category = example["label"]
            if category not in category_to_label:
                raise ValueError(
                    f"Unexpected category {category!r} in {ECOMMERCE_TEXT}."
                )
            example["label"] = category_to_label[category]
            return example

        dataset = dataset.map(normalize_category)

    return dataset
