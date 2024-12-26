import os
import pickle
import pandas as pd
import numpy as np
import logging
import ray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from constants import UPLOAD_DIR


class SentenceBertModel:
    def __init__(
        self,
        source_data,
        model_name="all-MiniLM-L6-v2",
        embeddings_fname="record_linkage_embeddings.pkl",
    ):
        self.source_data = source_data
        self.model_name = model_name
        self.embeddings_path = f"{UPLOAD_DIR}/{embeddings_fname}"
        self.embeddings = None

        if os.path.isfile(self.embeddings_path):
            os.remove(self.embeddings_path)

    def vectorize(self):
        logging.info("Creating embeddings for record linkage...")

        self.model = SentenceTransformer(self.model_name)
        record_vectors = self.model.encode(self.source_data)
        with open(self.embeddings_path, "wb") as file:
            pickle.dump(record_vectors, file)
            logging.info("Done creating record embeddings")

        return np.array(record_vectors)


def preprocessing(src_df, src_cols):
    """
    Prepares the text data by concatenating specified columns into a single column.
    """
    src_df["record_text"] = (
        src_df[src_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )
    return src_df


@ray.remote
def get_similar_records(embeddings, records_trained, similarity_threshold):
    """
    Finds records that are similar to each other based on cosine similarity.
    """
    sim_records_list = []

    for record, rec_idx in zip(records_trained, range(len(records_trained))):
        sim_scores = cosine_similarity(
            embeddings[rec_idx, :].reshape(1, -1), embeddings
        )[0]
        similar_records = [
            {
                "Original_Record": record,
                "Similar_Record": records_trained[i],
                "Similarity_Score": score,
            }
            for i, score in enumerate(sim_scores)
            if i != rec_idx and score > similarity_threshold
        ]

        similar_records = sorted(
            similar_records, key=lambda x: x["Similarity_Score"], reverse=True
        )

        sim_records_list.extend(similar_records)

    return sim_records_list


def save_similarity_table(sim_records_list, preprocessed_df, output_path):
    """
    Merges similar records with the preprocessed dataframe and saves it as a parquet file.
    """
    similar_records_df = pd.DataFrame(sim_records_list)
    print(similar_records_df.head())
    final_df = preprocessed_df.merge(
        similar_records_df,
        left_on="record_text",
        right_on="Original_Record",
        how="left",
    ).drop(columns=["Original_Record"])

    final_df = final_df.drop_duplicates(subset=["fname_c1", "Similarity_Score"])
    final_df = final_df.sort_values(by="Similarity_Score", ascending=False).reset_index(
        drop=True
    )

    if os.path.isfile(output_path):
        os.remove(output_path)

    final_df.to_parquet(output_path)
    logging.info(f"Similarity table saved to {output_path}.")
    
