"""Build the persistent local Qdrant catalog index."""

from __future__ import annotations

import argparse

from clothing_search.catalog import index_catalog
from clothing_search.config import load_app_config
from clothing_search.embeddings.encoder import FashionEncoder
from clothing_search.search.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog")
    parser.add_argument("--config", default="configs/app.yaml")
    arguments = parser.parse_args()

    config = load_app_config(arguments.config)
    encoder = FashionEncoder(config.embedding.model_name)
    store = QdrantStore(
        collection_name=config.search.collection_name,
        vector_size=config.search.vector_size,
        path=config.search.path,
    )
    count = index_catalog(
        arguments.catalog,
        encoder=encoder,
        store=store,
        batch_size=config.embedding.batch_size,
    )
    print(f"Indexed {count} catalog items.")


if __name__ == "__main__":
    main()
