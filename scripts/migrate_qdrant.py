#!/usr/bin/env python3
"""One-shot migration: copy all points from local Qdrant file DB to Docker Qdrant server.

Usage:
    # 1. Make sure Docker Qdrant is running:
    #    wagent qdrant start
    # 2. Run migration:
    #    python scripts/migrate_qdrant.py
    # 3. Verify and optionally remove old data:
    #    wagent qdrant status

The script is idempotent — re-running it will upsert existing points
(matched by UUID), so duplicates won't be created.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from wagent.config import get_settings
from wagent.rag.store import COLLECTION_NAME, VECTOR_DIM

BATCH_SIZE = 100


def main() -> None:
    cfg = get_settings()
    local_path = cfg.qdrant_abs_path

    if not local_path.exists():
        print(f"Local Qdrant DB not found at {local_path}")
        sys.exit(1)

    if not cfg.qdrant_url:
        print("QDRANT_URL is not set in .env — nothing to migrate to.")
        sys.exit(1)

    print(f"Source:      {local_path}  (local file)")
    print(f"Destination: {cfg.qdrant_url}  (Docker server)")

    local_client = QdrantClient(path=str(local_path))
    remote_client = QdrantClient(url=cfg.qdrant_url, timeout=60)

    local_collections = [c.name for c in local_client.get_collections().collections]
    if COLLECTION_NAME not in local_collections:
        print(f"Collection '{COLLECTION_NAME}' not found in local DB. Nothing to migrate.")
        local_client.close()
        remote_client.close()
        return

    local_info = local_client.get_collection(COLLECTION_NAME)
    total_local = local_info.points_count
    print(f"Local collection has {total_local} points")

    remote_collections = [c.name for c in remote_client.get_collections().collections]
    if COLLECTION_NAME not in remote_collections:
        remote_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Created remote collection '{COLLECTION_NAME}'")

    migrated = 0
    offset = None

    while True:
        points, next_offset = local_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        remote_points = [
            PointStruct(
                id=str(p.id),
                vector=p.vector,
                payload=p.payload,
            )
            for p in points
        ]

        remote_client.upsert(collection_name=COLLECTION_NAME, points=remote_points)
        migrated += len(remote_points)
        print(f"  Migrated {migrated}/{total_local} points...", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\nMigration complete: {migrated} points transferred")

    remote_info = remote_client.get_collection(COLLECTION_NAME)
    print(f"Remote collection now has {remote_info.points_count} points")

    if remote_info.points_count >= total_local:
        print("Verification PASSED — remote has all local points")
    else:
        print(
            f"WARNING: remote has {remote_info.points_count} points "
            f"but local had {total_local}. Check for issues."
        )

    local_client.close()
    remote_client.close()


if __name__ == "__main__":
    main()
