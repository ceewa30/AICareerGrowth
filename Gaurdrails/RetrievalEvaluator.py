from typing import List, Dict, Any, Tuple

class RetrievalEvaluator:
    def __init__(self, confidence_floor: float = 35.0, top_k_min: int = 1):
        """
        Configures quality control parameter floors for vector chunk retrieval.
        A default floor of 35.0% prevents unrelated contextual noise from bleeding into prompts.
        """
        self.confidence_floor = confidence_floor
        self.top_k_min = top_k_min

    def calculate_confidence(self, raw_distance: float) -> float:
        """
        Transforms standard L2 Squared vector distances into legible percentage values
        using your unified mathematical Euler constant decay scaling curve.
        """
        # Matches the core execution mathematics used in your tracker search loop
        confidence = (2.718281828459045 ** -raw_distance) * 100
        return max(0.0, min(100.0, round(confidence, 2)))

    def evaluate_retrieval_payload(self, raw_results: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Evaluates structural vector query payloads, isolating relevant data objects
        while evicting entries that do not meet the minimum confidence threshold floor.
        """
        print("🔍 [Retrieval Evaluation] Running vector quality control checks...")

        # 1. Grab raw query array items safely
        raw_docs = raw_results.get("documents") or []
        raw_metas = raw_results.get("metadatas") or []
        raw_dists = raw_results.get("distances") or []
        raw_ids = raw_results.get("ids") or []

        # 2. Defensively flatten structural 2D lists down to flat 1D arrays
        # This strips out ChromaDB's double nesting [[...]] entirely
        documents = raw_docs[0] if (raw_docs and isinstance(raw_docs[0], list)) else raw_docs
        metadatas = raw_metas[0] if (raw_metas and isinstance(raw_metas[0], list)) else raw_metas
        distances = raw_dists[0] if (raw_dists and isinstance(raw_dists[0], list)) else raw_dists
        ids = raw_ids[0] if (raw_ids and isinstance(raw_ids[0], list)) else raw_ids

        if not documents or len(documents) == 0:
            print("⚠️ Retrieval Warning: Zero matching elements discovered in this search execution branch.")
            return [], []

        clean_documents = []
        clean_metadatas = []

        # 3. Iterate through elements to compute quality scores
        for rank in range(len(documents)):
            doc_text = documents[rank]
            doc_id = ids[rank] if rank < len(ids) else f"unknown_chunk_{rank}"
            doc_meta = metadatas[rank] if rank < len(metadatas) else {}

            # Pull the distance parameter variant safely
            distance_item = distances[rank] if rank < len(distances) else 0.5

            # 🌟 TYPE-AGNOSTIC SAFE GUARD: If distance_item is still somehow wrapped in a list, peel it
            if isinstance(distance_item, list) and len(distance_item) > 0:
                raw_dist = float(distance_item[0])
            else:
                raw_dist = float(distance_item)

            # Execute your negative decay calculation safely without crashing
            score = self.calculate_confidence(raw_dist)

            # Apply the quality threshold boundary gate floor filter (e.g., 35.0%)
            if score >= self.confidence_floor:
                clean_documents.append(doc_text)

                enriched_meta = doc_meta.copy() if doc_meta else {}
                enriched_meta["eval_confidence_match"] = score
                clean_metadatas.append(enriched_meta)
            else:
                print(f"🚫 Quality Gate Eviction: Dropping document block ID '{doc_id}' due to low score: {score}%")

        print(f"📊 Quality Check Complete: Retained {len(clean_documents)} / {len(documents)} context blocks.")
        return clean_documents, clean_metadatas