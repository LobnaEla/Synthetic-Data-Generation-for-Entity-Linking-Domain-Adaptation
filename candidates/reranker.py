import gc

import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-4B"

# Pointwise format expected by a stock (non fine-tuned) Qwen3-Reranker checkpoint:
# one query (context + mention + entity type, no candidate list) scored against
# one document (a single candidate title) per pair. The model is not shown the
# other candidates -- it only judges relevance of this one document to the query.
# "Pick the Q-ID or NONE" (listwise) does not apply with a single candidate per
# prompt, so the question is a yes/no relevance judgment instead.
QUERY_TEMPLATE = (
    "Identify the Wikidata entity for the mention marked with **double "
    "asterisks** in the context.\n\n"
    "Context:\n{context_before}**{mention}**{context_after}\n\n"
    "Entity type: {entity_type}\n\n"
    "Judge whether the candidate document correctly identifies this entity. "
    "Note that the answer can only be \"yes\" or \"no\"."
)


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 4,
        max_length: int = 1024,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            print(f"[Reranker] Loading {self.model_name} with CrossEncoder ...")

            # Qwen's model card recommends CrossEncoder for Qwen3-Reranker.
            # On P100, float16 is safer/lighter than bfloat16.
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                model_kwargs={
                    "torch_dtype": dtype,
                    "device_map": "auto",
                },
            )

            print(f"[Reranker] Loaded {self.model_name}")

    def unload(self):
        """Release reranker GPU memory before loading the final LLM."""
        if self._model is not None:
            del self._model
            self._model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Reranker] Unloaded and CUDA cache cleared")

    @staticmethod
    def _split_context(context: str, mention: str) -> tuple[str, str, str]:
        idx = context.find(mention)

        if idx >= 0:
            return (
                context[:idx],
                context[idx: idx + len(mention)],
                context[idx + len(mention):],
            )

        # Fallback: keep the prompt valid even if exact surface form is not found.
        return context + "\n\nMention: ", mention, ""

    def build_query(self, context: str, mention: str, entity_type: str) -> str:
        context_before, marked_mention, context_after = self._split_context(
            context,
            mention,
        )

        return QUERY_TEMPLATE.format(
            context_before=context_before,
            mention=marked_mention,
            context_after=context_after,
            entity_type=entity_type,
        )

    def rerank(
        self,
        context: str,
        mention: str,
        entity_type: str,
        candidates: list[str],
        k: int = 10,
    ) -> list[str]:
        """Rerank candidates for one mention using pointwise CrossEncoder scoring.

        Query side:
            context with the mention marked, plus entity type. No candidate
            list -- a stock Qwen3-Reranker checkpoint scores query/document
            relevance, it does not compare across candidates.

        Document side:
            one candidate title.

        The model returns one relevance score per (query, document) pair.
        Higher score = more relevant.
        """
        self._load()

        if not candidates:
            return []

        query = self.build_query(context, mention, entity_type)
        pairs = [(query, title) for title in candidates]

        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        return [title for title, _ in scored[:k]]

    def rerank_dataframe(self, df, k: int = 10, unload_after: bool = True):
        """Add candidates_reranked to a dataframe.

        This keeps the reranking loop inside candidates/reranker.py.
        eval_accuracy.py only calls this method before loading the final LLM.
        """
        print(f"[RERANK] Reranking to top-{k} candidates ...")
        reranked = []

        for rec in tqdm(df.to_dict("records"), desc="rerank"):
            reranked.append(
                self.rerank(
                    context=rec["context"],
                    mention=rec["mention"],
                    entity_type=rec["entity_type"],
                    candidates=rec.get("candidates", []),
                    k=k,
                )
            )

        df = df.copy()
        df["candidates_reranked"] = reranked

        if unload_after:
            self.unload()

        return df
