import csv
import gc
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-4B"
NONE_TOKEN = "NONE"

# Exact same prompt template as filter_zelda_rerank.py from teammate's rerank fine-tuning code.
PROMPT_TEMPLATE = (
    "Identify the Wikidata entity for the mention marked with **double "
    "asterisks** in the context.\n\n"
    "Context:\n{context_before}**{mention}**{context_after}\n\n"
    "Candidates:\n{candidates}\n\n"
    "Which candidate does the mention refer to? Answer with only the Q-ID "
    "(e.g. Q42) or \"NONE\"."
)


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        qid_titles_path: str = "data/wikidata_qid_titles.csv",
        batch_size: int = 4,
        max_length: int = 1024,
    ):
        self.model_name = model_name
        self.qid_titles_path = Path(qid_titles_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._title_to_qid: dict[str, str] | None = None

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

    def _load_title_to_qid(self) -> dict[str, str]:
        """Load title -> QID mapping if available.

        The teammate prompt displays candidates as: QID — title.
        Your current retrieval pipeline returns candidate titles, so this mapping
        reconstructs the QID part when data/wikidata_qid_titles.csv is available.
        """
        if self._title_to_qid is not None:
            return self._title_to_qid

        self._title_to_qid = {}

        if not self.qid_titles_path.exists():
            print(f"[Reranker] QID/title file not found: {self.qid_titles_path}")
            print("[Reranker] Candidates will be formatted with titles only.")
            return self._title_to_qid

        with self.qid_titles_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return self._title_to_qid

            title_col = "title" if "title" in reader.fieldnames else None

            qid_col = None
            for col in ("qid", "QID", "id", "entity_id", "wikidata_id"):
                if col in reader.fieldnames:
                    qid_col = col
                    break

            if not title_col or not qid_col:
                print(
                    f"[Reranker] Could not detect title/qid columns in "
                    f"{self.qid_titles_path}. Columns: {reader.fieldnames}"
                )
                print("[Reranker] Candidates will be formatted with titles only.")
                return self._title_to_qid

            for row in reader:
                title = (row.get(title_col) or "").strip()
                qid = (row.get(qid_col) or "").strip()

                if title and qid:
                    self._title_to_qid[title] = qid
                    self._title_to_qid[title.replace("_", " ")] = qid
                    self._title_to_qid[title.replace(" ", "_")] = qid

        print(f"[Reranker] Loaded {len(self._title_to_qid)} title→QID entries")
        return self._title_to_qid

    @staticmethod
    def _split_context(context: str, mention: str) -> tuple[str, str, str]:
        """Create the same context_before / mention / context_after fields as teammate."""
        idx = context.find(mention)

        if idx >= 0:
            return (
                context[:idx],
                context[idx: idx + len(mention)],
                context[idx + len(mention):],
            )

        # Fallback: keep the prompt valid even if exact surface form is not found.
        return context + "\n\nMention: ", mention, ""

    def _candidate_line(self, title: str) -> str:
        title_to_qid = self._load_title_to_qid()
        qid = title_to_qid.get(title) or title_to_qid.get(title.replace("_", " "))

        if qid:
            return f"{qid} — {title}"

        return title

    def _format_candidates(self, candidates: list[str]) -> str:
        lines = [f"  {self._candidate_line(c)}" for c in candidates]
        lines.append(f"  {NONE_TOKEN} — none of the candidates above")
        return "\n".join(lines)

    def build_prompt(self, context: str, mention: str, candidates: list[str]) -> str:
        context_before, marked_mention, context_after = self._split_context(
            context,
            mention,
        )

        return PROMPT_TEMPLATE.format(
            context_before=context_before,
            mention=marked_mention,
            context_after=context_after,
            candidates=self._format_candidates(candidates),
        )

    def rerank(
        self,
        context: str,
        mention: str,
        candidates: list[str],
        k: int = 10,
    ) -> list[str]:
        """Rerank candidates for one mention using CrossEncoder.

        Query side:
            teammate reranking prompt with all candidates.

        Document side:
            one candidate line, e.g. "Q30 — United States".

        The model returns one relevance score per pair.
        Higher score = more relevant.
        """
        self._load()

        if not candidates:
            return []

        prompt = self.build_prompt(context, mention, candidates)
        docs = [self._candidate_line(c) for c in candidates]
        pairs = [(prompt, doc) for doc in docs]

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
                    candidates=rec.get("candidates", []),
                    k=k,
                )
            )

        df = df.copy()
        df["candidates_reranked"] = reranked

        if unload_after:
            self.unload()

        return df