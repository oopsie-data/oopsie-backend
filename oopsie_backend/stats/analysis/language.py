"""Steps, verbs and objects of the task instructions, and task clusters by meaning.

An instruction is parsed with spaCy, and only its first sentence counts, since later
sentences of long instructions describe the setup or the success condition. Every verb
that heads the sentence or is coordinated with another step verb starts a step ("pick up the cup and place it in the sink, then close
the drawer" has three steps). A step's verb is its lemma with particle ("pick up"), merged
through ``verb_synonyms``; a light verb such as "use" ("use the box to push the cube")
is replaced by the verb of its purpose clause (``light_verbs``); its objects are the lemmatized head nouns of its direct and
prepositional objects, without spatial nouns such as "left" (``object_stopwords``). Imperatives that the tagger reads as nouns are recovered when their
lemma is a known verb.

Task clusters merge instructions whose sentence embeddings reach a cosine similarity
threshold: the connected components of the graph that links every such pair.

spaCy and sentence-transformers come with the ``analysis`` extra (``uv sync --extra analysis``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

STEP_DEPS = {"ROOT", "conj"}
OBJECT_DEPS = {"dobj", "pobj", "obj", "attr", "oprd"}


@dataclass
class Step:
    verb: str | None
    objects: list[str]


def normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".!?;:")


class Parser:
    def __init__(self, cfg: dict):
        import spacy

        self.nlp = spacy.load(cfg.get("spacy_model", "en_core_web_sm"), disable=["ner"])
        self.synonyms = cfg.get("verb_synonyms") or {}
        self.known = {w for phrase in self.synonyms for w in phrase.split()[:1]}
        self.not_objects = set(cfg.get("object_stopwords") or [])
        self.light = set(cfg.get("light_verbs") or [])

    def _verb(self, token) -> str:
        lemma = token.lemma_.lower()
        if lemma in self.light:
            purpose = [c for c in token.children if c.dep_ in {"xcomp", "advcl"} and c.pos_ == "VERB"]
            if purpose:
                return self._verb(purpose[0])
        particles = [c.lemma_.lower() for c in token.children if c.dep_ == "prt"]
        nxt = token.nbor(1).lemma_.lower() if token.i + 1 < len(token.doc) else None
        candidates = [" ".join([lemma, *particles])] if particles else []
        candidates += [f"{lemma} {nxt}"] if nxt else []
        for phrase in [*candidates, lemma]:
            if phrase in self.synonyms:
                return self.synonyms[phrase]
        return " ".join([lemma, *particles])

    def _is_step(self, token) -> bool:
        if token.dep_ not in STEP_DEPS:
            return False
        if token.pos_ == "VERB":
            return True
        return token.i == token.sent.start and token.lemma_.lower() in self.known

    def _objects(self, verb) -> list[str]:
        out = []
        for t in verb.subtree:
            if (
                t.dep_ in OBJECT_DEPS
                and t.pos_ in {"NOUN", "PROPN"}
                and t.is_alpha
                and len(t.text) > 1
                and t.lemma_.lower() not in self.not_objects
                and self._step_of(t, verb)
            ):
                out.append(t.lemma_.lower())
        return sorted(set(out))

    def _step_of(self, token, verb) -> bool:
        """True when the nearest step verb above ``token`` is ``verb``."""
        t = token.head
        while t.i != verb.i and t.head.i != t.i:
            if self._is_step(t):
                return False
            t = t.head
        return t.i == verb.i

    def parse_all(self, texts: list[str], batch_size: int = 512) -> dict[str, list[Step]]:
        """Steps of every distinct normalized instruction."""
        unique = sorted({normalize(t) for t in texts if normalize(t)})
        out = {}
        for text, doc in zip(unique, self.nlp.pipe(unique, batch_size=batch_size)):
            sent = next(doc.sents)
            steps = [Step(self._verb(t), self._objects(t)) for t in sent if self._is_step(t)]
            out[text] = steps or [Step(None, sorted({t.lemma_.lower() for t in sent if t.pos_ == "NOUN"}))]
        return out


CONTENT_POS = {"NOUN", "VERB", "ADJ"}


def content_words(
    texts: list[str], cfg: dict, first_sentence: bool = False, pos: set[str] = CONTENT_POS, batch_size: int = 512
) -> Counter:
    """Lemmas of the words of part of speech ``pos`` (nouns, verbs and adjectives) of each text (of its first
    sentence with ``first_sentence``), counted once per text, without spaCy's stop words, words of two letters or
    fewer, and ``word_stopwords``."""
    import spacy

    nlp = spacy.load(cfg.get("spacy_model", "en_core_web_sm"), disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    stop = set(cfg.get("word_stopwords") or [])
    normalized = [normalize(t) for t in texts]
    unique = sorted({t for t in normalized if t})
    words = {}
    for text, doc in zip(unique, nlp.pipe(unique, batch_size=batch_size)):
        tokens = next(doc.sents) if first_sentence else doc
        words[text] = {
            t.lemma_.lower() for t in tokens
            if t.pos_ in pos and t.is_alpha and not t.is_stop and len(t.text) > 2
            and t.lemma_.lower() not in stop
        }
    return Counter(w for t in normalized if t for w in words[t])


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """Unit-length sentence embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False))


def clusters(embeddings: np.ndarray, threshold: float, block: int = 2048) -> np.ndarray:
    """Cluster id of every row: connected components of pairs with cosine similarity ≥ ``threshold``."""
    n = len(embeddings)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, block):
        sims = embeddings[start : start + block] @ embeddings.T
        rows, cols = np.nonzero(sims >= threshold)
        for i, j in zip(rows + start, cols):
            if j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])
