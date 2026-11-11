"""
Module 2: Earnings Call Q&A Evasiveness Scoring Engine
------------------------------------------------------
Computes a multi-dimensional evasiveness and responsiveness profile for
each (question, answer) pair:

1. Lexical Deflection Score:
   - Explicit refusal/deflection phrases ("we do not break that out", "too early to tell", etc.)
   - Loughran-McDonald uncertainty modals and hedging words ("might", "could", "perhaps").
2. Numeric Specificity Score:
   - Presence of concrete guidance metrics ($, %, basis points, bps, revenue/margin figures).
   - Direct answers typically provide quantitative numbers; evasions are qualitative rhetoric.
3. Elaboration & Verbosity Ratio:
   - Ratio of answer word count to question word count (identifies filibustering vs. abrupt brush-offs).
4. Semantic Alignment / Entailment:
   - Joint semantic relevance between question and answer. Supports Cross-Encoder / DeBERTa-v3
     when PyTorch is available, with a fast zero-dependency TF-IDF cosine baseline.
5. Composite Evasiveness Factor:
   - Calibrated normalized score E_pair in [0, 1] (0.0 = direct/transparent, 1.0 = highly evasive).
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Lexical Deflection & Non-Answer Patterns
# -----------------------------------------------------------------------------
DEFLECTION_PATTERNS = [
    # Explicit refusal to provide data / guidance
    re.compile(
        r"(?:do\s+not|don't|not\s+going\s+to|won't|prefer\s+not\s+to|can't|cannot)\s+"
        r"(?:break\s+(?:it\s+)?out|disclose|give|provide|comment|guide|speculate|quantify|share)",
        re.IGNORECASE
    ),
    # Premature / uncertainty dismissals
    re.compile(
        r"(?:too\s+early|premature|hard\s+to\s+say|difficult\s+to\s+predict|time\s+will\s+tell|"
        r"can't\s+predict|not\s+in\s+a\s+position\s+to|wait\s+and\s+see)",
        re.IGNORECASE
    ),
    # Deflecting back to prepared remarks
    re.compile(
        r"(?:as\s+(?:we|I)\s+(?:said|mentioned|noted|highlighted)\s+(?:earlier|before|in\s+(?:the|my)\s+prepared))",
        re.IGNORECASE
    ),
    # Offlining the inquiry
    re.compile(
        r"(?:take\s+(?:that|this|it)\s+offline|follow\s+up\s+offline|happy\s+to\s+follow\s+up\s+after)",
        re.IGNORECASE
    ),
    # Proprietary / competitive refusal
    re.compile(
        r"(?:competitive|proprietary)\s+(?:reasons?|information|dynamics|sensitivities)",
        re.IGNORECASE
    )
]

# -----------------------------------------------------------------------------
# 2. Loughran-McDonald Uncertainty & Weak Modal Lexicon (Selected Standard)
# -----------------------------------------------------------------------------
LM_UNCERTAINTY_WORDS = {
    "almost", "alter", "ambiguity", "ambiguous", "anticipate", "approximate",
    "approximately", "arbitrary", "assume", "assumes", "assumption", "believe",
    "believes", "cautious", "clarify", "conceivable", "conceivably", "conditional",
    "contingency", "could", "depend", "dependent", "depending", "depends",
    "doubt", "doubtful", "exposure", "fluctuate", "fluctuating", "fluctuation",
    "hidden", "indefinite", "indefinitely", "instability", "intangible", "likelihood",
    "may", "maybe", "might", "nearly", "ordinarily", "pending", "perhaps",
    "possibility", "possible", "possibly", "potential", "potentially", "precaution",
    "predict", "predictable", "predictability", "prediction", "preliminary", "probable",
    "probably", "random", "reassess", "reconsider", "risk", "risky", "rough",
    "roughly", "rumor", "seldom", "somewhat", "speculate", "speculation",
    "speculative", "sudden", "susceptible", "tentative", "tentatively", "turbulent",
    "uncertain", "uncertainly", "uncertainty", "unclear", "unconfirmed", "undecided",
    "undefined", "undetermined", "unexpected", "unforeseen", "unknown", "unobservable",
    "unplanned", "unpredictable", "unproved", "unquantifiable", "unresolved",
    "unstable", "untested", "unusual", "vague", "variability", "variable", "vary"
}

# Regex for financial figures ($, %, basis points, bps, millions, billions)
NUMERIC_FINANCIAL_PATTERN = re.compile(
    r"(?:\$[0-9]+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?%|[0-9]+(?:\.[0-9]+)?\s*(?:basis\s+points|bps|million|billion|cents?))",
    re.IGNORECASE
)


class EvasivenessScorer:
    """
    Multi-signal scoring engine for measuring executive evasiveness in earnings call Q&A.
    """

    def __init__(self, use_neural_cross_encoder: bool = False, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.use_neural = use_neural_cross_encoder
        self.cross_encoder = None

        if self.use_neural:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[*] Initializing Cross-Encoder ({model_name})...")
                self.cross_encoder = CrossEncoder(model_name)
                print("[+] Cross-Encoder loaded successfully.")
            except ImportError:
                print("[!] sentence_transformers not installed. Falling back to native semantic baseline.")
                self.use_neural = False

    @staticmethod
    def _clean_tokens(text: str) -> List[str]:
        words = re.findall(r"\b[a-z]{2,}\b", text.lower())
        return words

    def compute_lexical_signals(self, question: str, answer: str) -> Dict[str, float]:
        """Computes lexical deflection, uncertainty density, and numeric specificity."""
        a_tokens = self._clean_tokens(answer)
        total_a_words = max(1, len(a_tokens))

        # 1. Explicit Deflection Phrase Matches
        deflection_matches = sum(len(pat.findall(answer)) for pat in DEFLECTION_PATTERNS)

        # 2. Loughran-McDonald Uncertainty Density
        unc_matches = sum(1 for w in a_tokens if w in LM_UNCERTAINTY_WORDS)
        uncertainty_density = unc_matches / total_a_words

        # 3. Numeric Specificity (Direct financial quantification)
        num_matches = len(NUMERIC_FINANCIAL_PATTERN.findall(answer))
        numeric_density = num_matches / total_a_words

        # 4. Elaboration / Length Ratio
        q_tokens = self._clean_tokens(question)
        total_q_words = max(1, len(q_tokens))
        length_ratio = total_a_words / total_q_words

        return {
            "deflection_count": deflection_matches,
            "uncertainty_density": uncertainty_density,
            "numeric_count": num_matches,
            "numeric_density": numeric_density,
            "length_ratio": length_ratio,
            "a_words": total_a_words,
            "q_words": total_q_words
        }

    def compute_semantic_relevance(self, question: str, answer: str) -> float:
        """
        Computes semantic alignment score between Question and Answer.
        Returns a float in [0.0, 1.0], where 1.0 = highly relevant, 0.0 = completely disjoint.
        """
        if self.use_neural and self.cross_encoder is not None:
            # Neural Cross-Encoder prediction
            scores = self.cross_encoder.predict([(question, answer)])
            # Normalize to [0, 1] range depending on model output (sigmoid or softmax)
            raw = float(scores[0]) if hasattr(scores, "__len__") else float(scores)
            return 1.0 / (1.0 + math.exp(-raw))

        # Native Lexical-Semantic Jaccard / Overlap Fallback
        q_set = set(self._clean_tokens(question))
        a_set = set(self._clean_tokens(answer))
        if not q_set or not a_set:
            return 0.5

        # Exclude ultra-common stopwords from overlap calculation
        stopwords = {
            "the", "and", "to", "of", "in", "for", "on", "that", "is", "we",
            "you", "it", "with", "as", "at", "are", "this", "be", "from", "or",
            "have", "an", "was", "but", "not", "by", "what", "about", "can", "if"
        }
        content_q = q_set - stopwords
        content_a = a_set - stopwords

        if not content_q or not content_a:
            return 0.5

        intersection = len(content_q & content_a)
        # Weighted overlap relative to question content terms
        relevance = min(1.0, (intersection / len(content_q)) * 1.5)
        return float(relevance)

    def score_pair(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Scores a single (question, answer) pair across all evasion dimensions.
        Returns a dictionary of raw metrics and a unified composite evasiveness score.
        """
        lex = self.compute_lexical_signals(question, answer)
        relevance = self.compute_semantic_relevance(question, answer)

        # -------------------------------------------------------------
        # Composite Evasiveness Calibration Formula
        # -------------------------------------------------------------
        # 1. Base Evasion from low semantic relevance:
        #    If relevance is 1.0 -> evasion_base = 0.0; if relevance is 0.0 -> evasion_base = 0.50
        evasion_base = (1.0 - relevance) * 0.45

        # 2. Deflection penalty: each explicit refusal adds +0.25 (capped at 0.50)
        deflection_penalty = min(0.50, lex["deflection_count"] * 0.25)

        # 3. Uncertainty / Hedging penalty:
        #    Average LM uncertainty in business text is ~2-4%. Above 7% indicates heavy hedging.
        hedging_penalty = min(0.25, max(0.0, (lex["uncertainty_density"] - 0.03) * 3.5))

        # 4. Numeric specificity bonus (mitigates evasion):
        #    If executive provides multiple hard numbers/percentages, reduces evasion score.
        numeric_bonus = min(0.20, lex["numeric_count"] * 0.04)

        # 5. Elaboration penalty (filibustering or abrupt dismissals):
        elaboration_penalty = 0.0
        if lex["a_words"] < 15 and lex["q_words"] > 40:
            # Abrupt non-answer to a substantial question
            elaboration_penalty = 0.15
        elif lex["length_ratio"] > 8.0 and lex["deflection_count"] > 0:
            # Running out the clock while deflecting
            elaboration_penalty = 0.10

        composite_evasiveness = (
            evasion_base +
            deflection_penalty +
            hedging_penalty +
            elaboration_penalty -
            numeric_bonus
        )
        # Bound strictly between 0.0 and 1.0
        composite_evasiveness = max(0.0, min(1.0, composite_evasiveness))

        return {
            "evasiveness_score": round(composite_evasiveness, 4),
            "semantic_relevance": round(relevance, 4),
            "deflection_matches": lex["deflection_count"],
            "uncertainty_density": round(lex["uncertainty_density"], 4),
            "numeric_count": lex["numeric_count"],
            "length_ratio": round(lex["length_ratio"], 2),
            "q_words": lex["q_words"],
            "a_words": lex["a_words"]
        }


if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), "data", "qa_pairs.parquet")
    if not os.path.exists(data_path):
        print(f"[!] {data_path} not found.")
        exit(1)

    print(f"[*] Initializing EvasivenessScorer...")
    scorer = EvasivenessScorer(use_neural_cross_encoder=False)

    print(f"[*] Loading sample pairs from {data_path}...")
    df = pd.read_parquet(data_path)

    # 1. Test on known direct answer
    sample_direct = df[df['answer'].str.contains(r'(?i)(was \$[0-9]|is [0-9]+%|guidance is)', regex=True)].iloc[0]
    res_direct = scorer.score_pair(sample_direct['question'], sample_direct['answer'])

    # 2. Test on known evasive / refusal answer
    sample_evasive = df[df['answer'].str.contains(r'(?i)(not going to comment|don\'t break that out|not going to give)', regex=True)].iloc[0]
    res_evasive = scorer.score_pair(sample_evasive['question'], sample_evasive['answer'])

    print("\n" + "=" * 80)
    print("                     SCORING ENGINE BENCHMARK TEST")
    print("=" * 80)
    print(f"\n[Case 1: Direct Financial Guidance ({sample_direct['ticker']})]")
    print(f"Question snippet: {repr(sample_direct['question'][:120])}")
    print(f"Answer snippet:   {repr(sample_direct['answer'][:120])}")
    print(f"  ──► Composite Evasiveness Score: {res_direct['evasiveness_score']:.4f} (Expected Low)")
    print(f"      Semantic Relevance: {res_direct['semantic_relevance']:.4f} | Numeric Count: {res_direct['numeric_count']} | Deflections: {res_direct['deflection_matches']}")

    print(f"\n[Case 2: Explicit Evasion / Refusal ({sample_evasive['ticker']})]")
    print(f"Question snippet: {repr(sample_evasive['question'][:120])}")
    print(f"Answer snippet:   {repr(sample_evasive['answer'][:120])}")
    print(f"  ──► Composite Evasiveness Score: {res_evasive['evasiveness_score']:.4f} (Expected High)")
    print(f"      Semantic Relevance: {res_evasive['semantic_relevance']:.4f} | Numeric Count: {res_evasive['numeric_count']} | Deflections: {res_evasive['deflection_matches']}")
    print("=" * 80)

