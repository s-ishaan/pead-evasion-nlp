"""
Module 1: Earnings Call Q&A Extractor & Turn Tokenizer
-------------------------------------------------------
Handles:
1. isolate_qa_section: Slices the transcript at the transition from prepared remarks to Q&A.
2. tokenize_speakers: Tokenizes single-line / unformatted transcript text into speaker turns.
3. extract_qa_pairs: Groups analyst questions with company management answers.
"""

import re
from typing import List, Dict, Tuple, Optional


# -------------------------------------------------------------------------
# Transition Regex Patterns (Ranked Cascade)
# -------------------------------------------------------------------------
# Pattern 0: Direct early interview markers (e.g. Netflix Q&A interview format)
_EARLY_INTERVIEW_PATTERN = re.compile(
    r'(?:Q\s*-\s*[A-Z][a-zA-Z\.\'\-\s]+:|'
    r'(?:(?:and\s+|we\s+will\s+take\s+)?(?:our|your|the)?\s*(?:first|1st)\s+question\s+(?:comes?|is|will\s+come|is\s+coming|from)\s+))',
    re.IGNORECASE
)

# Pattern 1: Operator cueing the Q&A session or first question
_OPERATOR_CUE_PATTERN = re.compile(
    r'(?:^|[.?!\]\s])\s*(Operator\s*:\s*.*?(?:'
    r'(?:first|1st)\s+question|'
    r'(?:begin|conduct|start).*?(?:question-and-answer|q\s*&\s*a)|'
    r'open\s+the\s+lines?|'
    r'we\s+will\s+take\s+(?:our\s+)?first\s+question))',
    re.IGNORECASE | re.DOTALL
)

# Pattern 2: Executive / IR handoff to Q&A
_HANDOFF_PATTERN = re.compile(
    r'(?:(?:now\s+)?(?:open|turn)(?:\s+\w+){0,3}\s+(?:the\s+)?(?:call|line|lines?|floor|it)\s+(?:up\s+)?(?:to|for|over\s+to)\s+(?:your\s+)?(?:questions?|q\s*&\s*a)|'
    r'(?:now\s+)?(?:begin|conduct|start)\s+(?:the\s+)?(?:question-and-answer|q\s*&\s*a)\s+(?:session|period|portion)|'
    r'(?:ready|happy)\s+to\s+take\s+(?:your\s+)?questions|'
    r'turn (?:the call|it) over to (?:the operator|.*?) for (?:the\s+)?(?:questions?|q\s*&\s*a)|'
    r'turn it over to you.*?(?:first\s+)?question)',
    re.IGNORECASE
)

# Pattern 3: Standalone "First Question" cues
_FIRST_QUESTION_PATTERN = re.compile(
    r'(?:(?:and\s+|we\s+will\s+take\s+)?(?:our|your|the)?\s*(?:first|1st)\s+question\s+(?:comes?|is|will\s+come|is\s+coming|from)\s+)',
    re.IGNORECASE
)

# Delimiter for continuous (single-line) text
# Transcripts lack newlines (\n). Speakers are preceded by sentence endings (.?!])
# followed by standard names (1 to 4 capitalized words/initials) and a colon.
_SPEAKER_PATTERN = re.compile(
    r'(?:(?<=[.?!\]])|(?<=\A))\s*'
    r'((?:[QA]\s*-\s*)?(?:Operator|Unidentified\s+(?:Participant|Analyst|Speaker)|[A-Z][a-zA-Z\.\'\-]+(?:\s+[A-Z][a-zA-Z\.\'\-]+){0,3}))'
    r'\s*:\s+'
)


def isolate_qa_section(transcript_text: str, min_offset: int = 1500) -> Optional[str]:
    """
    Slices an earnings call transcript at the transition phrase to Q&A,
    discarding the introductory disclaimer and prepared remarks.

    Parameters:
    -----------
    transcript_text : str
        Full transcript string.
    min_offset : int, default=1500
        Minimum character offset to bypass opening housekeeping remarks.
        (Standard calls transition around 12,000-25,000 chars; short calls around ~1,500 chars.)

    Returns:
    --------
    Optional[str]:
        The sliced string starting at the Q&A transition, or None if no Q&A section exists.
    """
    if not transcript_text or len(transcript_text.strip()) == 0:
        return None

    # Check for direct interview or early question format (e.g. Netflix Q&A format)
    early_match = _EARLY_INTERVIEW_PATTERN.search(transcript_text[:min_offset])
    if early_match:
        return transcript_text[early_match.start():].strip()

    if len(transcript_text) <= min_offset:
        return None

    search_area = transcript_text[min_offset:]
    best_idx: Optional[int] = None

    # 1. High confidence: Operator cueing the Q&A session
    op_match = _OPERATOR_CUE_PATTERN.search(search_area)
    if op_match:
        best_idx = min_offset + op_match.start()

    # 2. Executive / IR handoff
    if best_idx is None:
        handoff_match = _HANDOFF_PATTERN.search(search_area)
        if handoff_match:
            best_idx = min_offset + handoff_match.start()

    # 3. Direct first question phrase
    if best_idx is None:
        fq_match = _FIRST_QUESTION_PATTERN.search(search_area)
        if fq_match:
            best_idx = min_offset + fq_match.start()

    # 4. Fallback: Second Operator occurrence
    # In standard calls, Operator appears at intro (#1) and when starting Q&A (#2).
    if best_idx is None:
        op_matches = list(re.finditer(r'(?:^|[.?!\]\s])\s*Operator\s*:\s*', transcript_text))
        if len(op_matches) > 1 and op_matches[1].start() >= min_offset:
            best_idx = op_matches[1].start()

    if best_idx is not None:
        return transcript_text[best_idx:].strip()

    return None


def tokenize_speakers(qa_text: str) -> List[Dict[str, str]]:
    """
    Parses a continuous Q&A text block into a sequential list of speaker turns.
    Works reliably on transcripts with zero newline characters.

    Parameters:
    -----------
    qa_text : str
        The sliced Q&A section text.

    Returns:
    --------
    List[Dict[str, str]]:
        A list of dicts with 'speaker' and 'text'.
    """
    if not qa_text:
        return []

    matches = list(_SPEAKER_PATTERN.finditer(qa_text))
    turns = []

    for i, match in enumerate(matches):
        speaker_name = match.group(1).strip()
        start_idx = match.end()

        if i + 1 < len(matches):
            end_idx = matches[i + 1].start()
        else:
            end_idx = len(qa_text)

        spoken_text = qa_text[start_idx:end_idx].strip()

        if spoken_text:
            turns.append({
                'speaker': speaker_name,
                'text': spoken_text
            })

    return turns


def extract_qa_pairs(turns: List[Dict[str, str]], min_words: int = 3) -> List[Tuple[str, str]]:
    """
    Pairs question turns with their corresponding management answer turns.

    Handles:
    - Explicit Q/A markers ('Q - Analyst', 'A - Executive') if present.
    - Standard dialogue where Operator introduces analyst, analyst asks (and does follow-ups),
      and company management answers.
    - Merges multiple management response turns into a single answer.

    Parameters:
    -----------
    turns : List[Dict[str, str]]
        List of turn dictionaries from `tokenize_speakers`.
    min_words : int, default=3
        Minimum words required for both question and answer to exclude empty filler.

    Returns:
    --------
    List[Tuple[str, str]]:
        A list of (question_text, answer_text) tuples.
    """
    pairs: List[Tuple[str, str]] = []
    current_q_speaker: Optional[str] = None
    current_q: List[str] = []
    current_a: List[str] = []

    def flush():
        nonlocal current_q, current_a, current_q_speaker
        if current_q and current_a:
            q_str = " ".join(current_q).strip()
            a_str = " ".join(current_a).strip()
            if len(q_str.split()) >= min_words and len(a_str.split()) >= min_words:
                pairs.append((q_str, a_str))
        current_q = []
        current_a = []
        current_q_speaker = None

    for turn in turns:
        speaker = turn['speaker']
        text = turn['text']

        # 1. Handle explicit Q - / A - prefixes if present
        if speaker.startswith('Q -') or speaker.startswith('Q-'):
            flush()
            current_q_speaker = speaker
            current_q.append(text)
            continue
        elif speaker.startswith('A -') or speaker.startswith('A-'):
            current_a.append(text)
            continue

        # 2. Operator cues mark transition between analysts/questions
        if 'Operator' in speaker:
            flush()
            continue

        # 3. Conversational role resolution
        if current_q_speaker is None:
            # First non-operator speaker in a round is the Questioner (Analyst)
            current_q_speaker = speaker
            current_q.append(text)
        elif speaker == current_q_speaker:
            # The Questioner spoke again
            if current_a:
                # If management already answered, this is a follow-up question
                flush()
                current_q_speaker = speaker
                current_q.append(text)
            else:
                # Continuing / clarifying the initial question
                current_q.append(text)
        else:
            # Different speaker (Company Executive) answering
            current_a.append(text)

    # Flush any remaining pair at the end of the transcript
    flush()
    return pairs


if __name__ == "__main__":
    import os
    import pandas as pd

    data_path = os.path.join(os.path.dirname(__file__), "data", "transcripts.parquet")
    if not os.path.exists(data_path):
        print(f"[!] {data_path} not found.")
        exit(1)

    print(f"[*] Reading sample transcript from {data_path}...")
    df = pd.read_parquet(data_path)
    sample_row = df.iloc[0]

    ticker = sample_row["ticker"]
    edate = sample_row["earnings_date"]
    raw_text = sample_row["transcript"]

    print(f"\n[+] Testing with {ticker} on {edate} (total length: {len(raw_text):,} chars):")
    qa_section = isolate_qa_section(raw_text)

    if qa_section is None:
        print("[!] No Q&A section detected.")
    else:
        print(f"[✓] Isolated Q&A Section: {len(qa_section):,} chars remaining.")
        print(f"    Preview: {qa_section[:250]}...\n")

        turns = tokenize_speakers(qa_section)
        print(f"[✓] Tokenized into {len(turns)} speaker turns.")

        pairs = extract_qa_pairs(turns)
        print(f"[✓] Extracted {len(pairs)} Q&A pairs.\n")

        if pairs:
            print("--- Example Extracted Pair #1 ---")
            print(f"Question ({len(pairs[0][0].split())} words):\n  {pairs[0][0][:300]}...\n")
            print(f"Answer ({len(pairs[0][1].split())} words):\n  {pairs[0][1][:300]}...")
