"""
Dataset Extraction Script: Full Transcripts to Q&A Pairs
---------------------------------------------------------
Processes all ~11,494 earnings call transcripts from data/transcripts.parquet,
isolates Q&A sections, tokenizes speaker turns, pairs questions and answers,
and saves the resulting tabular dataset to data/qa_pairs.parquet.
"""

import os
import sys
import time
from typing import List, Dict, Any
import pandas as pd
from qa_extractor import isolate_qa_section, tokenize_speakers, extract_qa_pairs


INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "transcripts.parquet")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "qa_pairs.parquet")


def process_dataset(batch_size: int = 1000):
    if not os.path.exists(INPUT_PATH):
        print(f"[!] Error: {INPUT_PATH} does not exist.")
        sys.exit(1)

    print(f"[*] Loading transcript dataset from {INPUT_PATH}...")
    start_time = time.time()
    df = pd.read_parquet(INPUT_PATH)
    total_calls = len(df)
    print(f"[+] Loaded {total_calls:,} transcripts across {df['ticker'].nunique():,} unique tickers.")

    records: List[Dict[str, Any]] = []
    calls_with_qa = 0
    calls_without_qa = 0
    skipped_calls = []

    print(f"[*] Extracting Q&A pairs across all transcripts...")

    for idx, row in df.iterrows():
        raw_text = row["transcript"]
        qa_text = isolate_qa_section(raw_text)

        if not qa_text:
            calls_without_qa += 1
            skipped_calls.append((row["ticker"], str(row["earnings_date"])))
            continue

        turns = tokenize_speakers(qa_text)
        pairs = extract_qa_pairs(turns)

        if not pairs:
            calls_without_qa += 1
            skipped_calls.append((row["ticker"], str(row["earnings_date"])))
            continue

        calls_with_qa += 1

        for pair_idx, (q_str, a_str) in enumerate(pairs, start=1):
            records.append({
                "ticker": row["ticker"],
                "company": row.get("company", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "earnings_date": row["earnings_date"],
                "year": int(row["year"]) if pd.notnull(row["year"]) else None,
                "quarter": int(row["quarter"]) if pd.notnull(row["quarter"]) else None,
                "pair_id": pair_idx,
                "question": q_str,
                "answer": a_str,
                "question_words": len(q_str.split()),
                "answer_words": len(a_str.split()),
            })

        if (idx + 1) % batch_size == 0 or (idx + 1) == total_calls:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"    [{idx + 1:,} / {total_calls:,}] ({((idx + 1) / total_calls) * 100:.1f}%) | "
                  f"{len(records):,} pairs extracted | Rate: {rate:.0f} calls/sec")

    df_pairs = pd.DataFrame(records)
    print(f"\n[*] Saving {len(df_pairs):,} pairs to {OUTPUT_PATH}...")
    df_pairs.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    total_elapsed = time.time() - start_time

    print(f"[✓] Saved {OUTPUT_PATH} ({file_size_mb:.2f} MB) in {total_elapsed:.1f}s.")

    # -------------------------------------------------------------
    # Overview Statistics & Findings
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    DATASET EXTRACTION OVERVIEW")
    print("=" * 80)
    print(f"Total Transcripts Ingested:    {total_calls:,}")
    print(f"Calls with Q&A Extracted:      {calls_with_qa:,} ({calls_with_qa / total_calls * 100:.2f}%)")
    print(f"Calls without Q&A:             {calls_without_qa:,} ({calls_without_qa / total_calls * 100:.2f}%)")
    print(f"Total Q&A Pairs Extracted:     {len(df_pairs):,}")
    print(f"Average Pairs per Call:        {len(df_pairs) / max(1, calls_with_qa):.1f}")
    print(f"Median Pairs per Call:         {df_pairs.groupby(['ticker', 'earnings_date'])['pair_id'].count().median():.1f}")
    print("-" * 80)
    print("Word Count Statistics:")
    print(f"  Questions: mean = {df_pairs['question_words'].mean():.1f} words, "
          f"median = {df_pairs['question_words'].median():.0f} words, "
          f"max = {df_pairs['question_words'].max():,} words")
    print(f"  Answers:   mean = {df_pairs['answer_words'].mean():.1f} words, "
          f"median = {df_pairs['answer_words'].median():.0f} words, "
          f"max = {df_pairs['answer_words'].max():,} words")
    print("-" * 80)
    print("Distribution by Sector (Top 8):")
    for sec, count in df_pairs.groupby("sector")["pair_id"].count().nlargest(8).items():
        print(f"  {sec:<30} {count:>8,d} pairs")
    print("=" * 80)


if __name__ == "__main__":
    process_dataset()

