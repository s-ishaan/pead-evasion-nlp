"""
Interactive Transcript Search & Viewer Tool
-------------------------------------------
Search and inspect earnings call transcripts locally from ./data/transcripts.parquet.

Examples:
    # 1. List calls for a specific company
    python3 search_transcripts.py --ticker NVDA --year 2023

    # 2. Search for a keyword within transcripts (e.g. "supply chain", "guidance", "AI")
    python3 search_transcripts.py --ticker AAPL --keyword "iPhone"

    # 3. Search across all S&P 500 calls for a term
    python3 search_transcripts.py --year 2023 --keyword "recession" --limit 5

    # 4. Export a full transcript to a readable text file
    python3 search_transcripts.py --ticker MSFT --year 2023 --quarter 1 --export msft_call.txt
"""

import os
import argparse
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "transcripts.parquet")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cannot find {DATA_PATH}. Make sure p1.py has finished.")
    return pd.read_parquet(DATA_PATH)


def search(ticker=None, year=None, quarter=None, keyword=None, limit=10, export=None):
    print(f"[*] Loading transcripts from {DATA_PATH}...")
    df = load_data()
    
    # Apply filters
    if ticker:
        df = df[df["ticker"].str.upper() == ticker.upper()]
    if year:
        df = df[df["year"] == float(year)]
    if quarter:
        df = df[df["quarter"] == float(quarter)]
        
    print(f"[+] Found {len(df):,} transcripts matching metadata filters.")
    
    if keyword and not df.empty:
        # Case-insensitive substring search in the transcript text
        mask = df["transcript"].str.contains(keyword, case=False, na=False)
        df = df[mask]
        print(f"[+] {len(df):,} transcripts contain keyword: '{keyword}'")

    if df.empty:
        print("[!] No matching transcripts found.")
        return

    # Sort most recent first
    df = df.sort_values(by="earnings_date", ascending=False)
    
    # Export if requested
    if export:
        row = df.iloc[0]
        with open(export, "w", encoding="utf-8") as f:
            f.write(f"Company: {row['company']} ({row['ticker']})\n")
            f.write(f"Date: {row['earnings_date']} | Q{int(row['quarter'])} {int(row['year'])}\n")
            f.write("=" * 80 + "\n\n")
            f.write(row["transcript"])
        print(f"[✓] Exported full transcript to: {export}")
        return

    # Display matches
    results = df.head(limit)
    print("\n" + "=" * 90)
    print(f"{'TICKER':<8} {'COMPANY':<28} {'DATE':<12} {'QTR':<6} {'WORDS':<8} {'KEYWORD MATCHES'}")
    print("-" * 90)
    
    for _, row in results.iterrows():
        text = row["transcript"]
        words = len(text.split())
        kw_count = text.lower().count(keyword.lower()) if keyword else "-"
        qtr_str = f"Q{int(row['quarter'])} {int(row['year'])}"
        print(f"{row['ticker']:<8} {row['company'][:26]:<28} {row['earnings_date']:<12} {qtr_str:<6} {words:<8,d} {kw_count}")
    print("=" * 90)

    # Show snippet of the first result
    first = results.iloc[0]
    print(f"\n[Preview of top result: {first['company']} ({first['ticker']}) on {first['earnings_date']}]")
    text = first["transcript"]
    
    if keyword:
        # Find context around first keyword occurrence
        idx = text.lower().find(keyword.lower())
        start = max(0, idx - 150)
        end = min(len(text), idx + 250)
        snippet = text[start:end].replace("\n", " ").strip()
        print(f"... {snippet} ...\n")
    else:
        snippet = text[:400].replace("\n", " ").strip()
        print(f"{snippet} ...\n")

    if len(df) > limit:
        print(f"(* Showing top {limit} of {len(df):,} matches. Use --limit to see more or --export to save a full call.)")


def main():
    parser = argparse.ArgumentParser(description="Search and view earnings call transcripts.")
    parser.add_argument("--ticker", type=str, help="Stock ticker (e.g. AAPL, NVDA, MSFT)")
    parser.add_argument("--year", type=int, help="Year (2019-2024)")
    parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], help="Fiscal quarter (1-4)")
    parser.add_argument("--keyword", type=str, help="Search term inside transcript (e.g. 'margin', 'AI')")
    parser.add_argument("--limit", type=int, default=10, help="Max results to display (default: 10)")
    parser.add_argument("--export", type=str, help="Export top match to a .txt file (e.g. call.txt)")
    args = parser.parse_args()

    search(
        ticker=args.ticker,
        year=args.year,
        quarter=args.quarter,
        keyword=args.keyword,
        limit=args.limit,
        export=args.export,
    )


if __name__ == "__main__":
    main()

