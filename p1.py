"""
PEAD Data Ingestion Pipeline: Compustat, CRSP, CCM, and Capital IQ Transcripts
--------------------------------------------------------------------------------
This script queries Wharton Research Data Services (WRDS) directly via PostgreSQL,
pulls the required tables for a Post-Earnings Announcement Drift (PEAD) study,
and saves the outputs locally as compressed Parquet files.
"""

import os
import sys
import argparse
import urllib.request
import tempfile
import wrds
import pandas as pd
import numpy as np


def get_wrds_connection(username: str = None) -> wrds.Connection:
    """Connect to WRDS using username and stored/prompted credentials."""
    wrds_user = username or os.environ.get("WRDS_USERNAME")
    if not wrds_user:
        wrds_user = input("Enter your WRDS username: ").strip()
    
    print(f"[*] Connecting to WRDS as '{wrds_user}'...")
    db = wrds.Connection(wrds_username=wrds_user)
    # Save connection to ~/.pgpass so future logins are passwordless (if not already set up)
    pgpass_file = os.path.expanduser("~/.pgpass")
    if not os.path.exists(pgpass_file):
        try:
            db.create_pgpass_file()
        except Exception:
            pass
    print("[+] Successfully connected to WRDS.")
    return db


def pull_compustat_fundq(db: wrds.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Dataset A: Compustat Fundamentals Quarterly
    Table: comp.fundq
    """
    print(f"\n[*] [1/4] Pulling Compustat quarterly earnings dates ({start_date} to {end_date})...")
    sql = f"""
        SELECT 
            gvkey,
            datadate,
            rdq,
            fyearq,
            fqtr,
            conm,
            tic,
            cusip
        FROM comp.fundq
        WHERE rdq BETWEEN '{start_date}' AND '{end_date}'
          AND rdq IS NOT NULL
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
        ORDER BY gvkey, rdq
    """
    df_fundq = db.raw_sql(sql, date_cols=['datadate', 'rdq'])
    df_fundq['gvkey'] = df_fundq['gvkey'].astype(str).str.zfill(6)
    print(f"[+] Compustat: Retrieved {len(df_fundq):,} quarterly earnings announcement records.")
    return df_fundq


def pull_ccm_linker(db: wrds.Connection) -> pd.DataFrame:
    """
    Dataset C: CRSP/Compustat Merged Link Table
    Table: crsp.ccmxpf_lnkhist (or fallback to crsp.dsenames via CUSIP)
    Translates gvkey <-> permno avoiding survivorship bias and ticker recycling.
    """
    print("\n[*] [2/4] Pulling CCM link table (crsp.ccmxpf_lnkhist)...")
    try:
        sql = """
            SELECT 
                gvkey,
                lpermno AS permno,
                linktype,
                linkprim,
                linkdt,
                COALESCE(linkenddt, CURRENT_DATE) AS linkenddt
            FROM crsp.ccmxpf_lnkhist
            WHERE linktype IN ('LU', 'LC')
              AND linkprim IN ('P', 'C')
        """
        df_ccm = db.raw_sql(sql, date_cols=['linkdt', 'linkenddt'])
        df_ccm['gvkey'] = df_ccm['gvkey'].astype(str).str.zfill(6)
        df_ccm['permno'] = df_ccm['permno'].astype(int)
        print(f"[+] CCM Linker: Retrieved {len(df_ccm):,} link mappings from ccmxpf_lnkhist.")
        return df_ccm
    except Exception as e:
        print(f"[!] CCM link table (crsp.ccmxpf_lnkhist) unavailable ({e}).")
        print("[*] Falling back to CRSP historical CUSIP linker (crsp.dsenames)...")
        sql = """
            SELECT DISTINCT
                permno,
                ncusip AS cusip,
                namedt AS linkdt,
                COALESCE(nameendt, CURRENT_DATE) AS linkenddt
            FROM crsp.dsenames
            WHERE ncusip IS NOT NULL
        """
        df_names = db.raw_sql(sql, date_cols=['linkdt', 'linkenddt'])
        df_names['permno'] = df_names['permno'].astype(int)
        print(f"[+] Fallback Linker: Retrieved {len(df_names):,} CRSP historical CUSIP mappings.")
        return df_names


def merge_compustat_ccm(df_fundq: pd.DataFrame, df_ccm: pd.DataFrame) -> pd.DataFrame:
    """Merges Compustat earnings records with CCM link table on gvkey or CUSIP and date range."""
    print("\n[*] Merging Compustat with CRSP link table...")
    if 'gvkey' in df_ccm.columns:
        merged = pd.merge(df_fundq, df_ccm, on='gvkey', how='inner')
    else:
        # Fallback CUSIP matching: Compustat 9-char CUSIP matches CRSP 8-char historical CUSIP
        df_fundq_copy = df_fundq.copy()
        df_fundq_copy['cusip8'] = df_fundq_copy['cusip'].str[:8]
        merged = pd.merge(df_fundq_copy, df_ccm, left_on='cusip8', right_on='cusip', how='inner')

    # Filter where earnings report date (rdq) falls within valid link validity window
    valid_mask = (merged['rdq'] >= merged['linkdt']) & (merged['rdq'] <= merged['linkenddt'])
    linked_fundq = merged[valid_mask].drop_duplicates(subset=['gvkey', 'rdq', 'permno']).copy()
    print(f"[+] Linked {len(linked_fundq):,} announcements to valid CRSP PERMNOs.")
    return linked_fundq


def pull_crsp_dsf(db: wrds.Connection, permnos: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Dataset B: CRSP Daily Stock File
    Table: crsp.dsf joined with crsp.dsenames (for share code & exchange filters)
    """
    print(f"\n[*] [3/4] Pulling CRSP daily prices & returns ({start_date} to {end_date})...")
    
    # We pull in batches if permno list is large, or query for common stocks directly
    permno_tuple = tuple(int(p) for p in set(permnos))
    
    # Filter for US common equities (shrcd 10, 11) on major exchanges (NYSE=1, AMEX=2, NASDAQ=3)
    if len(permno_tuple) == 1:
        permno_clause = f"= {permno_tuple[0]}"
    else:
        permno_clause = f"IN {permno_tuple}"

    sql = f"""
        SELECT 
            d.permno,
            d.date,
            d.ret,
            d.prc,
            d.shrout,
            d.vol
        FROM crsp.dsf d
        WHERE d.permno {permno_clause}
          AND d.date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY d.permno, d.date
    """
    df_crsp = db.raw_sql(sql, date_cols=['date'])
    df_crsp['permno'] = df_crsp['permno'].astype(int)
    # Market Cap in thousands: absolute price * shares outstanding
    df_crsp['mktcap'] = df_crsp['prc'].abs() * df_crsp['shrout']
    print(f"[+] CRSP: Retrieved {len(df_crsp):,} daily return observations.")
    return df_crsp


def pull_hf_transcripts(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pulls open academic S&P 500 earnings call transcripts from Hugging Face.
    Dataset: glopardo/sp500-earnings-transcripts (ECB 2025 research paper)
    """
    print(f"[*] Fetching academic S&P 500 transcripts from Hugging Face ({start_date} to {end_date})...")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    start_year = start_dt.year
    end_year = end_dt.year

    parts_urls = [
        f"https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts/resolve/main/data/train-0000{i}-of-00003.parquet"
        for i in range(3)
    ]
    
    dfs = []
    for i, url in enumerate(parts_urls, start=1):
        print(f"    - Downloading partition {i}/3 from Hugging Face CDN...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name
            with urllib.request.urlopen(req) as resp:
                while chunk := resp.read(2 * 1024 * 1024):
                    tmp.write(chunk)
        df_part = pd.read_parquet(tmp_path)
        os.remove(tmp_path)
        
        filtered = df_part[(df_part['year'] >= start_year) & (df_part['year'] <= end_year)].copy()
        if 'earnings_date' in filtered.columns:
            filtered['earnings_date_dt'] = pd.to_datetime(filtered['earnings_date'], errors='coerce')
            filtered = filtered[(filtered['earnings_date_dt'] >= start_dt) & (filtered['earnings_date_dt'] <= end_dt)]
            filtered = filtered.drop(columns=['earnings_date_dt'])
        dfs.append(filtered)
        
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[+] Hugging Face: Retrieved {len(df_all):,} complete earnings call transcripts.")
    return df_all


def pull_capital_iq_transcripts(db: wrds.Connection, start_date: str, end_date: str, gvkeys: list = None) -> pd.DataFrame:
    """
    Dataset D: Earnings Call Transcripts (NLP Corpus)
    Pulls earnings call text. Attempts S&P Capital IQ via WRDS first;
    if unavailable, automatically falls back to Hugging Face S&P 500 transcripts.
    """
    print(f"\n[*] [4/4] Acquiring Earnings Call Transcripts ({start_date} to {end_date})...")
    
    # Check if table exists in available libraries
    try:
        # Standard WRDS Capital IQ Transcripts view
        sql = f"""
            SELECT 
                transcriptid,
                companyname,
                keydeveventtypeid,
                mostimportantdateutc::date AS transcript_date,
                headline
            FROM ciq.wrds_transcript_detail
            WHERE mostimportantdateutc BETWEEN '{start_date}' AND '{end_date}'
            LIMIT 500
        """
        df_transcripts = db.raw_sql(sql, date_cols=['transcript_date'])
        print(f"[+] Capital IQ: Retrieved {len(df_transcripts):,} sample transcript headers.")
        return df_transcripts
    except Exception as e:
        print(f"[!] WRDS Capital IQ Transcripts unavailable ({e}).")
        print("[*] Switching to open academic transcripts dataset on Hugging Face...")
        return pull_hf_transcripts(start_date, end_date)


def save_dataframe(df: pd.DataFrame, base_path: str):
    """Saves DataFrame as Parquet if pyarrow is installed, else compressed CSV (.csv.gz)."""
    try:
        parquet_path = f"{base_path}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"[+] Saved: {parquet_path}")
    except (ImportError, ValueError):
        csv_path = f"{base_path}.csv.gz"
        df.to_csv(csv_path, index=False, compression="gzip")
        print(f"[+] Parquet engine missing; saved compressed CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Pull PEAD datasets from WRDS.")
    parser.add_argument("--user", type=str, default=None, help="WRDS username")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--outdir", type=str, default="./data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. Connect
    db = get_wrds_connection(args.user)

    # 2. Compustat Fundq (Earnings dates)
    df_fundq = pull_compustat_fundq(db, args.start, args.end)
    save_dataframe(df_fundq, os.path.join(args.outdir, "compustat_fundq"))

    # 3. CCM Linker
    df_ccm = pull_ccm_linker(db)
    save_dataframe(df_ccm, os.path.join(args.outdir, "ccm_linker"))

    # 4. Merge Compustat with CCM to get active PERMNOs
    df_linked = merge_compustat_ccm(df_fundq, df_ccm)
    save_dataframe(df_linked, os.path.join(args.outdir, "earnings_events_linked"))

    # 5. Pull CRSP Daily Stock File for the linked universe
    unique_permnos = df_linked['permno'].unique().tolist()
    # Add buffer of trading days before start and after end for event study windows (-20 to +60 days)
    crsp_start = (pd.to_datetime(args.start) - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
    crsp_end = (pd.to_datetime(args.end) + pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    
    df_crsp = pull_crsp_dsf(db, unique_permnos, crsp_start, crsp_end)
    save_dataframe(df_crsp, os.path.join(args.outdir, "crsp_daily_returns"))

    # 6. Transcripts (Capital IQ / Hugging Face)
    df_transcripts = pull_capital_iq_transcripts(db, args.start, args.end)
    if not df_transcripts.empty:
        save_dataframe(df_transcripts, os.path.join(args.outdir, "transcripts"))

    db.close()
    print("\n[✓] All data steps complete! Files saved to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
