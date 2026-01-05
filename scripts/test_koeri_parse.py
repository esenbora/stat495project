#!/usr/bin/env python3
"""KOERI parse test"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fetch_koeri_data import KOERIDataFetcher

fetcher = KOERIDataFetcher()
df = fetcher.fetch_last_500()

print(f"\n✅ Toplam kayıt: {len(df)}")

if not df.empty:
    print("\nİlk 3 kayıt:")
    print(df.head(3))

    print("\nSütunlar:")
    print(df.columns.tolist())
