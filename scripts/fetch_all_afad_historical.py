#!/usr/bin/env python3
"""
AFAD'dan 1990-2025 arası TÜM tarihsel verileri çeker
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fetch_afad_data import AFADDataFetcher
from datetime import datetime

def main():
    print("=" * 70)
    print("🌍 AFAD TARİHSEL VERİ ÇEKME - TAM DATASET (1990-2025)")
    print("=" * 70)

    fetcher = AFADDataFetcher()

    # 1990'dan bugüne tüm veriler
    print("\n⚙️  Ayarlar:")
    print(f"  - Başlangıç yılı: 1990")
    print(f"  - Bitiş tarihi: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  - Minimum magnitude: 0.0 (tüm depremler)")
    print(f"  - Dönem aralığı: 6 ay\n")

    print("⏳ Veri çekme başlıyor... (Bu işlem 10-30 dakika sürebilir)\n")

    # Tüm tarihsel verileri çek
    df_historical = fetcher.fetch_historical_data(
        start_year=1990,
        min_magnitude=0.0,
        chunk_months=6
    )

    if not df_historical.empty:
        print("\n" + "=" * 70)
        print("💾 VERİ KAYDETME")
        print("=" * 70)

        # Ana dosyayı kaydet
        filepath = fetcher.save_data(df_historical, "afad_full_historical_1990_2025.csv")

        # Ek bilgiler
        print("\n📈 Detaylı İstatistikler:")

        if 'magnitude' in df_historical.columns:
            df_historical['magnitude_float'] = df_historical['magnitude'].astype(float)
            print(f"  - Ortalama magnitude: {df_historical['magnitude_float'].mean():.2f}")
            print(f"  - Medyan magnitude: {df_historical['magnitude_float'].median():.2f}")

            # Magnitude dağılımı
            print("\n  Magnitude Dağılımı:")
            print(f"    - M < 2.0: {len(df_historical[df_historical['magnitude_float'] < 2.0])} kayıt")
            print(f"    - M 2.0-3.9: {len(df_historical[(df_historical['magnitude_float'] >= 2.0) & (df_historical['magnitude_float'] < 4.0)])} kayıt")
            print(f"    - M 4.0-5.9: {len(df_historical[(df_historical['magnitude_float'] >= 4.0) & (df_historical['magnitude_float'] < 6.0)])} kayıt")
            print(f"    - M >= 6.0: {len(df_historical[df_historical['magnitude_float'] >= 6.0])} kayıt")

        if 'province' in df_historical.columns:
            print("\n  En Çok Deprem Olan İller (Top 10):")
            top_provinces = df_historical['province'].value_counts().head(10)
            for i, (province, count) in enumerate(top_provinces.items(), 1):
                print(f"    {i}. {province}: {count} deprem")

        print("\n" + "=" * 70)
        print("✅ BAŞARIYLA TAMAMLANDI!")
        print("=" * 70)
        print(f"\n📁 Veri dosyası: {filepath}")
        print(f"📊 Toplam kayıt: {len(df_historical):,}")
        print(f"📅 Tarih aralığı: 1990 - 2025 (36 yıl)")
        print("\n🎉 Tüm tarihsel veriler başarıyla çekildi ve kaydedildi!")

    else:
        print("\n❌ Veri çekilemedi!")

if __name__ == "__main__":
    main()
