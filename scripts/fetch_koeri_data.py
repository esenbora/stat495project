#!/usr/bin/env python3
"""
KOERI (Kandilli) Deprem Verisi Çekme Scripti
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from pathlib import Path
from io import StringIO

class KOERIDataFetcher:
    """KOERI web sayfasından deprem verilerini çeken sınıf"""

    def __init__(self):
        # KOERI son depremler sayfası
        self.base_url = "http://www.koeri.boun.edu.tr/scripts"
        self.latest_url = f"{self.base_url}/lst0.asp"  # Son depremler
        self.list500_url = f"{self.base_url}/lst1.asp"  # Son 500 deprem

        # Veri klasörü
        self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def parse_koeri_html(self, html_content):
        """
        KOERI HTML sayfasını parse eder

        Args:
            html_content: HTML içeriği

        Returns:
            pandas.DataFrame: Parse edilmiş deprem verileri
        """
        # <pre> tag'i içindeki veriyi bul
        pre_pattern = r'<pre>(.*?)</pre>'
        pre_match = re.search(pre_pattern, html_content, re.DOTALL)

        if not pre_match:
            print("❌ <pre> tag'i bulunamadı")
            return pd.DataFrame()

        pre_content = pre_match.group(1)

        # Satırları ayır
        lines = pre_content.strip().split('\n')

        # Başlık satırlarını atla (genellikle ilk 3-4 satır açıklama, sonra başlık)
        data_lines = []
        header_found = False

        for line in lines:
            # Başlık satırını bul (Tarih sütunu ile başlayan)
            if 'Tarih' in line and 'Saat' in line:
                header_found = True
                continue

            # Çizgi satırlarını atla
            if '---' in line or '===' in line or '...' in line:
                continue

            # Boş satırları atla
            if not line.strip():
                continue

            # Başlıktan sonra gelen satırlar veridir
            if header_found:
                data_lines.append(line)

        if not data_lines:
            print("❌ Veri satırları bulunamadı")
            return pd.DataFrame()

        # Verileri parse et
        earthquakes = []

        for line in data_lines:
            try:
                # KOERI formatı (whitespace ile ayrılmış):
                # Tarih      Saat      Enlem(N)  Boylam(E) Derinlik(km)  MD   ML   Mw   Yer ...
                # 2025.11.20 21:03:22  38.6147   30.5735   8.2           -.-  1.5  -.-  LOCATION

                # \r karakterlerini temizle
                line = line.replace('\r', '').strip()

                # Whitespace'e göre ayır
                parts = line.split()

                if len(parts) < 9:  # En az 9 alan olmalı
                    continue

                # Tarih ve saat
                date_str = parts[0]  # YYYY.MM.DD
                time_str = parts[1]  # HH:MM:SS

                # Konum
                lat = float(parts[2])
                lon = float(parts[3])
                depth = float(parts[4])

                # Magnitude değerleri
                md = parts[5] if parts[5] != '-.-' else None
                ml = parts[6] if parts[6] != '-.-' else None
                mw = parts[7] if parts[7] != '-.-' else None
                ms = None  # KOERI sayfasında Ms yok

                # Lokasyon (kalan parçalar, son 2-3 "Çözüm Niteliği" olabilir)
                # "İlksel" veya benzeri çözüm niteliğini çıkar
                location_parts = parts[8:]

                # Son kelime genellikle "İlksel" gibi çözüm niteliği
                if location_parts and location_parts[-1] in ['İlksel', 'İlksel\r', 'Revize']:
                    location_parts = location_parts[:-1]

                location = ' '.join(location_parts)

                # ISO format tarih-saat
                datetime_str = f"{date_str.replace('.', '-')}T{time_str}"

                # En büyük magnitude'u seç
                magnitudes = [md, ml, mw, ms]
                magnitudes = [float(m) for m in magnitudes if m and m != '-.-']
                magnitude = max(magnitudes) if magnitudes else None

                # Magnitude tipi
                mag_type = None
                if ml and ml != '-.-':
                    mag_type = 'ML'
                elif mw and mw != '-.-':
                    mag_type = 'Mw'
                elif md and md != '-.-':
                    mag_type = 'MD'
                elif ms and ms != '-.-':
                    mag_type = 'Ms'

                earthquake = {
                    'date': datetime_str,
                    'latitude': lat,
                    'longitude': lon,
                    'depth': depth,
                    'magnitude': magnitude,
                    'magnitude_type': mag_type,
                    'MD': md,
                    'ML': ml,
                    'Mw': mw,
                    'Ms': ms,
                    'location': location,
                    'provider': 'KOERI'
                }

                earthquakes.append(earthquake)

            except Exception as e:
                # Parse hatası olan satırları atla
                # print(f"Parse hatası: {e} - Satır: {line[:50]}")  # Debug
                continue

        df = pd.DataFrame(earthquakes)
        return df

    def fetch_latest(self):
        """Son depremleri çeker"""
        print("📡 KOERI son depremler çekiliyor...")

        try:
            response = requests.get(self.latest_url, timeout=30)
            response.encoding = 'windows-1254'  # KOERI sayfası windows-1254 encoding kullanıyor

            if response.status_code == 200:
                df = self.parse_koeri_html(response.text)
                print(f"✅ {len(df)} kayıt çekildi")
                return df
            else:
                print(f"❌ HTTP {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Hata: {e}")
            return pd.DataFrame()

    def fetch_last_500(self):
        """Son 500 depremi çeker"""
        print("📡 KOERI son 500 deprem çekiliyor...")

        try:
            response = requests.get(self.list500_url, timeout=30)
            response.encoding = 'windows-1254'  # KOERI sayfası windows-1254 encoding kullanıyor

            if response.status_code == 200:
                df = self.parse_koeri_html(response.text)
                print(f"✅ {len(df)} kayıt çekildi")
                return df
            else:
                print(f"❌ HTTP {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Hata: {e}")
            return pd.DataFrame()

    def save_data(self, df, filename):
        """Veriyi dosyaya kaydeder"""
        if df.empty:
            print("⚠️  Kaydedilecek veri yok")
            return None

        filepath = self.data_dir / filename

        # CSV olarak kaydet
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"💾 Veri kaydedildi: {filepath}")

        # JSON olarak da kaydet
        json_filepath = filepath.with_suffix('.json')
        df.to_json(json_filepath, orient='records', force_ascii=False, indent=2)
        print(f"💾 JSON kaydedildi: {json_filepath}")

        # Özet bilgi
        print(f"\n📊 Veri Özeti:")
        print(f"  - Toplam kayıt: {len(df)}")

        if 'date' in df.columns and len(df) > 0:
            print(f"  - Tarih aralığı: {df['date'].min()} - {df['date'].max()}")

        if 'magnitude' in df.columns and len(df) > 0:
            valid_mags = df['magnitude'].dropna()
            if len(valid_mags) > 0:
                print(f"  - Magnitude aralığı: {valid_mags.min():.2f} - {valid_mags.max():.2f}")

        return filepath


def main():
    """Ana fonksiyon"""
    print("=" * 60)
    print("🌍 KOERI (KANDİLLİ) DEPREM VERİSİ ÇEKME ARACI")
    print("=" * 60)

    fetcher = KOERIDataFetcher()

    # Son 500 depremi çek
    print("\n📅 Son 500 deprem çekiliyor...\n")
    df = fetcher.fetch_last_500()

    if not df.empty:
        print("\nİlk 5 kayıt:")
        print(df.head())

        # Verileri kaydet
        fetcher.save_data(df, "koeri_last_500.csv")

        print("\n" + "=" * 60)
        print("✅ BAŞARIYLA TAMAMLANDI!")
        print("=" * 60)

        print("\n⚠️  NOT: KOERI web sayfası sadece son ~500 depremi gösteriyor.")
        print("Daha eski veriler için KOERI'nin zeqdb arama sayfasını kullanmamız gerekiyor.")
        print("Web sayfası: http://www.koeri.boun.edu.tr/sismo/zeqdb/")
    else:
        print("❌ Veri çekilemedi")


if __name__ == "__main__":
    main()
