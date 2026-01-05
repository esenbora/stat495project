#!/usr/bin/env python3
"""
AFAD Deprem Verisi Çekme Scripti
Bu script AFAD API'sinden deprem verilerini çeker ve kaydeder.
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

class AFADDataFetcher:
    """AFAD API'sinden deprem verilerini çeken sınıf"""

    def __init__(self):
        # Yeni v2 API endpoint
        self.api_url_v2 = "https://servisnet.afad.gov.tr/apigateway/deprem/apiv2/event/filter"
        # Eski API endpoint (fallback)
        self.api_url_old = "https://deprem.afad.gov.tr/EventData/GetEventsByFilter"

        # Veri klasörü
        self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data_v2(self, start_date, end_date, min_magnitude=0.0):
        """
        Yeni v2 API ile veri çeker (GET request)

        Args:
            start_date: Başlangıç tarihi (datetime)
            end_date: Bitiş tarihi (datetime)
            min_magnitude: Minimum magnitude (varsayılan 0.0)

        Returns:
            list: Deprem verileri listesi
        """
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")

        params = {
            "start": start_str,
            "end": end_str,
            "minmag": min_magnitude
        }

        try:
            print(f"📡 v2 API çağrısı: {start_str} - {end_str}")
            response = requests.get(self.api_url_v2, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, dict) and 'data' in data:
                return data['data']
            elif isinstance(data, list):
                return data
            else:
                print(f"⚠️  Beklenmeyen veri formatı: {type(data)}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"❌ v2 API hatası: {e}")
            return None

    def fetch_data_old(self, start_date, end_date, skip=0, take=1000):
        """
        Eski API ile veri çeker (POST request)

        Args:
            start_date: Başlangıç tarihi (datetime)
            end_date: Bitiş tarihi (datetime)
            skip: Atlanacak kayıt sayısı
            take: Çekilecek kayıt sayısı

        Returns:
            list: Deprem verileri listesi
        """
        event_filter = {
            "EventSearchFilterList": [
                {"FilterType": 8, "Value": start_date.isoformat()},
                {"FilterType": 9, "Value": end_date.isoformat()},
            ],
            "Skip": skip,
            "Take": take,
            "SortDescriptor": {"field": "eventDate", "dir": "desc"},
        }

        try:
            print(f"📡 Eski API çağrısı: {start_date.date()} - {end_date.date()} (skip={skip})")
            response = requests.post(self.api_url_old, json=event_filter, timeout=30)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, dict) and 'data' in data:
                return data['data']
            elif isinstance(data, list):
                return data
            else:
                print(f"⚠️  Beklenmeyen veri formatı: {type(data)}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"❌ Eski API hatası: {e}")
            return None

    def fetch_all_data(self, start_date, end_date, use_v2=True, min_magnitude=0.0):
        """
        Belirtilen tarih aralığındaki tüm verileri çeker

        Args:
            start_date: Başlangıç tarihi (datetime veya str)
            end_date: Bitiş tarihi (datetime veya str)
            use_v2: v2 API kullan (varsayılan True)
            min_magnitude: Minimum magnitude

        Returns:
            list: Tüm deprem verileri
        """
        # String ise datetime'a çevir
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        all_data = []

        if use_v2:
            # v2 API tek seferde çekiyor
            data = self.fetch_data_v2(start_date, end_date, min_magnitude)
            if data is not None:
                all_data.extend(data)
                print(f"✅ {len(data)} kayıt çekildi")
            else:
                print("⚠️  v2 API başarısız, eski API'ye geçiliyor...")
                use_v2 = False

        if not use_v2:
            # Eski API pagination gerektiriyor
            skip = 0
            take = 1000

            while True:
                data = self.fetch_data_old(start_date, end_date, skip, take)

                if data is None or len(data) == 0:
                    break

                all_data.extend(data)
                print(f"✅ {len(data)} kayıt çekildi (toplam: {len(all_data)})")

                if len(data) < take:
                    break

                skip += take
                time.sleep(1)  # Rate limiting

        return all_data

    def fetch_historical_data(self, start_year=2000, min_magnitude=0.0, chunk_months=6):
        """
        Tarihsel verileri çeker (belirtilen yıldan bugüne)

        Args:
            start_year: Başlangıç yılı (varsayılan 2000)
            min_magnitude: Minimum magnitude
            chunk_months: Kaç aylık dönemlerde çekeceği

        Returns:
            pandas.DataFrame: Tüm veriler
        """
        start_date = datetime(start_year, 1, 1)
        end_date = datetime.now()

        all_data = []
        current_date = start_date

        print(f"\n🚀 Tarihsel veri çekiliyor: {start_date.date()} - {end_date.date()}")
        print(f"📊 Minimum magnitude: {min_magnitude}")
        print(f"⏱️  Dönem aralığı: {chunk_months} ay\n")

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_months*30), end_date)

            data = self.fetch_all_data(current_date, chunk_end, use_v2=True, min_magnitude=min_magnitude)

            if data:
                all_data.extend(data)
                print(f"📈 Toplam kayıt: {len(all_data)}\n")

            current_date = chunk_end
            time.sleep(2)  # Rate limiting

        print(f"\n✨ Toplam {len(all_data)} deprem kaydı çekildi!")

        # DataFrame'e çevir
        if all_data:
            df = pd.DataFrame(all_data)
            return df
        else:
            return pd.DataFrame()

    def save_data(self, df, filename=None):
        """
        Veriyi dosyaya kaydeder

        Args:
            df: pandas DataFrame
            filename: Dosya adı (opsiyonel)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"afad_earthquakes_{timestamp}.csv"

        filepath = self.data_dir / filename

        # CSV olarak kaydet
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"\n💾 Veri kaydedildi: {filepath}")

        # JSON olarak da kaydet
        json_filepath = filepath.with_suffix('.json')
        df.to_json(json_filepath, orient='records', force_ascii=False, indent=2)
        print(f"💾 JSON kaydedildi: {json_filepath}")

        # Özet bilgi
        print(f"\n📊 Veri Özeti:")
        print(f"  - Toplam kayıt: {len(df)}")
        if len(df) > 0:
            date_col = 'date' if 'date' in df.columns else 'eventDate'
            if date_col in df.columns:
                print(f"  - Tarih aralığı: {df[date_col].min()} - {df[date_col].max()}")
            if 'magnitude' in df.columns:
                try:
                    mag_min = float(df['magnitude'].min())
                    mag_max = float(df['magnitude'].max())
                    print(f"  - Magnitude aralığı: {mag_min:.2f} - {mag_max:.2f}")
                except (ValueError, TypeError):
                    print(f"  - Magnitude aralığı: {df['magnitude'].min()} - {df['magnitude'].max()}")

        return filepath


def main():
    """Ana fonksiyon"""
    print("=" * 60)
    print("🌍 AFAD DEPREM VERİSİ ÇEKME ARACI")
    print("=" * 60)

    fetcher = AFADDataFetcher()

    # Örnek 1: Son 30 günün verileri
    print("\n📅 Test: Son 30 günün verileri çekiliyor...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    data = fetcher.fetch_all_data(start_date, end_date)

    if data:
        df = pd.DataFrame(data)
        print(f"\n✅ {len(df)} kayıt bulundu")
        print("\nİlk 3 kayıt:")
        print(df.head(3))

        # Test verisini kaydet
        fetcher.save_data(df, "afad_test_30days.csv")
    else:
        print("❌ Veri çekilemedi")

    # Örnek 2: Tarihsel veri çekme (yorumdan çıkarın)
    # print("\n" + "="*60)
    # print("📜 TARİHSEL VERİ ÇEKME")
    # print("="*60)
    # df_historical = fetcher.fetch_historical_data(start_year=2020, min_magnitude=2.0, chunk_months=3)
    # if not df_historical.empty:
    #     fetcher.save_data(df_historical, "afad_historical_2020_present.csv")


if __name__ == "__main__":
    main()
