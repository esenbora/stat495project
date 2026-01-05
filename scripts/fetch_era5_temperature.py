#!/usr/bin/env python3
"""
ERA5 Sıcaklık Verisi Çekme Scripti

Copernicus CDS'den ERA5 reanalysis sıcaklık verilerini çeker.
"""

import cdsapi
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

class ERA5TemperatureFetcher:
    """ERA5 sıcaklık verilerini çeken sınıf"""

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data" / "raw" / "era5"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # CDS API client
        try:
            self.client = cdsapi.Client()
            print("✅ CDS API bağlantısı başarılı")
        except Exception as e:
            print(f"❌ CDS API bağlantı hatası: {e}")
            print("\n⚠️  Lütfen önce setup_cds_api.py scriptini çalıştırın!")
            raise

    def fetch_turkey_temperature(self, year, month, variable='2m_temperature'):
        """
        Türkiye için belirli bir ay-yıl sıcaklık verisi çeker

        Args:
            year: Yıl
            month: Ay (1-12)
            variable: Değişken ('2m_temperature' veya 'surface_temperature')

        Returns:
            str: İndirilen dosya yolu
        """
        # Türkiye sınırları (yaklaşık)
        # Kuzey: 42°N, Güney: 36°N, Batı: 26°E, Doğu: 45°E
        area = [42, 26, 36, 45]  # [North, West, South, East]

        # Dosya adı
        filename = f"era5_turkey_temp_{year}_{month:02d}.nc"
        filepath = self.data_dir / filename

        # Zaten indirilmiş mi kontrol et
        if filepath.exists():
            print(f"   ⏭️  Zaten mevcut: {filename}")
            return str(filepath)

        print(f"   📡 İndiriliyor: {year}-{month:02d}")

        try:
            self.client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': variable,
                    'year': str(year),
                    'month': f'{month:02d}',
                    'day': [f'{d:02d}' for d in range(1, 32)],  # Tüm günler
                    'time': [
                        '00:00', '03:00', '06:00', '09:00',
                        '12:00', '15:00', '18:00', '21:00'
                    ],  # 3 saatlik aralıklar
                    'area': area,  # Türkiye sınırları
                    'format': 'netcdf',  # NetCDF formatı
                },
                str(filepath)
            )

            print(f"   ✅ İndirildi: {filename}")
            return str(filepath)

        except Exception as e:
            print(f"   ❌ Hata: {e}")
            return None

    def fetch_yearly_data(self, year, variable='2m_temperature'):
        """
        Bir yılın tüm aylarını çeker

        Args:
            year: Yıl
            variable: Değişken

        Returns:
            list: İndirilen dosya yolları
        """
        print(f"\n📅 {year} yılı verileri çekiliyor...")

        files = []
        for month in range(1, 13):
            filepath = self.fetch_turkey_temperature(year, month, variable)
            if filepath:
                files.append(filepath)

        return files

    def fetch_historical_data(self, start_year=1990, end_year=None, variable='2m_temperature'):
        """
        Tarihsel verileri çeker (1990'dan bugüne)

        Args:
            start_year: Başlangıç yılı
            end_year: Bitiş yılı (None ise şu anki yıl)
            variable: Değişken

        Returns:
            list: Tüm indirilen dosyalar
        """
        if end_year is None:
            end_year = datetime.now().year

        print("=" * 70)
        print("🌡️  ERA5 SICAKLIK VERİSİ ÇEKME")
        print("=" * 70)
        print(f"\n⚙️  Ayarlar:")
        print(f"  - Başlangıç yılı: {start_year}")
        print(f"  - Bitiş yılı: {end_year}")
        print(f"  - Değişken: {variable}")
        print(f"  - Bölge: Türkiye (36-42°N, 26-45°E)")
        print(f"  - Zaman aralığı: 3 saatlik (8 ölçüm/gün)")
        print(f"  - Format: NetCDF")

        total_months = (end_year - start_year + 1) * 12
        print(f"\n📊 Toplam: {total_months} ay verisi çekilecek")
        print(f"⚠️  Bu işlem birkaç SAAT sürebilir!")
        print(f"💾 Dosyalar: {self.data_dir}")

        proceed = input("\nDevam etmek istiyor musunuz? (e/h): ")

        if proceed.lower() != 'e':
            print("❌ İşlem iptal edildi")
            return []

        all_files = []

        for year in range(start_year, end_year + 1):
            files = self.fetch_yearly_data(year, variable)
            all_files.extend(files)

        print("\n" + "=" * 70)
        print("✅ VERİ ÇEKME TAMAMLANDI!")
        print("=" * 70)
        print(f"\n📊 Toplam: {len(all_files)} dosya indirildi")
        print(f"💾 Konum: {self.data_dir}")

        return all_files


def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print("🌍 ERA5 SICAKLIK VERİSİ ÇEKME ARACI")
    print("=" * 70)

    try:
        fetcher = ERA5TemperatureFetcher()

        # Test: Tek bir ay çek
        print("\n📅 Test: 2024 Ocak ayı verisi çekiliyor...\n")
        test_file = fetcher.fetch_turkey_temperature(2024, 1)

        if test_file:
            print(f"\n✅ Test başarılı! Dosya: {test_file}")
            print("\n💡 Tüm tarihsel verileri çekmek için:")
            print("   Script içindeki yorumdan çıkarın veya doğrudan fetch_historical_data() çağırın")

            # TARIHSEL VERİ ÇEKME (yorumdan çıkarın)
            # print("\n" + "="*70)
            # fetcher.fetch_historical_data(start_year=1990, end_year=2025)

        else:
            print("\n❌ Test başarısız!")
            print("⚠️  Lütfen CDS API kurulumunuzu kontrol edin:")
            print("   python3 scripts/setup_cds_api.py")

    except Exception as e:
        print(f"\n❌ Hata: {e}")
        print("\n⚠️  CDS API kurulumu yapılmamış olabilir.")
        print("   Önce şu scripti çalıştırın:")
        print("   python3 scripts/setup_cds_api.py")


if __name__ == "__main__":
    main()
