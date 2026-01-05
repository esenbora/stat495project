#!/usr/bin/env python3
"""
Copernicus CDS API Kurulum Yardımcısı

Bu script CDS API'yi kurar ve konfigüre eder.
"""

import os
from pathlib import Path

def setup_cds_api():
    """CDS API kurulum talimatlarını gösterir"""

    print("=" * 70)
    print("🌍 COPERNICUS CDS API KURULUM REHBERİ")
    print("=" * 70)

    print("\n📋 ADIM 1: CDS API Paketini Kur")
    print("   Komutu çalıştırın:")
    print("   pip install cdsapi")

    print("\n📋 ADIM 2: Copernicus Hesabı Oluşturun")
    print("   1. https://cds.climate.copernicus.eu/ adresine gidin")
    print("   2. 'Register' butonuna tıklayın ve hesap oluşturun")
    print("   3. Email'inizi doğrulayın")

    print("\n📋 ADIM 3: API Key Alın")
    print("   1. https://cds.climate.copernicus.eu/how-to-api adresine gidin")
    print("   2. Sayfanın altında 'UID' ve 'API key' bilgilerinizi göreceksiniz")
    print("   3. Bu bilgileri kopyalayın")

    print("\n📋 ADIM 4: .cdsapirc Dosyası Oluşturun")

    cdsapirc_path = Path.home() / ".cdsapirc"

    print(f"   Dosya yolu: {cdsapirc_path}")

    if cdsapirc_path.exists():
        print("   ✅ .cdsapirc dosyası zaten mevcut!")
        with open(cdsapirc_path, 'r') as f:
            content = f.read()
            if 'url' in content and 'key' in content:
                print("   ✅ Dosya içeriği doğru görünüyor")
            else:
                print("   ⚠️  Dosya içeriği eksik olabilir")
    else:
        print("   ❌ .cdsapirc dosyası bulunamadı")
        print("\n   Şu içerikte bir dosya oluşturun:")
        print("   " + "-" * 60)
        print("   url: https://cds.climate.copernicus.eu/api")
        print("   key: UID:API-KEY")
        print("   " + "-" * 60)
        print("\n   UID ve API-KEY yerine kendi bilgilerinizi yazın!")

        create = input("\n   Şimdi oluşturmak ister misiniz? (e/h): ")

        if create.lower() == 'e':
            uid = input("   UID'nizi girin: ").strip()
            api_key = input("   API Key'inizi girin: ").strip()

            if uid and api_key:
                with open(cdsapirc_path, 'w') as f:
                    f.write("url: https://cds.climate.copernicus.eu/api\n")
                    f.write(f"key: {uid}:{api_key}\n")

                # Unix sistemlerde dosya izinlerini ayarla
                if os.name != 'nt':
                    os.chmod(cdsapirc_path, 0o600)

                print(f"   ✅ .cdsapirc dosyası oluşturuldu: {cdsapirc_path}")
            else:
                print("   ❌ UID veya API Key boş bırakılamaz!")

    print("\n📋 ADIM 5: Terms of Use'u Kabul Edin")
    print("   1. https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels")
    print("   2. Sayfanın altındaki 'Download data' sekmesine gidin")
    print("   3. 'Terms of use' linkine tıklayın ve kabul edin")
    print("   ⚠️  BU ADIM ÇOK ÖNEMLİ! Kabul etmeden API çalışmaz!")

    print("\n📋 ADIM 6: Test Edin")
    print("   Test scriptini çalıştırın:")
    print("   python3 scripts/test_cds_api.py")

    print("\n" + "=" * 70)
    print("✅ Kurulum talimatları tamamlandı!")
    print("=" * 70)

if __name__ == "__main__":
    setup_cds_api()
