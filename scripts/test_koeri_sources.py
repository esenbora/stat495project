#!/usr/bin/env python3
"""
KOERI veri kaynaklarını test eder
"""

import requests
from datetime import datetime, timedelta

def test_koeri_web():
    """KOERI web sayfasından veri çekmeyi test eder"""
    print("=" * 60)
    print("1. KOERI Web Sayfası Testi")
    print("=" * 60)

    urls = [
        "http://www.koeri.boun.edu.tr/scripts/lst0.asp",  # Son depremler
        "http://www.koeri.boun.edu.tr/scripts/lst1.asp",  # Son 500 deprem
        "http://www.koeri.boun.edu.tr/scripts/sondepremler.asp",  # Alternatif
    ]

    for url in urls:
        try:
            print(f"\n📡 Test: {url}")
            response = requests.get(url, timeout=10)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                content = response.text
                print(f"   İçerik boyutu: {len(content)} karakter")

                # Deprem verisi var mı kontrol et
                if "Tarih" in content or "Date" in content or "Magnitude" in content:
                    print(f"   ✅ Deprem verisi bulundu!")

                    # İlk birkaç satırı göster
                    lines = content.split('\n')[:10]
                    print(f"   İlk satırlar:")
                    for line in lines[:5]:
                        if line.strip():
                            print(f"     {line.strip()[:80]}")
                    return url, content

        except Exception as e:
            print(f"   ❌ Hata: {e}")

    return None, None

def test_fdsn_event():
    """FDSN event servisini test eder (varsa)"""
    print("\n" + "=" * 60)
    print("2. FDSN Event Servisi Testi")
    print("=" * 60)

    # Olası FDSN event endpoint'leri
    base_urls = [
        "http://eida-service.koeri.boun.edu.tr/fdsnws/event/1",
        "http://www.koeri.boun.edu.tr/fdsnws/event/1",
        "http://eida.koeri.boun.edu.tr/fdsnws/event/1",
    ]

    # Son 7 günün verileri
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)

    for base_url in base_urls:
        try:
            url = f"{base_url}/query"
            params = {
                'starttime': start_time.strftime('%Y-%m-%d'),
                'endtime': end_time.strftime('%Y-%m-%d'),
                'minlatitude': 36,
                'maxlatitude': 42,
                'minlongitude': 26,
                'maxlongitude': 45,
                'format': 'text'
            }

            print(f"\n📡 Test: {base_url}")
            response = requests.get(url, params=params, timeout=10)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                print(f"   ✅ FDSN event servisi aktif!")
                print(f"   İçerik boyutu: {len(response.text)} karakter")
                lines = response.text.split('\n')[:5]
                print(f"   İlk satırlar:")
                for line in lines:
                    if line.strip():
                        print(f"     {line.strip()}")
                return base_url

        except Exception as e:
            print(f"   ❌ Hata: {e}")

    return None

def test_third_party_apis():
    """Üçüncü parti KOERI API'lerini test eder"""
    print("\n" + "=" * 60)
    print("3. Üçüncü Parti API Testleri")
    print("=" * 60)

    apis = [
        {
            'name': 'Deprem API (GitHub)',
            'url': 'https://api.orhanaydogdu.com.tr/deprem/kandilli/live'
        },
        {
            'name': 'Alternatif KOERI API',
            'url': 'https://deprem.afad.gov.tr/apiv2/event/filter'  # AFAD zaten test ettik
        }
    ]

    for api in apis:
        try:
            print(f"\n📡 Test: {api['name']}")
            print(f"   URL: {api['url']}")
            response = requests.get(api['url'], timeout=10)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ JSON veri alındı!")

                    if isinstance(data, dict):
                        print(f"   Keys: {list(data.keys())}")
                        if 'result' in data:
                            print(f"   Deprem sayısı: {len(data['result'])}")
                    elif isinstance(data, list):
                        print(f"   Deprem sayısı: {len(data)}")

                except:
                    print(f"   İçerik (text): {len(response.text)} karakter")

        except Exception as e:
            print(f"   ❌ Hata: {e}")

if __name__ == "__main__":
    print("\n🔍 KOERI VERİ KAYNAKLARI ARAŞTIRMASI\n")

    # Test 1: KOERI web sayfası
    url, content = test_koeri_web()

    # Test 2: FDSN event servisi
    fdsn_url = test_fdsn_event()

    # Test 3: Üçüncü parti API'ler
    test_third_party_apis()

    print("\n" + "=" * 60)
    print("📊 SONUÇ")
    print("=" * 60)

    if url:
        print(f"\n✅ KOERI web sayfası erişilebilir: {url}")
    if fdsn_url:
        print(f"✅ FDSN event servisi bulundu: {fdsn_url}")

    print("\n💡 Öneriler:")
    if url:
        print("  1. KOERI web sayfasını parse edebiliriz")
    if fdsn_url:
        print("  2. FDSN event servisini kullanabiliriz")
    print("  3. Üçüncü parti API kullanabiliriz")
