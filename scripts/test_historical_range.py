#!/usr/bin/env python3
"""
AFAD API'nin ne kadar geriye gittiğini test eden script
"""

import requests
from datetime import datetime, timedelta

def test_year(year):
    """Belirli bir yılda veri var mı test eder"""
    url = "https://servisnet.afad.gov.tr/apigateway/deprem/apiv2/event/filter"

    start = f"{year}-01-01T00:00:00"
    end = f"{year}-12-31T23:59:59"

    params = {
        "start": start,
        "end": end,
        "minmag": 0.0
    }

    try:
        print(f"📅 Test ediliyor: {year} yılı...", end=" ")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict) and 'data' in data:
            count = len(data['data'])
        elif isinstance(data, list):
            count = len(data)
        else:
            count = 0

        if count > 0:
            print(f"✅ {count} kayıt bulundu")
            return True, count
        else:
            print(f"❌ Veri yok")
            return False, 0

    except Exception as e:
        print(f"❌ Hata: {e}")
        return False, 0

def find_earliest_year():
    """En erken hangi yıla kadar gidilebildiğini bulur"""
    print("=" * 60)
    print("🔍 AFAD TARİHSEL VERİ ARAŞTIRMASI")
    print("=" * 60)
    print("\nEn erken veri tarihini arıyoruz...\n")

    # Önce son yıl test et
    current_year = datetime.now().year
    success, count = test_year(current_year)

    if not success:
        print("\n❌ Güncel veri bile çekilemedi, API sorunu olabilir")
        return None

    # Binary search ile en erken yılı bul
    earliest_found = current_year
    min_year = 1900
    max_year = current_year

    print("\n🔎 Binary search ile en erken yıl aranıyor...\n")

    # Önce bazı key yılları test et
    test_years = [2020, 2015, 2010, 2005, 2000, 1995, 1990]

    for year in test_years:
        success, count = test_year(year)
        if success:
            earliest_found = year
        else:
            # Bu yıldan önce veri yok, arama alanını daralt
            break

    print(f"\n📊 Sonuç:")
    print(f"  - En erken bulunan yıl: {earliest_found}")
    print(f"  - Mevcut toplam yıl aralığı: {current_year - earliest_found + 1} yıl")

    return earliest_found

if __name__ == "__main__":
    earliest = find_earliest_year()

    if earliest:
        print(f"\n✅ AFAD verileri {earliest} yılından itibaren mevcut")
        print(f"\n💡 Öneri: {earliest} yılından bugüne kadar olan tüm verileri çekebiliriz")
