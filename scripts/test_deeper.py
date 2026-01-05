#!/usr/bin/env python3
"""
Daha geriye gidilebilir mi test eder
"""

import requests
from datetime import datetime

def test_year(year):
    """Belirli bir yılda veri var mı test eder"""
    url = "https://servisnet.afad.gov.tr/apigateway/deprem/apiv2/event/filter"
    start = f"{year}-01-01T00:00:00"
    end = f"{year}-12-31T23:59:59"
    params = {"start": start, "end": end, "minmag": 0.0}

    try:
        print(f"📅 {year}: ", end="", flush=True)
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
            print(f"✅ {count} kayıt")
            return True, count
        else:
            print(f"❌")
            return False, 0
    except Exception as e:
        print(f"❌ Hata")
        return False, 0

print("🔍 1990'dan öncesine bakılıyor...\n")

for year in range(1989, 1899, -10):
    success, count = test_year(year)
    if not success:
        print(f"\n❌ {year} ve öncesinde veri bulunamadı")
        break
