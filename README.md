# Deprem Araştırma Projesi

AFAD ve KOERI verilerini kullanarak deprem analizi yapan research projesi.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### AFAD Verilerini Çekme

```bash
cd scripts
python fetch_afad_data.py
```

Script iki mod ile çalışır:

1. **Test Modu**: Son 30 günün verileri (varsayılan)
2. **Tarihsel Mod**: Belirlediğiniz yıldan bugüne tüm veriler

### KOERI Verilerini Çekme

```bash
python scripts/fetch_koeri_data.py
```

Son 500 depremi çeker (son ~3 gün).

### ERA5 Sıcaklık Verilerini Çekme

**Adım 1: CDS API Kurulumu**

```bash
python scripts/setup_cds_api.py
```

Bu script size adım adım kurulum talimatları verecektir:
1. Copernicus hesabı oluşturma
2. API key alma
3. `.cdsapirc` dosyası oluşturma
4. Terms of Use kabul etme

**Adım 2: Veri Çekme**

```bash
python scripts/fetch_era5_temperature.py
```

Test için 2024 Ocak ayı verisi çeker. Tüm tarihsel verileri çekmek için script içindeki yorumdan çıkarın.

⚠️ **Önemli**:
- ERA5 veri çekme işlemi birkaç SAAT sürebilir
- Her ay için ayrı NetCDF dosyası oluşturulur
- Türkiye bölgesi (36-42°N, 26-45°E) için **3 saatlik** sıcaklık verileri (8 ölçüm/gün)

### Tarihsel Veri Çekme (AFAD)

`fetch_afad_data.py` dosyasında ilgili satırları yorumdan çıkarın:

```python
# Örnek: 2020'den bugüne, minimum 2.0 magnitude
df_historical = fetcher.fetch_historical_data(start_year=2020, min_magnitude=2.0, chunk_months=3)
if not df_historical.empty:
    fetcher.save_data(df_historical, "afad_historical_2020_present.csv")
```

## Proje Yapısı

```
stat495project/
├── data/
│   ├── raw/          # Ham veriler
│   └── processed/    # İşlenmiş veriler
├── scripts/
│   └── fetch_afad_data.py
├── requirements.txt
└── README.md
```

## Veri Kaynakları

- **AFAD**: Türkiye resmi deprem verileri (1990-2025, 537K+ kayıt)
- **KOERI**: Kandilli Rasathanesi verileri (son 500 deprem)
- **ERA5**: Copernicus iklim reanalysis sıcaklık verileri (1990-2025)

## Özellikler

### AFAD
- ✅ AFAD v2 API desteği
- ✅ Eski API fallback
- ✅ Pagination desteği
- ✅ 1990-2025 tarihsel veriler (537K+ kayıt)

### KOERI
- ✅ Web sayfası parser
- ✅ Son 500 deprem verisi
- ✅ Otomatik encoding tespiti

### ERA5
- ✅ Copernicus CDS API entegrasyonu
- ✅ Türkiye bölgesi sıcaklık verileri
- ✅ NetCDF format desteği
- ✅ 1990'dan bugüne tarihsel veriler

### Genel
- ✅ CSV ve JSON export
- ✅ Rate limiting
- ✅ Hata yönetimi
