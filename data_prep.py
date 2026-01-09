import pandas as pd
import numpy as np

# Veriyi yükle
df = pd.read_csv(r"C:\Users\elifi\OneDrive\Masaüstü\financalMarketing\BankChurners.csv")

# 1. Gereksiz (Çöp) Sütunların Temizlenmesi
# Veri setinin sonunda "Naive_Bayes..." ile başlayan ve modelleme için gereksiz olan iki sütun var.
cols_to_drop = [
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 2. Sütun İsimlerinin Standartlaştırılması (SQL ve Python uyumu için)
# Tüm sütun isimlerini küçük harfe çevirip varsa boşlukları temizleyelim.
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 3. Hedef Değişkenin (Target) Hazırlanması - Makine Öğrenmesi İçin
# 'attrition_flag' sütunu string (Existing/Attrited). Bunu 0 ve 1'e çevirelim.
# Attrited Customer (Terk Eden) = 1
# Existing Customer (Mevcut) = 0
df['churn_label'] = df['attrition_flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# 4. Veri Tiplerinin Kontrolü ve Dönüşümü
# Gereksiz boşlukları string sütunlardan temizle
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].str.strip()

# 5. Kontrol: Eksik Veri Var mı?
null_counts = df.isnull().sum().sum()

# Temizlenmiş veriyi kaydet
output_filename = 'BankChurners_Cleaned.csv'
df.to_csv(output_filename, index=False)

print(f"Veri temizlendi ve '{output_filename}' olarak kaydedildi.")
print(f"Toplam Satır: {df.shape[0]}")
print(f"Toplam Sütun: {df.shape[1]}")
print(f"Toplam Eksik Değer (Null): {null_counts}")
print("\nSütun İsimleri (Yeni Hali):")
print(list(df.columns))
print("\nİlk 5 Satır:")
print(df.head())