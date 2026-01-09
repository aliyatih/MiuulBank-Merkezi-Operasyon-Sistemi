import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Pandas ayarları
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# --- DÜZELTME BURADA ---
# Dosya yolunu aşağıya tırnak içine yapıştır.
# Windows kullanıyorsan başına 'r' koy ki hata vermesin (Örn: r"C:\Users\Ali\Desktop\BankChurners.csv")
# Mac kullanıyorsan direkt yapıştır (Örn: "/Users/Ali/Desktop/BankChurners.csv")

dosya_yolu = r"C:\Users\Fatih\Downloads\BankChurners.csv"

try:
    df = pd.read_csv(dosya_yolu)
    print("Harika! Dosya başarıyla yüklendi.")

    # 2. Gereksiz (Naive Bayes sonucu olan) son 2 sütunu at
    df = df.iloc[:, :-2]

    # 3. Müşteri Kayıp Durumunu Sayısallaştır
    df["Churn"] = df["Attrition_Flag"].apply(lambda x: 1 if x == "Attrited Customer" else 0)

    print("Veri ön izlemesi:")
    print(df.head())

except FileNotFoundError:
    print("HATA: Dosya yolu hala yanlış görünüyor. Lütfen yolu tırnakların içine doğru yapıştırdığından emin ol.")
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")

###################################################################################################################

def ab_testing_gender_spend(dataframe):
    print("### A/B Testi: Cinsiyete Göre Harcama Farklılığı ###")

    # 1. Grupları Ayır
    group_m = dataframe[dataframe["Gender"] == "M"]["Total_Trans_Amt"]
    group_f = dataframe[dataframe["Gender"] == "F"]["Total_Trans_Amt"]

    # 2. Varsayım Kontrolü: Normallik (Shapiro)
    # p < 0.05 ise H0 Red (Normal Dağılmıyor) -> Non-Parametrik Test Yap
    stat_m, p_m = shapiro(group_m)
    stat_f, p_f = shapiro(group_f)

    print(f"Normallik Testi P-Values -> Male: {p_m:.4f}, Female: {p_f:.4f}")

    # 3. Hipotez Testi
    if (p_m > 0.05) and (p_f > 0.05):
        # Parametrik Test (T-Test)
        stat, p_value = ttest_ind(group_m, group_f, equal_var=True)
        test_type = "T-Test (Parametrik)"
    else:
        # Non-Parametrik Test (Mann-Whitney U)
        stat, p_value = mannwhitneyu(group_m, group_f)
        test_type = "Mann-Whitney U (Non-Parametrik)"

    print(f"Uygulanan Test: {test_type}")
    print(f"P-Value: {p_value:.4f}")

    if p_value < 0.05:
        print("SONUÇ: H0 Reddedilir. İki grup arasında ANLAMLI bir fark vardır.")
        print(f"Erkek Ortalaması: {group_m.mean():.2f}, Kadın Ortalaması: {group_f.mean():.2f}")
    else:
        print("SONUÇ: H0 Reddedilemez. Anlamlı bir fark yoktur.")


# Fonksiyonu çalıştır
ab_testing_gender_spend(df)


##################################################################################################################################33

def create_segments_kmeans(dataframe):
    # Segmentasyon için kullanılacak değişkenler
    rfm_df = dataframe[["Total_Trans_Amt", "Total_Trans_Ct"]]

    # 1. Standartlaştırma (0-1 arasına çekme)
    sc = MinMaxScaler((0, 1))
    rfm_scaled = sc.fit_transform(rfm_df)

    # 2. K-Means Modeli (Örneğin 4 Segment olsun)
    kmeans = KMeans(n_clusters=4, random_state=42)
    k_fit = kmeans.fit(rfm_scaled)

    # 3. Segmentleri Veriye İşle
    dataframe["Segment_No"] = k_fit.labels_

    # 4. Segmentleri İsimlendir (Ortalamalara bakarak karar verilir)
    # Segmentlerin istatistiklerine bakalım
    summary = dataframe.groupby("Segment_No")[["Total_Trans_Amt", "Total_Trans_Ct"]].mean()
    print("Segment İstatistikleri:")
    print(summary)

    # Örnek İsimlendirme (Çıktıya göre burayı güncelleyebilirsin)
    # Diyelim ki 3. grup en yüksek harcamayı yapanlar çıktı:
    segment_mapping = {
        0: "Bronz Müşteri",
        1: "Gümüş Müşteri",
        2: "Riskli Grup",
        3: "Altın (VIP) Müşteri"
    }
    # Not: Hangi numaranın hangi segment olduğunu yukarıdaki 'summary' çıktısına bakarak sen belirlemelisin.

    dataframe["Segment_Name"] = dataframe["Segment_No"].map(segment_mapping)

    return dataframe


# Fonksiyonu çalıştır
df_final = create_segments_kmeans(df)
print(df_final["Segment_Name"].value_counts())


################################## GORSELLEŞTİRME ##################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# --- AYARLAR ---
# Pandas ve Görünüm ayarları
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")
plt.rcParams.update({'figure.dpi': 150}) # Grafikler net çıksın diye


# --- 2. VERİ HAZIRLIĞI ---
# Gereksiz son 2 sütunu at (Naive Bayes çıktıları)
df = df.iloc[:, :-2]
# Churn durumunu 1-0 yap
df["Churn"] = df["Attrition_Flag"].apply(lambda x: 1 if x == "Attrited Customer" else 0)

# --- 3. SEGMENTASYON (GRAFİK İÇİN GEREKLİ) ---
# K-Means Segmentasyonu tekrar çalıştırıyoruz ki grafikleri çizebilelim
X = df[["Total_Trans_Amt", "Total_Trans_Ct"]]
sc = MinMaxScaler((0, 1))
X_scaled = sc.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df["Segment_No"] = kmeans.fit_predict(X_scaled)

# İsimlendirme
seg_summary = df.groupby("Segment_No")[["Total_Trans_Amt", "Total_Trans_Ct"]].mean()
seg_summary["Score"] = seg_summary["Total_Trans_Amt"] + seg_summary["Total_Trans_Ct"]
seg_summary = seg_summary.sort_values("Score")

label_map = {
    seg_summary.index[0]: "Bronze (Düşük)",
    seg_summary.index[1]: "Silver (Orta)",
    seg_summary.index[2]: "Gold (Yüksek)",
    seg_summary.index[3]: "Diamond (VIP)"
}
df["Segment_Name"] = df["Segment_No"].map(label_map)


# --- 4. GÖRSELLEŞTİRME ---

# GRAFİK 1: Segment Dağılımı (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Total_Trans_Ct", y="Total_Trans_Amt", hue="Segment_Name", data=df, palette="viridis", s=60, alpha=0.9)
plt.title("Müşteri Segmentasyonu", fontsize=14, fontweight='bold')
plt.xlabel("İşlem Adedi")
plt.ylabel("Harcama Tutarı")
plt.legend(title="Segment")
plt.tight_layout()
plt.show()

# GRAFİK 2: A/B Testi - Cinsiyete Göre Harcama (Boxplot)
plt.figure(figsize=(8, 6))
sns.boxplot(x="Gender", y="Total_Trans_Amt", data=df, palette="coolwarm")
plt.title("Cinsiyete Göre Harcama Dağılımı (A/B Testi)", fontsize=14, fontweight='bold')
plt.xlabel("Cinsiyet")
plt.ylabel("Toplam Harcama")
plt.tight_layout()
plt.show()

# GRAFİK 3: Segmentlerin Churn Oranları (Bar Plot)
churn_rates = df.groupby("Segment_Name")["Churn"].mean().reset_index().sort_values("Churn", ascending=False)
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x="Segment_Name", y="Churn", data=churn_rates, palette="Reds_r")
plt.title("Segmentlere Göre Müşteri Kayıp (Churn) Oranları", fontsize=14, fontweight='bold')
plt.xlabel("Segment")
plt.ylabel("Churn Oranı")
plt.ylim(0, 0.6)
# Barların üzerine oran yaz
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2%'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.tight_layout()
plt.show()

################################## GORSELLEŞTİRME 2 ##################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Stil Ayarları
sns.set(style="whitegrid")
plt.rcParams.update({'figure.dpi': 150}) # Yüksek çözünürlük

# 1. Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x="Gender", y="Total_Trans_Amt", data=df, palette="muted", inner="quartile")
plt.title("Dağılım Analizi: Cinsiyete Göre Harcama Yoğunluğu", fontsize=14, fontweight='bold')
plt.xlabel("Cinsiyet")
plt.ylabel("Harcama Tutarı")
plt.savefig("Olcumleme_1_Violin.png")
plt.show()

# 2. Güven Aralıklı Bar Plot
plt.figure(figsize=(8, 6))
# ci=95 parametresi %95 Güven Aralığını (Confidence Interval) çizer
ax = sns.barplot(x="Gender", y="Total_Trans_Amt", data=df, palette="pastel", errorbar=('ci', 95), capsize=0.1)
plt.title("Ortalama Harcama Karşılaştırması (%95 Güven Aralığı ile)", fontsize=14, fontweight='bold')
plt.xlabel("Cinsiyet")
plt.ylabel("Ortalama Harcama")
# Ortalamaları barın üzerine yaz
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')
plt.savefig("Olcumleme_2_GuvenAraligi.png")
plt.show()

# 3. Histogram (Overlaid)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Total_Trans_Amt", hue="Gender", kde=True, element="step", palette="dark", alpha=0.6)
plt.title("Harcama Frekans Dağılımı (Kadın vs Erkek)", fontsize=14, fontweight='bold')
plt.xlabel("Harcama Tutarı")
plt.ylabel("Müşteri Sayısı")
plt.savefig("Olcumleme_3_Histogram.png")
plt.show()