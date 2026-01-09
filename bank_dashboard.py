import streamlit as st
import pandas as pd
import pyodbc
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ------------------------------------------------------------------------------------------------
# 1. AYARLAR VE VERİTABANI BAĞLANTISI
# ------------------------------------------------------------------------------------------------
st.set_page_config(page_title="MiuulBank Analytics", page_icon="🏦", layout="wide")

# --- BURAYI KENDİ BİLGİSAYAR ADINLA DEĞİŞTİR ---
SERVER_NAME = 'LAPTOP-VECCGU4R\SQLEXPRESS'
DATABASE_NAME = 'miuulProje'
TABLE_NAME = 'BankChurners_Cleaned'


def get_db_connection():
    """MS SQL Veritabanına güvenli bağlantı kurar."""
    try:
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={SERVER_NAME};'
            f'DATABASE={DATABASE_NAME};'
            'Trusted_Connection=yes;'
        )
        return conn
    except Exception as e:
        st.error(f"❌ Veritabanı Bağlantı Hatası: {e}")
        st.stop()


def run_query(query):
    """SQL sorgusu çalıştırır ve Pandas DataFrame döner."""
    conn = get_db_connection()
    if conn:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    return None


# ------------------------------------------------------------------------------------------------
# 2. MAKİNE ÖĞRENMESİ
# ------------------------------------------------------------------------------------------------
@st.cache_resource
def train_model():
    query = f"SELECT * FROM {TABLE_NAME}"
    df = run_query(query)

    leakage_cols = [c for c in df.columns if 'Naive_Bayes' in c]
    ignore_cols = ['clientnum', 'attrition_flag', 'churn_label'] + leakage_cols
    X = df.drop([c for c in ignore_cols if c in df.columns], axis=1)
    y = df['churn_label']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X.columns, accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test,
                                                                                                           y_pred)


# ------------------------------------------------------------------------------------------------
# 3. SAYFA TASARIMI
# ------------------------------------------------------------------------------------------------
def main():
    st.sidebar.title("🏦 MiuulBank Panel")
    menu = st.sidebar.radio("İşlemler", ["Ana Sayfa (KPI)", "Müşteri Analizi (SQL)", "Yapay Zeka Tahmini"])
    st.sidebar.success("Veritabanı Bağlantısı: Aktif ✅")

    # --- MENÜ 1: ANA SAYFA (KPI) ---
    if menu == "Ana Sayfa (KPI)":
        st.title("📊 Yönetici Özeti")
        st.markdown("Veritabanından anlık çekilen banka performans metrikleri.")

        # KPI Sorgusu
        df_kpi = run_query(
            f"SELECT COUNT(*) as TotalCust, SUM(CASE WHEN churn_label = 1 THEN 1 ELSE 0 END) as ChurnCount, AVG(credit_limit) as AvgLimit, AVG(total_trans_amt) as AvgSpend FROM {TABLE_NAME}")

        col1, col2, col3, col4 = st.columns(4)
        total = df_kpi['TotalCust'][0]
        churn = df_kpi['ChurnCount'][0]
        col1.metric("Toplam Müşteri", f"{total:,}")
        col2.metric("Kayıp Oranı", f"%{(churn / total) * 100:.1f}", delta_color="inverse")
        col3.metric("Ort. Kredi Limiti", f"${df_kpi['AvgLimit'][0]:,.0f}")
        col4.metric("Ort. Harcama", f"${df_kpi['AvgSpend'][0]:,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Eğitim Seviyesine Göre Dağılım")
            df_edu = run_query(f"SELECT education_level, COUNT(*) as Count FROM {TABLE_NAME} GROUP BY education_level")
            st.plotly_chart(px.pie(df_edu, names='education_level', values='Count', hole=0.4), use_container_width=True)
        with c2:
            st.subheader("Kart Tipi Bazında Ort. Harcama")
            df_card = run_query(
                f"SELECT card_category, AVG(total_trans_amt) as AvgSpend FROM {TABLE_NAME} GROUP BY card_category")
            st.plotly_chart(px.bar(df_card, x='card_category', y='AvgSpend', color='card_category'),
                            use_container_width=True)

    # --- MENÜ 2: MÜŞTERİ ANALİZİ (SQL) ---
    elif menu == "Müşteri Analizi (SQL)":
        st.title("🔍 Stratejik Raporlama Merkezi")
        st.markdown("Veriye dayalı karar alma mekanizması için hazırlanmış SQL senaryoları.")

        report_type = st.radio("Rapor Kategorisi:",
                               ["🚨 Erken Uyarı & Risk", "💎 Satış & Büyüme Fırsatları", "📉 Operasyonel Analiz"],
                               horizontal=True)

        scenarios = {}

        if report_type == "🚨 Erken Uyarı & Risk":
            scenarios = {
                "Sessiz Kayıplar (Pasifleşen VIP'ler)": {
                    "desc": "Kredi limiti 10.000$ üstü olup son 3 aydır inaktif olan müşteriler. Terk etme riski çok yüksektir.",
                    "query": f"SELECT clientnum, customer_age, credit_limit, months_inactive_12_mon, total_trans_amt FROM {TABLE_NAME} WHERE credit_limit > 10000 AND months_inactive_12_mon >= 3 ORDER BY credit_limit DESC"
                },
                "Limitini Zorlayanlar (Borç Riski)": {
                    "desc": "Limit kullanım oranı %90'ın üzerinde olan ve henüz Churn olmamış müşteriler. Ödeme güçlüğü çekebilirler.",
                    "query": f"SELECT clientnum, credit_limit, total_revolving_bal, avg_utilization_ratio FROM {TABLE_NAME} WHERE avg_utilization_ratio > 0.90 AND churn_label = 0 ORDER BY total_revolving_bal DESC"
                },
                "Ani İşlem Kesintisi": {
                    "desc": "Geçmişte çok aktif olup (60+ işlem), son dönemde işlem sayısı %50'den fazla düşen sadık müşteriler.",
                    "query": f"SELECT clientnum, total_trans_ct, total_ct_chng_q4_q1, total_trans_amt FROM {TABLE_NAME} WHERE total_trans_ct > 60 AND total_ct_chng_q4_q1 < 0.5 ORDER BY total_ct_chng_q4_q1 ASC"
                },
                "Riskli Aktif Müşteriler": {
                    "desc": "Şu an bankayı terk etmiş (Churn=1) ancak geçmiş harcaması yüksek olan, geri kazanılması gereken müşteriler.",
                    "query": f"SELECT TOP 200 clientnum, customer_age, total_trans_amt, avg_utilization_ratio FROM {TABLE_NAME} WHERE churn_label = 1 ORDER BY total_trans_amt DESC"
                }
            }

        elif report_type == "💎 Satış & Büyüme Fırsatları":
            scenarios = {
                "Uyuyan Devler (Yüksek Gelir)": {
                    "desc": "Geliri 80K$+ olup kartını neredeyse hiç kullanmayanlar. Özel kampanya ile uyandırılmalılar.",
                    "query": f"SELECT clientnum, income_category, credit_limit, avg_utilization_ratio FROM {TABLE_NAME} WHERE (income_category = '$80K - $120K' OR income_category = '$120K +') AND avg_utilization_ratio < 0.10 ORDER BY credit_limit DESC"
                },
                "Limit Artış Adayları": {
                    "desc": "Limiti 5000$ altı ama limiti dolmak üzere olan sadık müşteriler. Limit artışı teklifi için uygundur.",
                    "query": f"SELECT clientnum, credit_limit, avg_utilization_ratio, total_trans_amt FROM {TABLE_NAME} WHERE credit_limit < 5000 AND avg_utilization_ratio > 0.70 AND churn_label = 0 ORDER BY avg_utilization_ratio DESC"
                },
                "Platinum Kart Adayları": {
                    "desc": "Blue kart sahibi olup 10.000$ üzeri harcama yapan müşteriler. Kart yükseltme (Upgrade) teklifi verilmeli.",
                    "query": f"SELECT clientnum, card_category, total_trans_amt, total_trans_ct FROM {TABLE_NAME} WHERE card_category = 'Blue' AND total_trans_amt > 10000 ORDER BY total_trans_amt DESC"
                },
                "Genç Profesyoneller": {
                    "desc": "30 yaş altı ve yüksek eğitimli (Doktora/Yüksek Lisans) potansiyel müşteriler.",
                    "query": f"SELECT clientnum, customer_age, education_level, income_category FROM {TABLE_NAME} WHERE customer_age < 30 AND (education_level = 'Graduate' OR education_level = 'Doctorate') ORDER BY customer_age ASC"
                }
            }

        elif report_type == "📉 Operasyonel Analiz":
            scenarios = {
                "Eğitim Seviyesi & Harcama": {
                    "desc": "Hangi eğitim seviyesindeki müşteriler bankaya daha çok kazandırıyor?",
                    "query": f"SELECT education_level, COUNT(*) as Müşteri, AVG(total_trans_amt) as Ort_Harcama FROM {TABLE_NAME} GROUP BY education_level ORDER BY Ort_Harcama DESC"
                },
                "En Sadık Müşteriler (4+ Yıl)": {
                    "desc": "48 aydan uzun süredir bankada olan ve hala aktif en değerli müşteriler.",
                    "query": f"SELECT TOP 500 clientnum, months_on_book, total_relationship_count, total_trans_amt FROM {TABLE_NAME} WHERE months_on_book >= 48 AND churn_label = 0 ORDER BY total_trans_amt DESC"
                },
                "Kart Karlılık Analizi": {
                    "desc": "Kart tiplerine göre (Blue, Silver vb.) toplam işlem hacmi karşılaştırması.",
                    "query": f"SELECT card_category, SUM(total_trans_amt) as Toplam_Hacim, AVG(avg_utilization_ratio) as Ort_Kullanım FROM {TABLE_NAME} GROUP BY card_category ORDER BY Toplam_Hacim DESC"
                }
            }

        # --- (Placeholder Tekniği) ---

        # 1. Önce açıklama için BOŞ BİR ALAN (Yer Tutucu) yaratıyoruz.
        description_placeholder = st.empty()

        # 2. Seçim kutusunu oluşturuyoruz (Key: Benzersiz kimlik)
        selected_option = st.selectbox("Senaryo Seçiniz:", list(scenarios.keys()), key=f"sb_{report_type}")

        # 3. Verileri çekiyoruz
        selected_data = scenarios[selected_option]
        description = selected_data["desc"]
        sql_query = selected_data["query"]

        # 4. Yer tutucunun içini dolduruyoruz (Bu yöntem ekranı zorla günceller)
        description_placeholder.info(f"💡 **Senaryo Açıklaması:** {description}")

        with st.expander("🛠️ SQL Sorgusunu Görüntüle"):
            st.code(sql_query, language="sql")

        if st.button("Raporu Oluştur ve Getir", type="primary"):
            df_report = run_query(sql_query)
            if df_report is not None and not df_report.empty:
                st.success(f"✅ Toplam {len(df_report)} kayıt bulundu.")
                st.dataframe(df_report, use_container_width=True)
                csv = df_report.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Excel İndir", csv, "rapor.csv", "text/csv")
            else:
                st.warning("⚠️ Kayıt bulunamadı.")

    # --- MENÜ 3: YAPAY ZEKA TAHMİNİ ---
    elif menu == "Yapay Zeka Tahmini":
        st.title("🤖 Gelişmiş Risk Analizi & Karar Destek")
        st.markdown("---")

        model, model_cols, acc, prec, rec = train_model()

        st.sidebar.header("👤 Müşteri Profili Simülasyonu")

        with st.sidebar.form("prediction_form"):
            st.sidebar.subheader("1. Mevcut Durum")
            age = st.sidebar.slider("Yaş", 20, 80, 46)
            trans_amt = st.sidebar.number_input("Yıllık Harcama ($)", 0, 30000, 4000)
            trans_ct = st.sidebar.number_input("İşlem Adedi", 0, 150, 60)

            st.sidebar.subheader("2. Finansal Durum")
            inactive = st.sidebar.slider("İnaktif Olduğu Ay Sayısı", 0, 12, 2)
            revolving = st.sidebar.number_input("Dönen Bakiye (Borç)", 0, 5000, 1000)
            limit = st.sidebar.number_input("Kredi Limiti", 1000, 40000, 8000)

            submitted = st.form_submit_button("Risk Analizini Çalıştır", type="primary")

        if submitted:
            utilization = revolving / limit if limit > 0 else 0
            default_change = 0.0 if trans_ct == 0 else 0.7

            input_df = pd.DataFrame({
                'customer_age': [age],
                'total_trans_amt': [trans_amt],
                'total_trans_ct': [trans_ct],
                'months_inactive_12_mon': [inactive],
                'total_revolving_bal': [revolving],
                'credit_limit': [limit],
                'avg_utilization_ratio': [utilization],
                'total_ct_chng_q4_q1': [default_change],
                'total_amt_chng_q4_q1': [default_change],
                'avg_open_to_buy': [limit - revolving],
                'dependent_count': [2],
                'months_on_book': [36],
                'total_relationship_count': [3],
                'contacts_count_12_mon': [3 if inactive > 2 else 2]
            })

            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=model_cols, fill_value=0)

            prob_risk = model.predict_proba(input_encoded)[0][1]
            risk_score = int(prob_risk * 100)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Tahmin Sonucu")
                if risk_score > 50:
                    st.error(f"🚨 RİSKLİ MÜŞTERİ")
                    st.metric("Terk Etme İhtimali", f"%{risk_score}", delta="Yüksek Risk", delta_color="inverse")
                    if inactive > 3:
                        st.warning("⚠️ Müşteri uzun süredir pasif durumda.")
                    if trans_ct == 0:
                        st.warning("⚠️ Hiç işlem yapmıyor.")
                else:
                    st.success(f"✅ SADIK MÜŞTERİ")
                    st.metric("Sadakat Skoru", f"%{100 - risk_score}", delta="Güvenli", delta_color="normal")

            with col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=risk_score, title={'text': "Risk Endeksi"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkred" if risk_score > 50 else "green"}}
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("---")

            st.header("📊 Model Performansı")
            tab1, tab2, tab3 = st.tabs(["🧠 Özellik Önemi", "📈 Gerçek Metrikler", "🔥 Korelasyonlar"])

            with tab1:
                importances = model.feature_importances_
                feature_imp_df = pd.DataFrame({'Feature': model_cols, 'Importance': importances}).sort_values(
                    by='Importance', ascending=False).head(10)
                fig_imp = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h',
                                 title="Model Kararını Etkileyen Faktörler")
                st.plotly_chart(fig_imp, use_container_width=True)

            with tab2:
                st.write("Modelin test verisi üzerindeki **gerçek** performans skorları:")
                c1, c2, c3 = st.columns(3)
                c1.metric("Doğruluk (Accuracy)", f"%{acc * 100:.1f}")
                c2.metric("Kesinlik (Precision)", f"%{prec * 100:.1f}")
                c3.metric("Duyarlılık (Recall)", f"%{rec * 100:.1f}")
                st.info("Bu değerler veritabanındaki verilerle anlık hesaplanmıştır.")

            with tab3:
                conn = get_db_connection()
                if conn:
                    df_corr = pd.read_sql(f"SELECT TOP 1000 * FROM {TABLE_NAME}", conn)
                    numeric_df = df_corr.select_dtypes(include=[np.number])
                    fig_corr = px.imshow(numeric_df.corr(), text_auto=False, aspect="auto",
                                         title="Korelasyon Haritası")
                    st.plotly_chart(fig_corr, use_container_width=True)
                    conn.close()


if __name__ == "__main__":
    main()