import io
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

# 1. ضبط إعدادات الصفحة
st.set_page_config(
    page_title="نظام التنبؤ الذكي بالمتأخرات الدراسية",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# دعم الاتجاه العربي (RTL) وتنسيق الخطوط
st.markdown(
    """
<style>
    /* تطبيق اتجاه النص العربي */
    html, body, [class*="css"], .stMarkdown, .stDataFrame {
        direction: rtl;
        text-align: right;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


# 2. بناء وتدريب نموذج الغابة العشوائية مع التخزين المؤقت
@st.cache_resource
def setup_rf_model():
    np.random.seed(42)
    n_samples = 1000

    family_income = np.random.randint(10000, 150000, n_samples)
    scholarship_ratio = np.random.uniform(0.0, 1.0, n_samples)
    previous_delays = np.random.randint(0, 6, n_samples)

    income_score = (150000 - family_income) / 140000
    scholarship_score = 1.0 - scholarship_ratio
    delays_score = previous_delays / 5.0

    combined_score = (
        (income_score * 0.4) + (scholarship_score * 0.3) + (delays_score * 0.3)
    )
    noise = np.random.normal(0, 0.08, n_samples)

    will_delay = np.where(combined_score + noise > 0.52, 1, 0)

    df = pd.DataFrame(
        {
            "family_income": family_income,
            "scholarship_ratio": scholarship_ratio * 100,
            "previous_delays": previous_delays,
            "will_delay": will_delay,
        }
    )

    features = ["family_income", "scholarship_ratio", "previous_delays"]
    X = df[features]
    y = df["will_delay"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=5
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    importances = model.feature_importances_

    return model, accuracy, df, features, importances


model, accuracy, df_train, features_list, importances = setup_rf_model()

# --- رأس الصفحة الرسمي ---
st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>نظام التنبؤ الذكي بالمتأخرات الدراسية 🎓</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center; color: #3B82F6;'>Smart Tuition Fees Risk & Prediction System</h4>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #555;'>نظام ريادي مدعوم بالذكاء الاصطناعي لاستشراف الأزمات المالية للطلاب ومساعدة الشؤون المالية بالكلية على اتخاذ قرارات استباقية.</p>",
    unsafe_allow_html=True,
)
st.write("---")

# --- تقسيم النظام إلى تبويبات رئيسية ---
tab1, tab2, tab3 = st.tabs(
    [
        "👤 فحص طالب فردي",
        "📁 فحص دفعة كاملة (Batch Upload)",
        "🛠️ لوحة التحليل الأكاديمي (لجنة المناقشة)",
    ]
)

# ==========================================
# 🟢 التبويب الأول: فحص طالب فردي
# ==========================================
with tab1:
    st.subheader("📝 إدخال البيانات التحليلية للطالب")
    student_name = st.text_input(
        "اسم الطالب بالكامل", value="أحمد محمد علي", key="single_name"
    )

    col1, col2 = st.columns(2)
    with col1:
        family_income_input = st.number_input(
            "الدخل الشهري لأسرة الطالب (ريال / جنيه)",
            min_value=0,
            value=45000,
            step=1000,
        )
        scholarship_ratio_input = st.slider(
            "نسبة المنحة الدراسية التي يتلقاها الطالب (%)",
            min_value=0,
            max_value=100,
            value=25,
        )

    with col2:
        prev_delays_input = st.slider(
            "سجل التأخيرات السابقة (عدد الفصول الماضية)",
            min_value=0,
            max_value=5,
            value=1,
        )
        st.info(
            "💡 **آلية العمل:** يتم دمج قدرة الأسرة الائتمانية مع سلوك الالتزام التاريخي عبر خوارزمية Random Forest."
        )

    if st.button(
        "تحليل وتوقع حالة الطالب 🔍", use_container_width=True, key="single_btn"
    ):
        input_data = pd.DataFrame(
            [
                {
                    "family_income": family_income_input,
                    "scholarship_ratio": scholarship_ratio_input,
                    "previous_delays": prev_delays_input,
                }
            ]
        )

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        delay_probability = probabilities[1] * 100
        commit_probability = probabilities[0] * 100

        st.write("---")
        st.subheader(f"🎯 تقرير الحالة المتوقعة للطالب: {student_name}")

        res_col1, res_col2 = st.columns(2)
        if prediction == 1:
            with res_col1:
                st.error("⚠️ **تنبيه:** هناك احتمالية مرتفعة لتعثر الطالب مالياً.")
                st.metric(
                    label="احتمالية التعثر المتوقعة",
                    value=f"{delay_probability:.1f}%",
                )
            with res_col2:
                st.markdown("""
                **📥 الإجراء الإداري الموصى به:**
                * التواصل مع ولي الأمر لتقديم خطة سداد ميسرة.
                * دراسة إمكانية إدراج الطالب في صندوق دعم الطلاب.
                """)
        else:
            with res_col1:
                st.success("✅ **حالة أمان مالي:** الطالب يظهر التزاماً وقدرة جيدة على السداد.")
                st.metric(
                    label="احتمالية السداد المنتظم",
                    value=f"{commit_probability:.1f}%",
                )
            with res_col2:
                st.markdown("""
                **📥 الإجراء الإداري الموصى به:**
                * لا يتطلب إجراء استباقي، حالة الطالب المالية مستقرة.
                """)

# ==========================================
# 🔵 التبويب الثاني: فحص دفعة كاملة (Batch Upload)
# ==========================================
with tab2:
    st.subheader("📁 تحليل دفعات الطلاب عبر ملفات CSV / Excel")
    st.write(
        "تتيح هذه الميزة لإدارة الكلية رفع كشوفات مئات الطلاب دفعة واحدة لاستخراج قائمة المعرضين للتعثر المالي فوراً."
    )

    # 1. زر تحميل قالب تجريبي
    template_df = pd.DataFrame(
        {
            "اسم الطالب": [
                "أحمد محمود",
                "سارة عبد الله",
                "عمر خالد",
                "فاطمة الزهراء",
            ],
            "الدخل الشهري": [25000, 85000, 15000, 110000],
            "نسبة المنحة": [0, 50, 10, 75],
            "التأخيرات السابقة": [3, 0, 4, 0],
        }
    )

    csv_template = template_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 تحميل قالب Excel/CSV جاهز للاستخدام",
        data=csv_template,
        file_name="students_template.csv",
        mime="text/csv",
    )

    st.write("")
    uploaded_file = st.file_uploader(
        "قم بسحب وإفلات ملف الطلاب هنا (CSV أو Excel):",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            st.success(
                f"✅ تم تحميل الملف بنجاح! يحتوي الملف على `{len(batch_df)}` سجلاً."
            )

            # معالجة وتوحيد أسماء الأعمدة لتتوافق مع النموذج
            column_mapping = {}
            for col in batch_df.columns:
                c = col.strip()
                if "دخل" in c or "income" in c.lower():
                    column_mapping[col] = "family_income"
                elif "منح" in c or "scholarship" in c.lower():
                    column_mapping[col] = "scholarship_ratio"
                elif "تأخير" in c or "delay" in c.lower():
                    column_mapping[col] = "previous_delays"
                elif "اسم" in c or "name" in c.lower():
                    column_mapping[col] = "student_name"

            processed_df = batch_df.rename(columns=column_mapping)

            required_cols = [
                "family_income",
                "scholarship_ratio",
                "previous_delays",
            ]
            missing_cols = [
                c for c in required_cols if c not in processed_df.columns
            ]

            if missing_cols:
                st.error(
                    f"⚠️ الملف المرفوع ينقصه بعض الأعمدة الأساسية المطلوبة: `{missing_cols}`. يرجى استخدام القالب النموذجي أعلاه."
                )
            else:
                # التأكد من وجود عمود اسم الطالب
                if "student_name" not in processed_df.columns:
                    processed_df["student_name"] = [
                        f"طالب {i+1}" for i in range(len(processed_df))
                    ]

                # تنظيف البيانات من القيم الفارغة إن وجدت
                input_features = processed_df[required_cols].fillna(0)

                # التنبؤ المباشر لجميع السجلات دفعة واحدة
                predictions = model.predict(input_features)
                probabilities = model.predict_proba(input_features)[:, 1] * 100

                # دمج النتائج
                results_df = batch_df.copy()
                results_df["حالة الخطر المتوقعة"] = np.where(
                    predictions == 1, "⚠️ معرض للتعثر", "✅ سداد منتظم"
                )
                results_df["نسبة احتمالية التعثر (%)"] = np.round(
                    probabilities, 1
                )
                results_df["التوصية الإدارية"] = np.where(
                    predictions == 1,
                    "متابعة استباقية / جدولة",
                    "لا يتطلب إجراء",
                )

                # عرض إحصائيات سريعة (KPIs)
                total_students = len(results_df)
                at_risk_count = int(np.sum(predictions == 1))
                safe_count = total_students - at_risk_count
                risk_percentage = (at_risk_count / total_students) * 100

                st.write("---")
                st.subheader("📊 ملخص مؤشرات المخاطر للدفعة المرفوعة:")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("إجمالي الطلاب", f"{total_students}")
                kpi2.metric(
                    "الطلاب الملتزمون",
                    f"{safe_count}",
                    f"{(safe_count/total_students)*100:.1f}%",
                )
                kpi3.metric(
                    "المعرضون للتعثر",
                    f"{at_risk_count}",
                    f"-{risk_percentage:.1f}%",
                    delta_color="inverse",
                )
                kpi4.metric(
                    "متوسط نسبة الخطر العامة",
                    f"{np.mean(probabilities):.1f}%",
                )

                st.write("---")

                # خيارات التصفية التفاعلية
                filter_choice = st.radio(
                    "عرض الطلاب حسب الحالة:",
                    [
                        "جميع الطلاب",
                        "الطلاب المعرضون للتعثر فقط ⚠️",
                        "الطلاب الملتزمون فقط ✅",
                    ],
                    horizontal=True,
                )

                if filter_choice == "الطلاب المعرضون للتعثر فقط ⚠️":
                    filtered_df = results_df[
                        results_df["حالة الخطر المتوقعة"] == "⚠️ معرض للتعثر"
                    ]
                elif filter_choice == "الطلاب الملتزمون فقط ✅":
                    filtered_df = results_df[
                        results_df["حالة الخطر المتوقعة"] == "✅ سداد منتظم"
                    ]
                else:
                    filtered_df = results_df

                st.dataframe(filtered_df, use_container_width=True)

                # زر تصدير النتائج بعد التحليل
                exported_csv = results_df.to_csv(index=False).encode(
                    "utf-8-sig"
                )
                st.download_button(
                    label="📥 تنزيل تقرير التحليل المالي الكامل (CSV معرب)",
                    data=exported_csv,
                    file_name="financial_risk_analysis_report.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"حدث خطأ أثناء قراءة الملف: {str(e)}")

# ==========================================
# 🟣 التبويب الثالث: لوحة التحليل الأكاديمي
# ==========================================
with tab3:
    st.subheader("🔬 التحليل العلمي لنظام الذكاء الاصطناعي")
    st.write(
        "هذا القسم يوضح للمشرفين ولجنة المناقشة معايير الدقة الرياضية وتفسير قرارات النموذج:"
    )

    acad_col1, acad_col2 = st.columns(2)
    with acad_col1:
        st.write(f"📊 **حجم عينة التدريب:** `1000` سجل طالب.")
        st.write(
            f"📈 **دقة التنبؤ المقياسة (Test Accuracy):** `{accuracy * 100:.1f}%`"
        )
        st.write(
            "⚙️ **الخوارزمية المعتمدة:** `Random Forest Classifier (100 Trees, Max Depth=5)`"
        )

    with acad_col2:
        st.write("🧠 **أهمية المتغيرات في اتخاذ القرار (Feature Importance):**")
        importance_labels = {
            "family_income": "الدخل الشهري للأسرة",
            "scholarship_ratio": "نسبة المنحة الدراسية",
            "previous_delays": "سجل التأخيرات السابقة",
        }
        for feat, imp in zip(features_list, importances):
            percent = imp * 100
            st.write(f"**{importance_labels[feat]}:** `{percent:.1f}%`")
            st.progress(float(imp))

    st.write("---")
    st.write("📋 **عينة من بيانات تدريب الخوارزمية (أول 5 صفوف):**")
    st.dataframe(df_train.head(5), use_container_width=True)
