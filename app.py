import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# st.set_page_config(layout="wide")

# ======= Load data ======= #
df = pd.read_csv('data/heart_disease_uci.csv')
df.drop(['id', 'dataset'], axis=1, inplace=True, errors='ignore')

# ======= Handle missing values ======= #
if 'fbs' in df.columns:
    df = df[df['fbs'].isin([0, 1])]
df.dropna(subset=['slope', 'thal', 'ca'], inplace=True)

# ======= Label encoding ======= #
label_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ======= Binary target ======= #
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop('num', axis=1, inplace=True)

# ======= Outlier removal ======= #
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target')
if 'fbs' in numeric_cols:
    numeric_cols = numeric_cols.drop('fbs')
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# ======= Split X/y and scale ======= #
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======= Train model ======= #
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ======= Feature importance ======= #
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# ======= Performance ======= #
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])

# ======= Sidebar Navigation ======= #
with st.sidebar:
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    def set_page(p): st.session_state.page = p

    st.button("🏠 Home", on_click=set_page, args=("Home",))
    st.button("🧪 데이터 분석", on_click=set_page, args=("Data Analysis",))
    st.button("📊 데이터 시각화", on_click=set_page, args=("EDA",))
    st.button("📈 머신러닝 보고서", on_click=set_page, args=("Model Report",))

menu = st.session_state.page

# ======= Page: Home ======= #
def home():
    st.title("💓 심장병 분류 대시보드")
    st.markdown(f"""
    - 📌 본 대시보드는 머신러닝을 활용한 심장병 분류 모델 결과를 시각화합니다.  
    - 🎯 **타겟 정의:**  
        - `0` = 심장병 없음  
        - `1` = 심장병 있음  
    - 🔍 **사용된 특성 수:** {X.shape[1]}개
    """)


# ======= Page: Data Analysis ======= #
def analyze():
    st.title("데이터 분석")

    with st.expander("📖 데이터 컬럼 설명 보기"):
        st.markdown("""
        - age : 나이  
        - sex : 성별 (Male, Female)  
        - cp : 흉통 유형 (Chest Pain Type)  
        - trestbps : 안정 시 혈압 (Resting Blood Pressure)  
        - chol : 콜레스테롤 수치 (Serum Cholesterol)  
        - restecg : 안정 심전도 결과 (Resting ECG)  
        - thalach : 최대 심박수 (Max Heart Rate Achieved)  
        - exang : 운동 중 협심증 여부 (Exercise Induced Angina)  
        - oldpeak : 운동으로 인한 ST 하강치  
        - slope : ST 분절 기울기  
        - ca : 주요 혈관 수 (형광 투시 검사 결과)  
        - thal : 결함 유형 (Normal, Fixed Defect, Reversible Defect)  
        """)

    analyze_tabs = st.tabs(["Head", "Statistics", "Column Info", "Conditional Filter"])

    with analyze_tabs[0]:
        st.subheader("📌 Preview Data")
        st.dataframe(df.head())

    with analyze_tabs[1]:
        st.subheader("📊 Summary Statistics")
        st.dataframe(df.describe().T)

    with analyze_tabs[2]:
        st.subheader("📋 Column Details")
        col = st.selectbox("Select a column", df.columns)
        col_data = df[col]
        st.write(pd.DataFrame({
            'Type': [col_data.dtype],
            'Missing': [col_data.isnull().sum()],
            'Uniques': [col_data.nunique()],
            'Examples': [col_data.unique()[:5]]
        }))

    with analyze_tabs[3]:
        st.subheader("🔍 Filter by Column Value")
        col = st.selectbox("Select column", df.columns, key="filter_col")
        val = st.selectbox("Select value", df[col].unique(), key="filter_val")
        st.write(f"Filtered rows: {df[df[col] == val].shape[0]}")
        st.dataframe(df[df[col] == val])

# ======= Page: EDA ======= #
def eda():
    st.title("데이터 시각화")
    tabs = st.tabs([
        "Distribution", "Violin Plot",
        "Categorical Count", "Heatmap", "Boxplot"
    ])

    with tabs[0]:
        st.subheader("▶ Histogram & KDE for Numeric Features")

        colors = sns.color_palette('pastel', len(numeric_cols))

        # 차트를 2열로 구성
        num_cols = 2
        num_rows = int(np.ceil(len(numeric_cols) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i])
            axes[i].set_title(f"{col} Distribution", fontsize=14)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
            axes[i].grid(True)
            axes[i].set_facecolor("#f9f9f9")

        # 남는 축은 숨기기
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("▶ Violin Plot by Target")

        colors = sns.color_palette('pastel', len(numeric_cols))
        num_cols = 2
        num_rows = int(np.ceil(len(numeric_cols) / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.violinplot(data=df, x='target', y=col, ax=axes[i], palette=[colors[i]])
            axes[i].set_title(f"{col} by Target", fontsize=14)
            axes[i].grid(True)
            axes[i].set_facecolor("#f9f9f9")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        st.pyplot(fig)

    with tabs[2]:
        st.subheader("▶ Categorical Feature Counts")

        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        colors = sns.color_palette('Set3', len(cat_cols))
        num_cols = 2
        num_rows = int(np.ceil(len(cat_cols) / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            sns.countplot(data=df, x=col, ax=axes[i], color=colors[i])
            axes[i].set_title(f"{col} Count", fontsize=14)
            axes[i].grid(True)
            axes[i].set_facecolor("#f9f9f9")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        st.pyplot(fig)

    with tabs[3]:
        st.subheader("▶ Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
    with tabs[4]:
        st.subheader("▶ Boxplots of 'oldpeak' and 'cp' by Sex and Heart Disease")

        df_plot = df[['sex', 'target', 'oldpeak', 'cp']].copy()
        df_plot['Heart Disease'] = df_plot['target'].map({0: 'No Disease', 1: 'Disease'})

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(data=df_plot, x='sex', y='oldpeak', hue='Heart Disease', ax=axes[0], palette='Set2')
        axes[0].set_title("Oldpeak by Sex and Heart Disease")
        axes[0].set_xlabel("Sex (0 = Female, 1 = Male)")
        axes[0].set_ylabel("Oldpeak")

        sns.boxplot(data=df_plot, x='sex', y='cp', hue='Heart Disease', ax=axes[1], palette='Set3')
        axes[1].set_title("Chest Pain Type by Sex and Heart Disease")
        axes[1].set_xlabel("Sex (0 = Female, 1 = Male)")
        axes[1].set_ylabel("Chest Pain Type")

        plt.tight_layout()
        st.pyplot(fig)


# ======= Page: Model Report ======= #
def model_performance():
    st.title("머신러닝 보고서")
    st.write(f"### Accuracy: {accuracy:.2f}")

    report_df = pd.DataFrame(classification_report(
        y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True
    )).T.reset_index().rename(columns={'index': 'Label'})
    st.dataframe(report_df)

    st.write("### Feature Importances")
    st.bar_chart(feature_importances.set_index("Feature"))

# ======= Router ======= #
if menu == 'Home':
    home()
elif menu == 'Data Analysis':
    analyze()
elif menu == 'EDA':
    eda()
elif menu == 'Model Report':
    model_performance()
