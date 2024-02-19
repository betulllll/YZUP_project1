import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Veri setini yükleme fonksiyonu
@st.cache(allow_output_mutation=True)
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Gereksiz sütunları temizle
def clean_data(df):
    if 'id' in df.columns and 'Unnamed: 32' in df.columns:
        df.drop(columns=['id', 'Unnamed: 32'], axis=1, inplace=True)
    return df


# Veriyi M ve B olarak etiketle ve X ve Y olarak böl
def encode_data(df):
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
    Y = df['diagnosis']
    X = df.drop('diagnosis', axis=1)
    return X, Y


# Korelasyon matrisini çizdir
def draw_correlation_matrix(df):
    plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()


# Veriyi eğitim ve test setlerine ayır
def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, random_state=42)


# Modeli eğit
def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train)
    return model


# Model sonuçlarını göster
def show_results(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    st.write("### Model Performansı:")
    st.write("Accuracy:", accuracy_score(Y_test, Y_pred))
    st.write("Precision:", precision_score(Y_test, Y_pred))
    st.write("Recall:", recall_score(Y_test, Y_pred))
    st.write("F1 Score:", f1_score(Y_test, Y_pred))
    st.write("Confusion Matrix:")
    confusion_mat = confusion_matrix(Y_test, Y_pred)
    st.write(confusion_mat)
    fig, ax = plt.subplots()
    sns.heatmap(confusion_mat, annot=True, cmap='coolwarm', fmt='g', cbar=True, ax=ax)
    st.pyplot(fig)


# Ana uygulama
def show_data(df):
    st.write("### İlk 10 Satır:")
    st.write(df.head(10))
    st.write("### Sütunlar:")
    st.write(df.columns)


class App:
    def __init__(self):
        self.dataset_name = None
        self.model_name = None
        self.Init_Streamlit_Page()

        self.params = dict()
        self.clf = None
        self.X, self.y = None, None

    def run(self):
        pass

    def Init_Streamlit_Page(self):
        st.title("Meme Kanseri Teşhisi")
        file_path = st.sidebar.file_uploader("Veri Seti Yükle", type=['csv'])


        if file_path:
            df = load_data(file_path)
            show_data(df)
            df_cleaned = clean_data(df)  # Temizlenmiş veri setini sakla

            # Her model için orijinal veri setinin kopyasını kullanarak işlem yap
            df_copy = df_cleaned.copy()

            X, Y = encode_data(df_copy)

            draw_correlation_matrix(df_copy)

            st.write("Malignant ve benign verilerin dağılımı:")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_copy, x='radius_mean', y='texture_mean', hue='diagnosis', palette='viridis', ax=ax)
            ax.set_xlabel('Radius Mean')
            ax.set_ylabel('Texture Mean')
            ax.set_title('Radius Mean vs Texture Mean')
            st.pyplot(fig)
            X = df_copy.drop(columns=['diagnosis'])
            y = df_copy['diagnosis']

            # Veriyi eğitim ve test setlerine ayır
            X_train, X_test, Y_train, Y_test = split_data(X, Y)

            # Model seçimi
            self.model_name = st.sidebar.selectbox("Model Seçin", ["KNN", "SVM", "Naive Bayes"])

            if self.model_name == "KNN":
                st.sidebar.write("KNN seçildi.")
                k_value = st.sidebar.slider("K Değeri Seçin", 1, 20)
                model = KNeighborsClassifier(n_neighbors=k_value)
            elif self.model_name == "SVM":
                st.sidebar.write("SVM seçildi.")
                model = SVC(kernel='linear')
            else:
                st.sidebar.write("Naive Bayes seçildi.")
                model = GaussianNB()


            # Modeli eğit
            trained_model = train_model(model, X_train, Y_train)

            # Sonuçları göster
            show_results(trained_model, X_test, Y_test)
