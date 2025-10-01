import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
import numpy as np

st.set_page_config(page_title="Análise ENEM", layout="centered")

@st.cache_data(ttl=3600*20)
def load_data():
    return pd.read_csv('data/enem.csv', encoding='latin-1', sep=';').dropna()

# Limpa os dados removendo outliers usando IsolationForest e tratando valores nulos
@st.cache_data(ttl=3600*20)
def clean_data(df):
    df_clean = df.copy()
    
    numeric_cols = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    df_clean = df_clean.dropna(subset=numeric_cols)
    df_clean = df_clean[(df_clean[numeric_cols] > 0).all(axis=1)]
    
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers = isolation_forest.fit_predict(df_clean[numeric_cols])
    df_clean = df_clean[outliers == 1]
    
    return df_clean

def question1(df):
    st.subheader("Renda Familiar vs Notas de Ciências da Natureza")
    
    # Calcular médias por faixa de renda
    estatisticas_descritivas = df.groupby('Q006')['NU_NOTA_CN'].describe()
    
    # Criar abas para alternar entre gráfico e tabela
    tab1, tab2 = st.tabs(["📊 Gráfico", "📋 Tabela Descritiva"])
    
    with tab1:
        st.bar_chart(estatisticas_descritivas['mean'], x_label="Faixa de Renda", y_label="Nota Média")
    
    with tab2:
        st.dataframe(estatisticas_descritivas)
    
    # Adicionar legenda explicativa em um expander
    with st.expander("📋 Ver Legenda das Faixas de Renda"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("A: Nenhuma Renda")
            st.caption("B: Até R$ 1.100") 
            st.caption("C: R$ 1.100 - 1.650")
            st.caption("D: R$ 1.650 - 2.200")
            st.caption("E: R$ 2.200 - 2.750")
            st.caption("F: R$ 2.750 - 3.300")
            st.caption("G: R$ 3.300 - 4.400")
            st.caption("H: R$ 4.400 - 5.500")
            st.caption("I: R$ 5.500 - 6.600")
            
        with col2:
            st.caption("J: R$ 6.600 - 7.700")
            st.caption("K: R$ 7.700 - 8.800")
            st.caption("L: R$ 8.800 - 9.900")
            st.caption("M: R$ 9.900 - 11.000")
            st.caption("N: R$ 11.000 - 13.200")
            st.caption("O: R$ 13.200 - 16.500")
            st.caption("P: R$ 16.500 - 22.000")
            st.caption("Q: Acima de R$ 22.000")

def question3(df):
    st.subheader("Distribuição de Desempenho nas Provas Objetivas por Região Geográfica")
    
    # Criar coluna da região baseada no primeiro dígito do código do município
    df = df.copy()
    df['REGIAO_COD'] = df['CO_MUNICIPIO_PROVA'].astype(str).str[0]
    
    # Mapear códigos das regiões para nomes
    mapeamento_regioes = {
        '1': 'Norte',
        '2': 'Nordeste', 
        '3': 'Sudeste',
        '4': 'Sul',
        '5': 'Centro-Oeste'
    }
    
    df['REGIAO'] = df['REGIAO_COD'].map(mapeamento_regioes)
    
    # Calcular média geral das provas objetivas por participante
    colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']
    df['MEDIA_GERAL'] = df[colunas_notas].mean(axis=1)
    
    # Calcular estatísticas descritivas por região
    estatisticas_descritivas = df.groupby('REGIAO')['MEDIA_GERAL'].describe()
    
    # Criar abas para alternar entre gráfico e tabela
    tab1, tab2 = st.tabs(["📊 Gráfico", "📋 Tabela Descritiva"])
    
    with tab1:
        st.bar_chart(estatisticas_descritivas['mean'], x_label="Região", y_label="Nota Média")
    
    with tab2:
        st.dataframe(estatisticas_descritivas)
    
    # Adicionar explicação em um expander
    with st.expander("📋 Ver Informações sobre as Regiões"):
        st.caption("**Norte (1):** Acre, Amapá, Amazonas, Pará, Rondônia, Roraima, Tocantins")
        st.caption("**Nordeste (2):** Alagoas, Bahia, Ceará, Maranhão, Paraíba, Pernambuco, Piauí, Rio Grande do Norte, Sergipe")
        st.caption("**Sudeste (3):** Espírito Santo, Minas Gerais, Rio de Janeiro, São Paulo")
        st.caption("**Sul (4):** Paraná, Rio Grande do Sul, Santa Catarina")
        st.caption("**Centro-Oeste (5):** Distrito Federal, Goiás, Mato Grosso, Mato Grosso do Sul")

def main():
    # Carregar dados
    df_original = load_data()
    
    # Aplicar limpeza de dados
    df = clean_data(df_original)
    
    st.title("Análise dos Dados do ENEM")
    st.info("📈 Os dados foram automaticamente limpos removendo outliers e valores nulos usando IsolationForest do scikit-learn")
    
    # Mostrar estatísticas de debug em um expander colapsável
    with st.expander("🔍 Ver Estatísticas de Processamento dos Dados"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dados Originais", f"{len(df_original):,}")
        
        with col2:
            st.metric("Dados Limpos", f"{len(df):,}")
        
        with col3:
            outliers_removidos = len(df_original) - len(df)
            percentual = outliers_removidos/len(df_original)*100
            st.metric("Outliers Removidos", f"{outliers_removidos:,}", f"{percentual:.1f}%")

    # Pergunta 1: Qual é a relação entre a renda familiar declarada pelos participantes e suas notas médias na prova de Ciências da Natureza?
    question1(df)
    
    # Pergunta 3: Qual é a distribuição de desempenho nas provas objetivas por regiões geográficas do Brasil?
    question3(df)
    

if __name__ == '__main__': 
    main()