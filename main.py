import pandas as pd
import streamlit as st

# Configurar página para ocupar toda a tela
st.set_page_config(page_title="Análise ENEM", layout="centered")

@st.cache_data(ttl=3600*20)
def load_data():
    return pd.read_csv('data/enem.csv', encoding='latin-1', sep=';').dropna()

def question1(df):
    st.subheader("Renda Familiar vs Notas de Ciências da Natureza")
    
    # Calcular médias por faixa de renda
    medias = df.groupby('Q006')['NU_NOTA_CN'].mean().round(1)
    
    # Ordenar pelos códigos (A-Q)
    medias = medias.sort_index()
    
    # Exibir gráfico com códigos no eixo X
    st.bar_chart(medias, x_label="Faixa de Renda", y_label="Nota Média")
    
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
    

def main():
    # Carregar dados
    df = load_data()
    st.title("Análise dos Dados do ENEM")

    # Pergunta 1: Qual é a relação entre a renda familiar declarada pelos participantes e suas notas médias na prova de Ciências da Natureza?
    question1(df)
    

if __name__ == '__main__': 
    main()