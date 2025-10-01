import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

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
    st.subheader("Renda Familiar vs Notas das Provas Objetivas")
    
    # Definir as colunas das provas objetivas
    colunas_provas = {
        'NU_NOTA_CN': 'Ciências da Natureza',
        'NU_NOTA_CH': 'Ciências Humanas', 
        'NU_NOTA_LC': 'Linguagens e Códigos',
        'NU_NOTA_MT': 'Matemática'
    }
    
    # Calcular estatísticas descritivas por faixa de renda para cada prova
    estatisticas_por_prova = {}
    for coluna, nome_prova in colunas_provas.items():
        estatisticas_por_prova[nome_prova] = df.groupby('Q006')[coluna].describe()
    
    # Criar abas para cada tipo de visualização
    tab1, tab2, tab3 = st.tabs(["📊 Gráficos", "📋 Tabelas Descritivas", "🔗 Análise de Correlação"])
    
    with tab1:
        # Criar gráficos para cada prova
        for nome_prova, estatisticas in estatisticas_por_prova.items():
            st.write(f"**{nome_prova}**")
            st.bar_chart(estatisticas['mean'], x_label="Faixa de Renda", y_label="Nota Média")
            st.write("---")
    
    with tab2:
        # Criar tabelas descritivas para cada prova
        for nome_prova, estatisticas in estatisticas_por_prova.items():
            st.write(f"**{nome_prova}**")
            st.dataframe(estatisticas)
            st.write("---")
    
    with tab3:      
        # Converter faixas de renda para valores numéricos ordinais
        mapeamento_renda = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
            'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17
        }
        
        df_corr = df.copy()
        df_corr['RENDA_NUMERICA'] = df_corr['Q006'].map(mapeamento_renda)
        
        # Calcular correlações individuais
        correlacoes = {}
        for coluna, nome_prova in colunas_provas.items():
            correlacao = df_corr['RENDA_NUMERICA'].corr(df_corr[coluna])
            correlacoes[nome_prova] = correlacao
        
        # Mostrar correlações individuais
        st.write("#### 📈 Correlação Individual de cada Prova com Renda Familiar:")
        
        # Criar DataFrame para visualização
        df_correlacoes = pd.DataFrame(list(correlacoes.items()), columns=['Prova', 'Correlação'])
        df_correlacoes = df_correlacoes.sort_values('Correlação', ascending=False)
        
        # Mostrar gráfico de barras das correlações
        st.bar_chart(df_correlacoes.set_index('Prova')['Correlação'], 
                    x_label="Prova", y_label="Coeficiente de Correlação")
        
        # Mostrar interpretação
        prova_maior_correlacao = max(correlacoes, key=correlacoes.get)
        valor_maior_correlacao = correlacoes[prova_maior_correlacao]
        
        st.success(f"🏆 **{prova_maior_correlacao}** tem a maior correlação com a renda familiar: **{valor_maior_correlacao:.4f}**")
        
        with st.expander("📊 Como interpretar os valores de correlação"):
            st.write("""
            - **0.7 a 1.0**: Correlação forte positiva
            - **0.3 a 0.7**: Correlação moderada positiva  
            - **0.1 a 0.3**: Correlação fraca positiva
            - **-0.1 a 0.1**: Correlação muito fraca ou inexistente
            - **-0.3 a -0.1**: Correlação fraca negativa
            - **-0.7 a -0.3**: Correlação moderada negativa
            - **-1.0 a -0.7**: Correlação forte negativa
            """)
    
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

    # Pergunta 1: Qual é a relação entre a renda familiar declarada pelos participantes e suas notas médias nas provas objetivas?
    question1(df)
    
    # Pergunta 3: Qual é a distribuição de desempenho nas provas objetivas por regiões geográficas do Brasil?
    question3(df)
    

if __name__ == '__main__': 
    main()