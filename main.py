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
    st.subheader("Renda Familiar vs Nota Geral das Provas Objetivas")
    
    # Calcular a nota geral (média das provas objetivas) para cada participante
    colunas_provas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']
    df = df.copy()
    df['NOTA_GERAL'] = df[colunas_provas].mean(axis=1)
    
    # Calcular estatísticas descritivas da nota geral por faixa de renda
    estatisticas_nota_geral = df.groupby('Q006')['NOTA_GERAL'].describe()
    
    # Criar abas para cada tipo de visualização
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Análise Interpretativa", "📊 Gráficos", "📋 Tabelas Descritivas", "🔗 Análise de Correlação"])
    
    with tab1:
        st.write("### 🎓 O que os dados nos revelam sobre renda e desempenho no ENEM?")
        
        # Calcular algumas estatísticas para a análise
        nota_mais_alta = estatisticas_nota_geral['mean'].max()
        renda_mais_alta = estatisticas_nota_geral['mean'].idxmax()
        nota_mais_baixa = estatisticas_nota_geral['mean'].min()
        renda_mais_baixa = estatisticas_nota_geral['mean'].idxmin()
        diferenca_notas = nota_mais_alta - nota_mais_baixa
        
        # Análise principal
        st.write("#### 🔍 **Principais Descobertas:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **🏆 Maior Desempenho:**
            - Faixa de renda: **{renda_mais_alta}**
            - Nota média: **{nota_mais_alta:.1f} pontos**
            """)
        
        with col2:
            st.warning(f"""
            **📉 Menor Desempenho:**
            - Faixa de renda: **{renda_mais_baixa}**
            - Nota média: **{nota_mais_baixa:.1f} pontos**
            """)
        
        st.write("#### 💡 **O que isso significa na prática?**")
        
        st.write(f"""
        **Diferença de Desempenho:** Existe uma diferença de **{diferenca_notas:.1f} pontos** entre 
        as faixas de renda mais alta e mais baixa. Isso representa aproximadamente 
        **{(diferenca_notas/nota_mais_baixa)*100:.1f}%** de diferença no desempenho.
        """)
        
        st.write("#### 🤔 **Possíveis Explicações:**")
        
        st.write("""
        **Por que famílias com maior renda tendem a ter filhos com melhor desempenho?**
        
        🏠 **Recursos e Ambiente:** Melhor infraestrutura para estudos, materiais didáticos de qualidade e acesso à tecnologia.
        
        📚 **Investimento Educacional:** Cursos preparatórios, aulas particulares e educação complementar.
        
        🎯 **Condições de Estudo:** Menor necessidade de trabalhar, mais tempo para se dedicar aos estudos e menos preocupações financeiras.
        
        🏫 **Qualidade Educacional:** Acesso a escolas de melhor qualidade e atividades extracurriculares.
        """)
        
        st.warning("""
        ⚠️ **Importante lembrar:** Estes dados mostram uma tendência geral, mas existem muitas 
        exceções! Estudantes de todas as faixas de renda podem ter excelente desempenho 
        com dedicação, boas estratégias de estudo e apoio adequado.
        """)
    
    with tab2:
        st.write("**Nota Geral das Provas Objetivas por Faixa de Renda**")
        st.bar_chart(estatisticas_nota_geral['mean'], x_label="Faixa de Renda", y_label="Nota Média Geral")
    
    with tab3:
        st.write("**Estatísticas Descritivas da Nota Geral por Faixa de Renda**")
        st.dataframe(estatisticas_nota_geral)
    
    with tab4:      
        # Converter faixas de renda para valores numéricos ordinais
        mapeamento_renda = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
            'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17
        }
        
        df_corr = df.copy()
        df_corr['RENDA_NUMERICA'] = df_corr['Q006'].map(mapeamento_renda)
        
        # Calcular correlação entre renda e nota geral
        correlacao_geral = df_corr['RENDA_NUMERICA'].corr(df_corr['NOTA_GERAL'])
        
        # Mostrar correlação
        st.write("#### 📈 Correlação entre Renda Familiar e Nota Geral:")
        
        st.metric(
            label="Coeficiente de Correlação",
            value=f"{correlacao_geral:.4f}",
            help="Valores próximos a 1 indicam correlação positiva forte"
        )
        
        # Interpretar o resultado
        if correlacao_geral > 0.7:
            interpretacao = "🟢 **Correlação Forte Positiva**"
            descricao = "Existe uma relação forte entre maior renda familiar e maiores notas."
        elif correlacao_geral > 0.3:
            interpretacao = "🟡 **Correlação Moderada Positiva**"
            descricao = "Existe uma relação moderada entre maior renda familiar e maiores notas."
        elif correlacao_geral > 0.1:
            interpretacao = "🟠 **Correlação Fraca Positiva**"
            descricao = "Existe uma relação fraca entre maior renda familiar e maiores notas."
        else:
            interpretacao = "🔴 **Correlação Muito Fraca ou Inexistente**"
            descricao = "Não há uma relação significativa entre renda familiar e notas."
        
        st.success(f"{interpretacao}")
        st.write(descricao)
        
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
    
    # Criar abas para alternar entre análise, gráfico, tabela e correlação
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Análise Interpretativa", "📊 Gráfico", "📋 Tabela Descritiva", "🔗 Análise de Correlação"])
    
    with tab1:
        st.write("### 🗺️ Como o desempenho no ENEM varia entre as regiões do Brasil?")
        
        # Calcular estatísticas para a análise
        melhor_regiao = estatisticas_descritivas['mean'].idxmax()
        melhor_nota = estatisticas_descritivas['mean'].max()
        pior_regiao = estatisticas_descritivas['mean'].idxmin()
        pior_nota = estatisticas_descritivas['mean'].min()
        diferenca_regioes = melhor_nota - pior_nota
        
        # Análise principal
        st.write("#### 🔍 **Principais Descobertas:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **🏆 Melhor Desempenho:**
            - Região: **{melhor_regiao}**
            - Nota média: **{melhor_nota:.1f} pontos**
            """)
        
        with col2:
            st.warning(f"""
            **📉 Menor Desempenho:**
            - Região: **{pior_regiao}**
            - Nota média: **{pior_nota:.1f} pontos**
            """)
        
        st.write("#### 💡 **O que isso significa na prática?**")
        
        st.write(f"""
        **Diferença Regional:** Existe uma diferença de **{diferenca_regioes:.1f} pontos** entre 
        a região com melhor e pior desempenho. Isso representa aproximadamente 
        **{(diferenca_regioes/pior_nota)*100:.1f}%** de diferença no desempenho entre regiões.
        """)
        
        # Ranking das regiões
        ranking_regioes = estatisticas_descritivas['mean'].sort_values(ascending=False)
        
        st.write("#### 🏅 **Ranking das Regiões por Desempenho:**")
        
        for i, (regiao, nota) in enumerate(ranking_regioes.items(), 1):
            if i == 1:
                emoji = "🥇"
                cor = "success"
            elif i == 2:
                emoji = "🥈"
                cor = "info"
            elif i == 3:
                emoji = "🥉"
                cor = "info"
            else:
                emoji = f"{i}º"
                cor = "secondary"
            
            with st.container():
                if cor == "success":
                    st.success(f"{emoji} **{regiao}**: {nota:.1f} pontos")
                elif cor == "info":
                    st.info(f"{emoji} **{regiao}**: {nota:.1f} pontos")
                else:
                    st.write(f"{emoji} **{regiao}**: {nota:.1f} pontos")
        
        st.write("#### 🤔 **Possíveis Explicações para as Diferenças:**")
        
        st.write("""
        **Por que existem diferenças regionais no desempenho do ENEM?**
        
        🏭 **Desenvolvimento Econômico:** Maior renda e investimento em infraestrutura educacional.
        
        🏫 **Qualidade da Educação:** Diferenças no investimento per capita, formação docente e recursos escolares.
        
        🌆 **Concentração Urbana:** Acesso a mais escolas, universidades e mercado competitivo.
        
        📚 **Acesso a Recursos:** Proximidade a centros urbanos, conectividade e materiais didáticos.
        """)
        
        # Análise estatística adicional
        desvio_padrao_medio = estatisticas_descritivas['std'].mean()
        coeficiente_variacao = (estatisticas_descritivas['std'] / estatisticas_descritivas['mean'] * 100).mean()
        
        st.write("#### 📊 **Análise da Variabilidade:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📈 Nota Média Geral", 
                f"{estatisticas_descritivas['mean'].mean():.1f} pts",
                help="Média de todas as regiões"
            )
        
        with col2:
            st.metric(
                "📏 Desvio Padrão Médio", 
                f"{desvio_padrao_medio:.1f} pts",
                help="Variabilidade média dentro das regiões"
            )
        
        with col3:
            st.metric(
                "🔄 Coef. de Variação", 
                f"{coeficiente_variacao:.1f}%",
                help="Percentual de variação entre regiões"
            )
        
        st.info("""
        💡 **Interpretação:** Um coeficiente de variação de 13.4% indica uma variabilidade moderada 
        entre as regiões. Isso significa que, embora existam diferenças regionais visíveis, 
        elas não são extremamente grandes quando comparadas à variação dentro de cada região.
        
        📊 **Referência:** 
        - Baixa variação: < 10%
        - Moderada variação: 10% - 20%  
        - Alta variação: > 20%
        """)
        
        st.warning("""
        ⚠️ **Importante lembrar:** Estes dados refletem tendências regionais gerais, mas cada região 
        possui grande diversidade interna. Estudantes excepcionais existem em todas as regiões, 
        e fatores individuais como dedicação, qualidade da escola específica e apoio familiar 
        podem ser mais determinantes que a região geográfica.
        """)
    
    with tab2:
        st.bar_chart(estatisticas_descritivas['mean'], x_label="Região", y_label="Nota Média")
    
    with tab3:
        st.dataframe(estatisticas_descritivas)
    
    with tab4:
        st.write("#### 🔗 Correlação entre Região e Desempenho:")
        
        # Converter regiões para valores numéricos ordinais (baseado no ranking de desempenho)
        ranking_regioes = estatisticas_descritivas['mean'].sort_values(ascending=False)
        mapeamento_regiao_numerica = {regiao: i+1 for i, regiao in enumerate(ranking_regioes.index)}
        
        df_corr = df.copy()
        df_corr['REGIAO_NUMERICA'] = df_corr['REGIAO'].map(mapeamento_regiao_numerica)
        
        # Calcular correlação entre região (ordenada por desempenho) e nota geral
        correlacao_regional = df_corr['REGIAO_NUMERICA'].corr(df_corr['MEDIA_GERAL'])
        
        # Como ordenamos do melhor para o pior (1=melhor), a correlação será negativa
        # Vamos inverter o sinal para facilitar a interpretação
        correlacao_regional_abs = abs(correlacao_regional)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Coeficiente de Correlação",
                value=f"{correlacao_regional_abs:.4f}",
                help="Correlação entre posição regional no ranking e desempenho individual"
            )
        
        with col2:
            # Calcular variância explicada pela região (R²)
            r_quadrado = correlacao_regional_abs ** 2
            st.metric(
                label="Variância Explicada (R²)",
                value=f"{r_quadrado:.4f}",
                help="Percentual da variação individual explicada pela região"
            )
        
        # Interpretar o resultado
        if correlacao_regional_abs > 0.7:
            interpretacao = "🟢 **Correlação Forte**"
            descricao = "A região geográfica tem uma influência forte no desempenho individual."
        elif correlacao_regional_abs > 0.3:
            interpretacao = "🟡 **Correlação Moderada**"
            descricao = "A região geográfica tem uma influência moderada no desempenho individual."
        elif correlacao_regional_abs > 0.1:
            interpretacao = "🟠 **Correlação Fraca**"
            descricao = "A região geográfica tem uma influência fraca no desempenho individual."
        else:
            interpretacao = "🔴 **Correlação Muito Fraca**"
            descricao = "A região geográfica tem influência mínima no desempenho individual."
        
        st.success(f"{interpretacao}")
        st.write(descricao)
        
        # Análise adicional por disciplina
        st.write("#### 📚 **Correlação por Disciplina:**")
        
        disciplinas = {
            'NU_NOTA_CN': 'Ciências da Natureza',
            'NU_NOTA_CH': 'Ciências Humanas', 
            'NU_NOTA_LC': 'Linguagens e Códigos',
            'NU_NOTA_MT': 'Matemática'
        }
        
        correlacoes_disciplinas = {}
        for codigo, nome in disciplinas.items():
            corr = abs(df_corr['REGIAO_NUMERICA'].corr(df_corr[codigo]))
            correlacoes_disciplinas[nome] = corr
        
        # Mostrar correlações por disciplina
        for disciplina, corr in correlacoes_disciplinas.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{disciplina}:**")
            with col2:
                st.write(f"{corr:.4f}")
        
        # Encontrar disciplina com maior e menor correlação regional
        disciplina_maior_corr = max(correlacoes_disciplinas, key=correlacoes_disciplinas.get)
        disciplina_menor_corr = min(correlacoes_disciplinas, key=correlacoes_disciplinas.get)
        
        st.info(f"""
        💡 **Insights por Disciplina:**
        
        - **Maior influência regional:** {disciplina_maior_corr} ({correlacoes_disciplinas[disciplina_maior_corr]:.4f})
        - **Menor influência regional:** {disciplina_menor_corr} ({correlacoes_disciplinas[disciplina_menor_corr]:.4f})
        
        Isso pode indicar que algumas áreas do conhecimento são mais sensíveis às 
        diferenças regionais de infraestrutura, recursos ou tradição educacional.
        """)
        
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