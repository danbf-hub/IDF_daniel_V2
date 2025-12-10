import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from idf_core import processar_idf

#Leitura fixa da planilha de coeficientes
try:
    df_coef = pd.read_excel("Coeficientes.xlsx")

    # Remove linhas em branco e padroniza texto
    df_coef = df_coef.dropna(subset=["UF", "NOME MUNIC"])
    df_coef["UF"] = df_coef["UF"].astype(str).str.strip().str.upper()
    df_coef["NOME MUNIC"] = df_coef["NOME MUNIC"].astype(str).str.strip()

    # Lista única de estados
    ufs_disponiveis = sorted(df_coef["UF"].unique())


except Exception as e:
    st.error(f"Erro ao carregar a planilha de coeficientes fixa: {e}")
    st.stop()

# Função robusta para ler .csv
def ler_dados_precipitacao(arquivo):
    nome = arquivo.name
    def tentar_ler_csv(delimitador, skip=13):
        try:
            df = pd.read_csv(arquivo, encoding='latin1', delimiter=delimitador, skiprows=skip)
            if df.empty or df.columns.size == 0:
                raise ValueError("Sem colunas")
            return df
        except Exception:
            return None

    if nome.lower().endswith(".csv"):
        delimitadores = [';', ',', '\t']
        for delim in delimitadores:
            arquivo.seek(0)
            df = tentar_ler_csv(delim, skip=13)
            if df is not None:
                break
        else:
            for delim in delimitadores:
                arquivo.seek(0)
                df = tentar_ler_csv(delim, skip=0)
                if df is not None:
                    break
            else:
                st.error("Erro: não foi possível identificar o delimitador do CSV.")
                return None

        df.columns = [col if pd.notna(col) else f"Coluna_{i}" for i, col in enumerate(df.columns)]
        df = df.fillna("")
        return df

    else:
        st.error("Formato de arquivo não suportado.")
        return None

# Layout
st.set_page_config(page_title="Gerador de Curvas IDF", layout="wide")
st.title("🌧️Gerador de Curvas IDF via Distribuição GEV🌧️")

# Passo 1 - Upload dos dados de chuva
st.header("1. Upload do arquivo de precipitação (.csv)")
st.write("Delimitador sendo (;), com cabeçalho, formatação da Plataforma Hidroweb da ANA")
arquivo_chuvas = st.file_uploader("Dados de Precipitação", type=["csv"])

# Passo 2 - Tipo de dado
st.header("2. Qual é o tipo de dados de precipitação?")
tipo_dado = st.radio(
    "Selecione uma opção:",
    options=["Diário", "Horário", "5 minutos", "1 minuto"],
    index=0,
    horizontal=True
)

# Passo 3 - Sigla do estado
estado = st.selectbox(
    label="Selecione a UF (sigla do estado)",
    options=ufs_disponiveis,
    index=None
)

# Passo 4 - Nome do município
municipios_filtrados = []
if estado:
    municipios_filtrados = sorted(df_coef[df_coef["UF"] == estado]["NOME MUNIC"].unique())

municipio = st.selectbox(
    label="Selecione o município",
    options=municipios_filtrados,
    index=None
)

# Diagnóstico
if arquivo_chuvas is not None:
    st.header("3. Diagnóstico dos dados de chuva (antes da análise)")
    df_prec = ler_dados_precipitacao(arquivo_chuvas)
    if df_prec is not None:
        try:
            df_diagnostico = df_prec.copy()
            df_diagnostico.columns = df_diagnostico.columns.str.strip().str.lower()
            df_diagnostico['data'] = df_diagnostico['data'].astype(str).str.strip()
            df_diagnostico['data'] = pd.to_datetime(df_diagnostico['data'], format="%d/%m/%Y", errors='coerce')
            df_diagnostico['maxima'] = df_diagnostico['maxima'].astype(str).str.replace(',', '.').str.strip()
            df_diagnostico['maxima'] = pd.to_numeric(df_diagnostico['maxima'], errors='coerce')
            df_diagnostico = df_diagnostico.dropna(subset=['data', 'maxima'])
            df_diagnostico['ano'] = df_diagnostico['data'].dt.year
            df_anuais = df_diagnostico.groupby('ano')['maxima'].max().reset_index()

            if df_anuais.empty:
                st.warning("Nenhum dado anual válido foi encontrado.")
            else:
                st.success("Os dados são válidos para processamento")

        except Exception as e:
            st.error(f"Erro ao processar o diagnóstico dos dados: {e}")

# Passo 4 - Executar a análise
st.header("4. Executar análise e gerar curva IDF")
if st.button("Executar análise"):
    if not (arquivo_chuvas and municipio):
        st.warning("Envie o arquivo de chuvas e preencha o nome do município.")
    elif tipo_dado == 'Diário':
        with st.spinner("Processando..."):
            df_prec = ler_dados_precipitacao(arquivo_chuvas)
            if df_prec is None:
                st.stop()

            nome_arquivo = arquivo_chuvas.name
            resultado = processar_idf(df_prec, df_coef, municipio, nome_arquivo=nome_arquivo)

            if isinstance(resultado, str):
                st.error(f"Erro: {resultado}")
            else:
                parametros, matriz, duracoes, trs = resultado

                # 🚨 AVISO: menos de 10 anos de dados
                if parametros.get("Num_Anos", 0) < 10:
                    st.warning("⚠️ Número de dados anuais inferior a 10 anos. A análise será severamente prejudicada!")

                st.success("Processamento concluído com sucesso!")
                st.subheader("Parâmetros GEV e Curva IDF")

                st.subheader("Parâmetros ajustados da Curva IDF")

                # Filtrar e renomear parâmetros desejados
                parametros_filtrados = {
                    "IDF a": parametros["IDF_a"],
                    "IDF b": parametros["IDF_b"],
                    "IDF c": parametros["IDF_c"],
                    "IDF d": parametros["IDF_d"],
                    "R² (ajuste)": parametros["R2"],
                    "Estatística KS": parametros["KS_Statistic"],
                    "KS p-value": parametros["KS_p_value"],
                    "Estatística AD": parametros["AD_Statistic"],
                    "Nº de anos": parametros["Num_Anos"]
                }

                # Exibir como tabela vertical
                df_parametros = pd.DataFrame.from_dict(parametros_filtrados, orient="index", columns=["Valor"])
                st.table(df_parametros.style.format(precision=4))

                # Recuperar parâmetros da equação
                a = parametros['IDF_a']
                b = parametros['IDF_b']
                c = parametros['IDF_c']
                d = parametros['IDF_d']

                # Exibir a equação IDF ajustada
                st.subheader("Equação IDF ajustada")

                eq_latex = (
                    r"I(t, T) = \frac{%.3f \cdot T^{%.3f}}{(t + %.3f)^{%.3f}}"
                    % (a, b, c, d)
                )
                st.latex(eq_latex)


                df_intensidade = pd.DataFrame(
                    matriz,
                    index=[f'TR_{tr}' for tr in trs],
                    columns=[f'{d}min' for d in duracoes]
                )
                st.subheader("Matriz de Intensidade (mm/h)")
                st.dataframe(df_intensidade)

                # Gráfico IDF
                st.subheader("Curvas IDF (Intensidade x Duração)")
                a = parametros['IDF_a']
                b = parametros['IDF_b']
                c = parametros['IDF_c']
                d = parametros['IDF_d']

                def idf_eq(t, T, a, b, c, d):
                    return a * (T**b) * ((t + c) ** -d)

                fig, ax = plt.subplots(figsize=(8, 5))
                duracao_plot = np.linspace(5, 1440, 300)
                for TR in trs:
                    i_tr = idf_eq(duracao_plot, TR, a, b, c, d)
                    ax.plot(duracao_plot, i_tr, label=f'TR = {TR} anos')

                ax.set_xlabel("Duração (min)")
                ax.set_ylabel("Intensidade (mm/h)")
                ax.set_title(f"Curvas IDF - {parametros['Arquivo']} - {municipio}")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

                # Exportações
                st.subheader("Exportar resultados")

                grafico_buffer = BytesIO()
                fig.savefig(grafico_buffer, format="png")
                grafico_buffer.seek(0)
                st.download_button(
                    label="📥 Baixar gráfico IDF (.png)",
                    data=grafico_buffer,
                    file_name=f"curva_IDF_{parametros['Arquivo'].replace('.xlsx','')}.png",
                    mime="image/png"
                )

                intensidade_buffer = BytesIO()
                df_intensidade.to_excel(intensidade_buffer, index=True)
                intensidade_buffer.seek(0)
                st.download_button(
                    label="📥 Baixar matriz de intensidade (.xlsx)",
                    data=intensidade_buffer,
                    file_name=f"matriz_intensidade_{parametros['Arquivo'].replace('.xlsx','')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.warning("Software ainda em desenvolvimento para o processamento desse tipo de dado")