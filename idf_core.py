import os
import numpy as np
import pandas as pd
from scipy.stats import genextreme as gev, kstest, anderson
from scipy.optimize import curve_fit

def processar_idf(df, df_coef, municipio, nome_arquivo="estacao", pasta_saida="resultados_streamlit"):
    os.makedirs(pasta_saida, exist_ok=True)

    tempos_retorno = [2, 5, 10, 25, 50, 100]
    colunas_coef = ['P24/dia', '720min', '360min', '240min', '120min', '60min',
                    '45min', '30min', '20min', '15min', '10min']
    duracoes_min = [1440, 720, 360, 240, 120, 60, 45, 30, 20, 15, 10]

    # Uniformizar nomes de colunas
    df.columns = df.columns.str.strip().str.lower()

    colunas_esperadas = {
        'estacaocodigo': 'EstacaoCodigo',
        'data': 'Data',
        'maxima': 'Maxima'
    }

    faltando = [col for col in colunas_esperadas if col not in df.columns]
    if faltando:
        return f"Colunas ausentes na planilha: {faltando}"

    df = df.rename(columns=colunas_esperadas)

    #Conversão de data
    df['Data'] = df['Data'].astype(str).str.strip()
    df['Data'] = pd.to_datetime(df['Data'], format="%d/%m/%Y", errors='coerce')

    #Conversão da coluna Maxima (vírgula para ponto)
    df['Maxima'] = df['Maxima'].astype(str).str.replace(',', '.').str.strip()
    df['Maxima'] = pd.to_numeric(df['Maxima'], errors='coerce')

    #Limpeza
    df = df.dropna(subset=['Data', 'Maxima'])
    df['Ano'] = df['Data'].dt.year

    #Cálculo dos máximos anuais
    df_anuais = df.groupby('Ano')['Maxima'].max().reset_index()
    dados_anuais = df_anuais['Maxima'].values.astype(float)

    if len(dados_anuais) < 3:
        return "Dados insuficientes para ajuste GEV."

    #Ajuste GEV
    shape, loc, scale = gev.fit(dados_anuais)
    D_ks, p_ks = kstest(dados_anuais, 'genextreme', args=(shape, loc, scale))
    ad_stat = anderson(dados_anuais, dist='gumbel_r').statistic

    #Precipitações para cada tempo de retorno
    precipitacoes_diarias = [
        loc - scale * np.log(-np.log(1 - 1 / T)) if shape == 0
        else loc + (scale / shape) * ((-np.log(1 - 1 / T)) ** (-shape) - 1)
        for T in tempos_retorno
    ]

    #Coeficientes de desagregação
    linha_municipio = df_coef[df_coef.iloc[:, 0] == municipio]
    if linha_municipio.empty:
        return f"Município '{municipio}' não encontrado na planilha de coeficientes."

    try:
        coeficientes = linha_municipio.iloc[0][colunas_coef].astype(str).str.replace(',', '.').astype(float).values
    except Exception as e:
        return f"Erro ao converter coeficientes de desagregação: {e}"

    chuvas_duracoes = [[P * coef for coef in coeficientes] for P in precipitacoes_diarias]
    intensidades = np.array(chuvas_duracoes) / (np.array(duracoes_min) / 60)

    # Ajuste da curva IDF
    T_grid, t_grid = np.meshgrid(tempos_retorno, duracoes_min, indexing='ij')
    I = intensidades.flatten()
    T = T_grid.flatten()
    t = t_grid.flatten()

    def idf_eq(t, T, a, b, c, d):
        return a * (T**b) * ((t + c) ** -d)

    def idf_model(xdata, a, b, c, d):
        t, T = xdata
        return idf_eq(t, T, a, b, c, d)

    popt, _ = curve_fit(
        idf_model, (t, T), I,
        p0=[1, 0.2, 10, 0.7],
        bounds=([0, -5, 0, 0], [1e4, 5, 500, 5]),
        maxfev=10000
    )
    a, b, c, d = popt

    I_pred = idf_eq(t, T, a, b, c, d)
    ss_res = np.sum((I - I_pred) ** 2)
    ss_tot = np.sum((I - np.mean(I)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    parametros = {
        'Arquivo': nome_arquivo,
        'Shape': shape, 'Loc': loc, 'Scale': scale,
        'KS_Statistic': D_ks, 'KS_p_value': p_ks,
        'AD_Statistic': ad_stat,
        'IDF_a': a, 'IDF_b': b, 'IDF_c': c, 'IDF_d': d,
        'R2': r_squared,
        'Num_Anos': len(dados_anuais)
    }

    return parametros, intensidades, duracoes_min, tempos_retorno
