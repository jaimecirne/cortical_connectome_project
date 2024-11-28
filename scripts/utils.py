# scripts/generate_connectomes.py

import pandas as pd
import numpy as np
import networkx as nx
import os


def preprocess_dataFrame(df):
    df = df.replace(',', '.', regex=True)
    df = df.replace('.', np.nan)
    numeric_columns = [
        'LoGammaCoherenceWin0Spike', 'LoGammaCoherenceSignifWin0Spike',
        'LoGammaCoherenceWin1Spike', 'LoGammaCoherenceSignifWin1Spike',
        'LoGammaCoherenceWin2Spike', 'LoGammaCoherenceSignifWin2Spike'
    ]

#    df[numeric_columns] = df[numeric_columns].astype(float).fillna(df[numeric_columns].mean())

    # Verificando valores não numéricos e substituindo por NaN
    cols_to_convert = df.columns.difference(['Record', 'Session', 'Condition'])

    for col in cols_to_convert:
        # Substituir '.' por NaN
        df[col] = df[col].replace('.', np.nan)
        # Converter para float, substituindo qualquer valor não conversível por NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remover colunas completamente vazias após a conversão
    df.dropna(axis=1, how='all', inplace=True)

    # Remover linhas completamente vazias após a conversão
    df.dropna(axis=0, how='all', inplace=True)

    # Verificar se o DataFrame ainda tem dados suficientes
    if df.empty:
        raise ValueError("O DataFrame está vazio após a limpeza. Verifique os dados de entrada.")

    # Divisão em variáveis preditoras (X) e variável alvo (y)
    if 'Condition' not in df.columns:
        raise ValueError("A coluna 'Condition' não está presente no DataFrame. Verifique o arquivo de entrada.")

    df['Condition'] = df['Condition'].astype(str)

    # Unique sessions, conditions, and windows
    sessions = list(df['Session'].unique())
    conditions = [str(x) for x in df['Condition'].unique()]
    windows = ['Win0', 'Win1', 'Win2']

    return sessions, conditions, windows, df