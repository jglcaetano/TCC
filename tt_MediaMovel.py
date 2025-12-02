import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Carregando seus dados ---
file_path = "dados_agua.csv"
if not os.path.exists(file_path):
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()

# Lê o CSV (sem cabeçalho)
dados = pd.read_csv(file_path, header=None)

# Normalização [0,1]
min_val = np.min(dados)
max_val = np.max(dados)
range_val = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8
dados = (dados - min_val) / range_val

# Pega apenas a primeira coluna (consumo de água)
consumo = dados.iloc[:, 0].values

# --- Divisão em treino (70%) e teste (30%) ---
n = len(consumo)
n_treino = int(n * 0.7)

treino = consumo[:n_treino]
teste = consumo[n_treino:]

# --- Previsão com Média Móvel ---
janela = 5
previsoes = []

# Para cada ponto no teste, prevê usando a média dos últimos "janela" valores do treino+previstos
historico = list(treino)
for t in range(len(teste)):
    if len(historico) < janela:
        yhat = np.mean(historico)
    else:
        yhat = np.mean(historico[-janela:])
    previsoes.append(yhat)
    historico.append(teste[t])  # aqui usamos os dados reais do teste para atualizar a janela

# --- Plotando resultados ---
plt.figure(figsize=(12,6))
plt.plot(range(n), consumo, label="Consumo Real", linewidth=2)
plt.plot(range(n_treino, n), previsoes, label="Previsão (Média Móvel)", linewidth=2, color="red")
plt.axvline(n_treino, color="black", linestyle="--", label="Início do Teste")
plt.title("Previsão de Consumo de Água - Média Móvel")
plt.xlabel("Amostras")
plt.ylabel("Consumo Normalizado")
plt.legend()
plt.grid(True)
plt.show()
