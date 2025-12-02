import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "media_movel.csv"
dados = pd.read_csv(file_path, header=None)

# --- Normalização Min-Max ---
# min_val = np.min(dados)
# max_val = np.max(dados)
# range_val = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8
# dados = (dados - min_val) / range_val

## --- Série de consumo ---
#consumo = dados.iloc[:, 0].values  # pega como array
#tempo = np.arange(len(consumo))    # eixo temporal
#
## --- Diferenciação ---
#diferenciada = np.diff(consumo)  
#
## --- Inicialização dos parâmetros ARIMA(1,1,0) simples ---
#phi = 0.5   # autoregressivo
#mu = 0.0    # média
#LR = 0.001  # taxa de aprendizado
#epochs = 500
#
## --- Treinamento manual ---
#for _ in range(epochs):
#    for t in range(1, len(diferenciada)):
#        # Previsão
#        y_pred = mu + phi * diferenciada[t-1]
#        # Erro
#        erro = diferenciada[t] - y_pred
#        # Atualização dos parâmetros
#        mu += LR * erro
#        phi += LR * erro * diferenciada[t-1]
#
## --- Previsão com o modelo treinado ---
#pred_diff = []
#for t in range(1, len(diferenciada)):
#    y_pred = mu + phi * diferenciada[t-1]
#    pred_diff.append(y_pred)
#
## Reconstrução da série original (inverso do diff)
#pred_reconstruida = np.r_[consumo[0], consumo[0] + np.cumsum(pred_diff)] * 1.5
## --- Ajuste dos eixos para não dar erro de dimensão ---
#tempo_pred = tempo[:len(pred_reconstruida)]
#
## --- Plot comparativo ---
#plt.figure(figsize=(12,6))
#plt.plot(tempo, consumo, label="Dados Reais (Normalizados)", color="blue", linewidth=1)
#plt.plot(tempo_pred, pred_reconstruida, label="ARIMA (manual)", color="red", linestyle="--", linewidth=2)
#plt.xlabel("Tempo")
#plt.ylabel("Consumo Normalizado")
#plt.title("Previsão ARIMA Manual vs Dados Reais")
#plt.legend()
#plt.grid(True)
#plt.show()
#

# --- Série de consumo ---
consumo = dados.iloc[:, 0].values
tempo = np.arange(len(consumo))

# --- Ajuste automático do modelo ARIMA ---
modelo = auto_arima(
    consumo,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    seasonal=False,         # coloque True se houver padrão sazonal
    stepwise=True,
    trace=True,             # mostra o progresso no console
    error_action='ignore',
    suppress_warnings=True,
    information_criterion='aic'
)

# --- Exibe resumo do modelo selecionado ---
print(modelo.summary())

# --- Previsão ---
n_forecast = 50  # número de pontos à frente
predicoes = modelo.predict(n_periods=n_forecast)

# --- Eixos para visualização ---
tempo_pred = np.arange(len(consumo), len(consumo) + n_forecast)

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(tempo, consumo, label="Dados Reais", color="blue")
plt.plot(tempo_pred, predicoes, label="Previsão Auto-ARIMA", color="red", linestyle="--")
plt.title("Previsão de Consumo com Auto-ARIMA")
plt.xlabel("Tempo")
plt.ylabel("Consumo")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

