import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def tanh(x):
    return np.tanh(x)

# --- Carregamento dos Dados ---
with open('dados_treinamento_TDNN.pkl', 'rb') as arquivo:
    epocas_max, p_finais, erros_por_epoca, x_train, x_test, atraso = pickle.load(arquivo)

# --- Fase de TESTE ---
print('\n\n--- Iniciando a Fase de TESTES ---\n\n')

arquivo = open("Dados_teste.pkl", 'wb')

dados_completos = np.vstack((x_train, x_test))

d_estimado_testes = [[] for _ in range(3)]
erro_medio_relativo = [[0, 0, 0] for _ in range(3)]
variancia_erro = [[0, 0, 0] for _ in range(3)]
mse_percentual = [[0, 0, 0] for _ in range(3)]
acuracia_percentual = [[0, 0, 0] for _ in range(3)]

# --- Plot da série completa ---
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(dados_completos)), dados_completos, 'k-', label='Dados Completos (Normalizados)')
plt.title('Série Temporal de Dados de Água Normalizados')
plt.xlabel('Amostras')
plt.ylabel('Valor Normalizado')
plt.grid(True)
plt.legend()

for topologia_idx in range(3):
    print(f'\n######## TOPOLOGIA {topologia_idx + 1} ########\n')

    np_atraso = atraso[topologia_idx]

    temp_X_all = []
    temp_d_all = []
    for i in range(np_atraso - 1, len(dados_completos)):
        janela_completa = dados_completos[i - (np_atraso - 1): i + 1].flatten()
        temp_X_all.append(janela_completa)
        temp_d_all.append(dados_completos[i])

    temp_X_all = np.array(temp_X_all)
    temp_d_all = np.array(temp_d_all).flatten()

    idx_inicio_teste = 184 - (np_atraso - 1)
    stop_idx_teste = idx_inicio_teste + len(dados_completos) - 184

    Xt = temp_X_all[idx_inicio_teste: stop_idx_teste]
    dt = temp_d_all[idx_inicio_teste: stop_idx_teste]

    if len(Xt) != len(dt):
        min_len = min(len(Xt), len(dt))
        Xt = Xt[:min_len]
        dt = dt[:min_len]

    Xt_with_bias = np.hstack((-np.ones((Xt.shape[0], 1)), Xt))
    d_estimado_testes[topologia_idx] = np.zeros((3, len(dt)))

    for teste_laco_idx in range(3):
        print(f'\n----- TESTE {teste_laco_idx + 1} -----')

        if p_finais[topologia_idx][teste_laco_idx] is None:
            print(f"Aviso: Pesos não encontrados. Pulando Teste {teste_laco_idx + 1}.")
            continue

        pesos_finais_treinamento = p_finais[topologia_idx][teste_laco_idx]

        # FORWARD PASS
        entrada_E1_teste = pesos_finais_treinamento['W1'] @ Xt_with_bias.T
        saida_Y1_teste = tanh(entrada_E1_teste)

        saida_Y1_with_bias_teste = np.vstack((-np.ones((1, saida_Y1_teste.shape[1])), saida_Y1_teste))
        entrada_E2_teste = pesos_finais_treinamento['W2'] @ saida_Y1_with_bias_teste
        saida_Y2_teste = tanh(entrada_E2_teste)

        saida_Y2_with_bias_teste = np.vstack((-np.ones((1, saida_Y2_teste.shape[1])), saida_Y2_teste))
        entrada_E3_teste = pesos_finais_treinamento['W3'] @ saida_Y2_with_bias_teste
        saida_Y3_teste = entrada_E3_teste

        y_pred = saida_Y3_teste.flatten()
        d_estimado_testes[topologia_idx][teste_laco_idx, :] = y_pred

        # --- Cálculo do Erro Percentual com base em valor fixo ---
        valor_referencia = np.max(np.abs(dt))
        if valor_referencia == 0:
            valor_referencia = 1e-8  # segurança contra divisão por zero

        erro_percentual = np.abs(dt - y_pred) / valor_referencia * 100

        # --- MÉTRICAS ---
        erro_medio_relativo[topologia_idx][teste_laco_idx] = np.mean(erro_percentual)
        variancia_erro[topologia_idx][teste_laco_idx] = np.var(erro_percentual)
        mse_percentual[topologia_idx][teste_laco_idx] = np.mean(erro_percentual ** 2)

        # Acurácia baseada em erro ≤ 10%
        tolerancia = 10  # em %
        acertos = np.sum(erro_percentual <= tolerancia)
        acuracia_percentual[topologia_idx][teste_laco_idx] = (acertos / len(erro_percentual)) * 100

        # --- PRINTS ---
        print(f'Erro relativo médio: {erro_medio_relativo[topologia_idx][teste_laco_idx]:.2f}%')
        print(f'Variância do erro relativo: {variancia_erro[topologia_idx][teste_laco_idx]:.4f}')
        print(f'Erro quadrático médio (MSE %): {mse_percentual[topologia_idx][teste_laco_idx]:.2f}%')
        print(f'Acurácia (erro ≤ 10%): {acuracia_percentual[topologia_idx][teste_laco_idx]:.2f}%')

        # --- Plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(dt, 'k-', linewidth=1.5, label='Real')
        plt.plot(y_pred, 'r-', linewidth=1, label='Estimado')
        plt.title(f'Topologia {topologia_idx + 1} - Teste {teste_laco_idx + 1}')
        plt.xlabel('Amostras')
        plt.ylabel('Saída')
        plt.legend()
        plt.grid(True)

# --- Resumo Visual dos Melhores Treinamentos ---
plt.figure(figsize=(12, 10))
plt.suptitle('Resumo dos Melhores Treinamentos por Topologia', fontsize=16)

for topologia_idx in range(3):
    erros_validos = [e for e in erro_medio_relativo[topologia_idx] if e != 0]
    if not erros_validos:
        print(f"Sem resultados válidos para Topologia {topologia_idx + 1}.")
        continue

    melhor_idx = np.argmin(erro_medio_relativo[topologia_idx])
    plt.subplot(3, 1, topologia_idx + 1)
    plt.plot(dt, 'k--', linewidth=1, label='Real')
    plt.plot(d_estimado_testes[topologia_idx][melhor_idx, :], 'r-', linewidth=1, label='Estimado')
    plt.title(f'Topologia {topologia_idx + 1} - Melhor Treinamento (T{melhor_idx + 1})')
    plt.xlabel('Amostras')
    plt.ylabel('Saída')
    plt.xlim([0, len(dt)])
    plt.ylim([-1, 2])
    plt.legend()
    plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

print("\n--- Todos os plots gerados. ---")