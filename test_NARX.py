import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def sigmoide(x, b):
    x = np.array(x)
    return 1 / (1 + np.exp(-b * x))

def dsigmoide(x, b):
    s = sigmoide(x, b)
    return b * s * (1 - s)

# --- Carregamento dos dados ---
with open('dados_treinamento_NARX2.pkl', 'rb') as arquivo:
    epocas_max, p_finais, erros_por_epoca, x_train, x_test, atraso, b, min_val, max_val = pickle.load(arquivo)

print('\n\n--- Iniciando a Fase de TESTES ---\n\n')

dados_completos = np.vstack((x_train, x_test))

# Variáveis para armazenar os resultados do teste
d_estimado_testes = [np.zeros((3, len(x_test) - 1)) for _ in range(3)]
erro_medio_relativo = [[0, 0, 0] for _ in range(3)]
variancia_erro = [[0, 0, 0] for _ in range(3)]
mse_percentual = [[0, 0, 0] for _ in range(3)]
acuracia_percentual = [[0, 0, 0] for _ in range(3)]
rmse_metric = [[0, 0, 0] for _ in range(3)]
mape_metric = [[0, 0, 0] for _ in range(3)]

# --- Inicialização do Erro Global ---
erro_global_total = 0
num_amostras_total = 0

# --- Plot da série completa ---
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(dados_completos)), dados_completos, 'k-', label='Dados Completos (Normalizados)')
plt.title('Série Temporal de Dados de Água Normalizados (Completo)')
plt.xlabel('Amostras')
plt.ylabel('Valor Normalizado')
plt.grid(True)
plt.legend()
plt.show(block=False)

# --- Loop pelas Topologias ---
for topologia_idx in range(3):
    print(f'\n######## TOPOLOGIA {topologia_idx + 1} ########\n')

    np_atraso = atraso[topologia_idx]

    X_all_windowed = []
    d_all_targets = []
    for i in range(np_atraso - 1, len(dados_completos)):
        janela = dados_completos[i - (np_atraso - 1) : i + 1].flatten()
        X_all_windowed.append(janela)
        d_all_targets.append(dados_completos[i])

    X_all_windowed = np.array(X_all_windowed)
    d_all_targets = np.array(d_all_targets).flatten()

    start_test_idx = 184 - (np_atraso - 1)
    end_test_idx = start_test_idx + len(dados_completos) - 184

    Xt_test = X_all_windowed[start_test_idx : end_test_idx]
    dt_test = d_all_targets[start_test_idx : end_test_idx]

    Xt_test_with_bias = np.hstack((-np.ones((Xt_test.shape[0], 1)), Xt_test))
    d_estimado_testes[topologia_idx] = np.zeros((3, len(dt_test)))

    # --- Loop pelos Testes ---
    for teste_laco_idx in range(3):
        print(f'----- TESTE {teste_laco_idx + 1} -----\n')

        if p_finais[topologia_idx][teste_laco_idx] is None:
            print(f"Aviso: Pesos para Topologia {topologia_idx + 1}, Treinamento {teste_laco_idx + 1} não encontrados. Pule o teste.")
            erro_medio_relativo[topologia_idx][teste_laco_idx] = np.nan
            variancia_erro[topologia_idx][teste_laco_idx] = np.nan
            mse_percentual[topologia_idx][teste_laco_idx] = np.nan
            acuracia_percentual[topologia_idx][teste_laco_idx] = np.nan
            rmse_metric[topologia_idx][teste_laco_idx] = np.nan
            mape_metric[topologia_idx][teste_laco_idx] = np.nan
            continue

        pesos_finais_treinamento = p_finais[topologia_idx][teste_laco_idx]
        saidas_estimadas_teste = []
        current_feedback_test = 0.0

        for amostra_idx in range(len(Xt_test)):
            current_input_features_test = Xt_test_with_bias[amostra_idx, :].reshape(-1, 1)
            input_to_L1_test = np.vstack((current_input_features_test, current_feedback_test))

            entrada_E1_test = pesos_finais_treinamento['W1'] @ input_to_L1_test
            saida_Y1_test = sigmoide(entrada_E1_test, b)

            input_to_L2_test = np.vstack((-np.ones((1, 1)), saida_Y1_test))
            entrada_E2_test = pesos_finais_treinamento['W2'] @ input_to_L2_test
            saida_Y2_test = sigmoide(entrada_E2_test, b)

            input_to_L3_test = np.vstack((-np.ones((1, 1)), saida_Y2_test))
            entrada_E3_test = pesos_finais_treinamento['W3'] @ input_to_L3_test
            saida_Y3_test = sigmoide(entrada_E3_test, b)

            predicted_value = saida_Y3_test.item()
            saidas_estimadas_teste.append(predicted_value)
            current_feedback_test = predicted_value

        d_estimado_testes[topologia_idx][teste_laco_idx, :] = np.array(saidas_estimadas_teste)

        # --- Cálculo das métricas ---
        valor_referencia = np.max(np.abs(dt_test))
        if valor_referencia == 0:
            valor_referencia = 1e-8

        erro_percentual = np.abs(dt_test - np.array(saidas_estimadas_teste)) / valor_referencia * 100

        erro_medio_relativo[topologia_idx][teste_laco_idx] = np.mean(erro_percentual)
        variancia_erro[topologia_idx][teste_laco_idx] = np.var(erro_percentual)
        mse_percentual[topologia_idx][teste_laco_idx] = np.mean(erro_percentual ** 2)

        tolerancia = 10
        acertos = np.sum(erro_percentual <= tolerancia)
        acuracia_percentual[topologia_idx][teste_laco_idx] = (acertos / len(erro_percentual)) * 100

        # --- Cálculo adicional: RMSE e MAPE ---
        dt_test_arr = np.array(dt_test)
        saidas_arr = np.array(saidas_estimadas_teste)
        mse = np.mean((dt_test_arr - saidas_arr) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((dt_test_arr - saidas_arr) / (dt_test_arr + 1e-8))) * 100

        rmse_metric[topologia_idx][teste_laco_idx] = rmse
        mape_metric[topologia_idx][teste_laco_idx] = mape

        # --- Atualização do Erro Global ---
        erro_global_total += np.sum((dt_test - np.array(saidas_estimadas_teste)) ** 2)
        num_amostras_total += len(dt_test)

        print(f'Erro relativo médio: {erro_medio_relativo[topologia_idx][teste_laco_idx]:.2f}%')
        print(f'Variância do erro relativo: {variancia_erro[topologia_idx][teste_laco_idx]:.4f}')
        print(f'Erro quadrático médio (MSE %): {mse_percentual[topologia_idx][teste_laco_idx]:.2f}%')
        print(f'Acurácia (erro ≤ 10%): {acuracia_percentual[topologia_idx][teste_laco_idx]:.2f}%')
        print(f'RMSE: {rmse:.6f}')
        print(f'MAPE: {mape:.2f}%')

        # --- Plot individual ---
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(dt_test)), dt_test, 'k-', linewidth=1.5, label='Real')
        plt.plot(np.arange(len(dt_test)), np.array(saidas_estimadas_teste), 'r-', linewidth=1, label='Estimado')
        plt.title(f'Topologia {topologia_idx + 1} - Teste {teste_laco_idx + 1}')
        plt.xlabel('Amostras')
        plt.ylabel('Saída')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

# --- Resumo dos melhores treinamentos por topologia ---
plt.figure(figsize=(12, 10))
plt.suptitle('Resumo dos Melhores Treinamentos por Topologia - NARX', fontsize=16)

for topologia_idx in range(3):
    valid_errors = [e for e in erro_medio_relativo[topologia_idx] if not np.isnan(e)]
    if not valid_errors:
        print(f"Não há resultados válidos para a Topologia {topologia_idx + 1} para plotar o melhor treinamento.")
        continue

    melhor_treinamento_idx = np.argmin(erro_medio_relativo[topologia_idx])

    plt.subplot(3, 1, topologia_idx + 1)
    plt.plot(np.arange(len(dt_test)), dt_test, 'k--', linewidth=1, label='Real')
    plt.plot(np.arange(len(dt_test)), d_estimado_testes[topologia_idx][melhor_treinamento_idx, :],
             'r-', linewidth=1, label='Estimado')
    plt.xlabel('Amostras')
    plt.ylabel('Saída')
    plt.xlim([0, len(dt_test)])
    plt.title(f'Topologia {topologia_idx + 1} - Melhor Treinamento (T{melhor_treinamento_idx + 1})')
    plt.legend()
    plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Impressão organizada de todos os erros por treinamento ---
for topologia_idx in range(3):
    print(f'\n=== Topologia {topologia_idx + 1} ===')
    for teste_laco_idx in range(3):
        if np.isnan(erro_medio_relativo[topologia_idx][teste_laco_idx]):
            print(f'Treinamento {teste_laco_idx + 1}: Pesos não encontrados')
        else:
            print(f'Treinamento {teste_laco_idx + 1}: Erro médio = {erro_medio_relativo[topologia_idx][teste_laco_idx]:.2f}%, '
                  f'Variância = {variancia_erro[topologia_idx][teste_laco_idx]:.4f}, '
                  f'MSE % = {mse_percentual[topologia_idx][teste_laco_idx]:.2f}%, '
                  f'Acurácia (≤10%) = {acuracia_percentual[topologia_idx][teste_laco_idx]:.2f}%, '
                  f'RMSE = {rmse_metric[topologia_idx][teste_laco_idx]:.6f}, '
                  f'MAPE = {mape_metric[topologia_idx][teste_laco_idx]:.2f}%')

# --- Cálculo final do Erro Global ---
if num_amostras_total > 0:
    erro_global = erro_global_total / num_amostras_total
    print(f'\n--- Erro Global (MSE total) de todas as topologias e testes: {erro_global:.6f} ---')
else:
    print("Nenhuma amostra válida para cálculo do erro global.")

print("\n--- Todos os plots e métricas gerados. ---")
