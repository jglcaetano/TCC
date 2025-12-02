import pandas as pd
import numpy as np
import os
import pickle

# --- Funções de Ativação ---
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

# --- Data ---
# Carregando o dataset
file_path = "dados_agua.csv"
if not os.path.exists(file_path):
    print(f"Erro: O arquivo '{file_path}' não foi encontrado. Certifique-se de que ele está no mesmo diretório do script.")
    exit()

dados = pd.read_csv(file_path, header=None)
x_atributos = dados.iloc[:, 0].values.reshape(-1, 1)

# --- Funções de Ativação ---
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

# Dividir os dados em treino e teste
corte = 263
x_treino = x_atributos[:corte, :]
x_teste = x_atributos[corte:, :]

# --- Normalização Min-Max ---
# Normaliza os dados para o intervalo [0, 1]
min_val = np.min(x_treino) # ESTES SÃO OS VALORES CHAVE PARA DESNORMALIZAÇÃO
max_val = np.max(x_treino) # ESTES SÃO OS VALORES CHAVE PARA DESNORMALIZAÇÃO

# Adiciona um pequeno valor ao denominador para evitar divisão por zero
range_val = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8

x_train = (x_treino - min_val) / range_val
x_test = (x_teste - min_val) / range_val

print(f"Shape de x_train após normalização Min-Max: {x_train.shape}")
print(f"Shape de x_test após normalização Min-Max: {x_test.shape}")

# --- Parâmetros ---
# Parâmetros de Treinamento
n = 0.0001 # Taxa de aprendizado ainda menor para maior estabilidade
epsilon = 1e-7 # Tolerância um pouco mais rigorosa
a = 0.08 # Momento um pouco maior para ajudar na convergência
epocas_max = 50000 # Valor máximo de épocas reduzido

# Quantidade de neurônios por camada
neuronios = [20, 30, 40] # Primeira camada escondida
neuronios_camada2 = [10, 20, 30] # Segunda camada escondida

# Atraso da entrada
atraso = [10, 15, 20]

# Listas para armazenar resultados (similar às cell arrays do MATLAB)
p_finais = [[None, None, None] for _ in range(3)] # 3 topologias, 3 treinamentos
erros_por_epoca = [[[] for _ in range(3)] for _ in range(3)] # Erros por época para cada treinamento de cada topologia
epocas_atingidas = [[0, 0, 0] for _ in range(3)] # Épocas atingidas para cada treinamento de cada topologia

# --- Treinamento ---
print('\n--- Iniciando a Fase de Treinamento ---\n')

for topologia_idx in range(3):
    print(f'\n\n--- TOPOLOGIA {topologia_idx + 1} --- \n\n')

    np_atraso = atraso[topologia_idx]

    # Geração de janelas para o conjunto de treinamento
    X_train_tdnn = []
    d_train = []

    for i in range(np_atraso - 1, len(x_train)):
        janela = x_train[i - (np_atraso - 1) : i + 1].flatten()
        X_train_tdnn.append(janela)
        d_train.append(x_train[i])

    X_train_tdnn = np.array(X_train_tdnn)
    d_train = np.array(d_train).flatten()

    # Adicionar o bias (-1) à matriz de entrada X_train_tdnn
    X_train_with_bias = np.hstack((-np.ones((X_train_tdnn.shape[0], 1)), X_train_tdnn))

    # Número de neurônios por camada
    n0 = X_train_with_bias.shape[1]
    n1 = neuronios[topologia_idx]
    n2_hidden = neuronios_camada2[topologia_idx]
    n3 = 1

    print(f'Neste treinamento foram utilizados:')
    print(f'{n0} Neurônios na Camada de entrada (com bias)')
    print(f'{n1} Neurônios na Camada Escondida 1')
    print(f'{n2_hidden} Neurônios na Camada Escondida 2')

    for treinamento_idx in range(3):
        print(f'\n--- TREINAMENTO {treinamento_idx + 1} --- \n')


        limit_W1 = np.sqrt(6 / (n0 + n1))
        limit_W2 = np.sqrt(6 / (n1 + 1 + n2_hidden))
        limit_W3 = np.sqrt(6 / (n2_hidden + 1 + n3))

        pesos = {
            'W1': np.random.uniform(-limit_W1, limit_W1, size=(n1, n0)),
            'W2': np.random.uniform(-limit_W2, limit_W2, size=(n2_hidden, n1 + 1)),
            'W3': np.random.uniform(-limit_W3, limit_W3, size=(n3, n2_hidden + 1))
        }

        p_anterior = {
            'W1': np.copy(pesos['W1']),
            'W2': np.copy(pesos['W2']),
            'W3': np.copy(pesos['W3'])
        }

        epoca = 0
        erro_atual = float('inf')

        erros_por_epoca[topologia_idx][treinamento_idx] = []

        while epoca < epocas_max:
            # --- FORWARD PASS ---
            entrada_E1 = pesos['W1'] @ X_train_with_bias.T
            saida_Y1 = tanh(entrada_E1)

            saida_Y1_with_bias = np.vstack((-np.ones((1, saida_Y1.shape[1])), saida_Y1))
            entrada_E2 = pesos['W2'] @ saida_Y1_with_bias
            saida_Y2 = tanh(entrada_E2)

            saida_Y2_with_bias = np.vstack((-np.ones((1, saida_Y2.shape[1])), saida_Y2))
            entrada_E3 = pesos['W3'] @ saida_Y2_with_bias
            saida_Y3 = entrada_E3

            # --- BACKPROPAGATION ---
            delta3 = (d_train - saida_Y3.flatten())
            delta3 = delta3.reshape(n3, -1)

            aux_W3 = np.copy(pesos['W3'])
            grad_W3 = n * delta3 @ saida_Y2_with_bias.T
            momentum_W3 = a * (pesos['W3'] - p_anterior['W3'])
            pesos['W3'] = pesos['W3'] + grad_W3 + momentum_W3
            p_anterior['W3'] = aux_W3

            # Gradiente CAMADA 2
            delta_aux2 = pesos['W3'].T @ delta3
            delta2 = delta_aux2[1:, :] * dtanh(entrada_E2) # Usando dtanh

            aux_W2 = np.copy(pesos['W2'])
            grad_W2 = n * delta2 @ saida_Y1_with_bias.T
            momentum_W2 = a * (pesos['W2'] - p_anterior['W2'])
            pesos['W2'] = pesos['W2'] + grad_W2 + momentum_W2
            p_anterior['W2'] = aux_W2

            # Gradiente CAMADA 1
            delta_aux1 = pesos['W2'].T @ delta2
            delta1 = delta_aux1[1:, :] * dtanh(entrada_E1) # Usando dtanh

            aux_W1 = np.copy(pesos['W1'])
            grad_W1 = n * delta1 @ X_train_with_bias
            momentum_W1 = a * (pesos['W1'] - p_anterior['W1'])
            pesos['W1'] = pesos['W1'] + grad_W1 + momentum_W1
            p_anterior['W1'] = aux_W1

            epoca += 1

            # Calcular o erro atual (MSE) para o critério de parada
            entrada_E1_calc_erro = pesos['W1'] @ X_train_with_bias.T
            saida_Y1_calc_erro = tanh(entrada_E1_calc_erro)

            saida_Y1_with_bias_calc_erro = np.vstack((-np.ones((1, saida_Y1_calc_erro.shape[1])), saida_Y1_calc_erro))
            entrada_E2_calc_erro = pesos['W2'] @ saida_Y1_with_bias_calc_erro
            saida_Y2_calc_erro = tanh(entrada_E2_calc_erro)

            saida_Y2_with_bias_calc_erro = np.vstack((-np.ones((1, saida_Y2_calc_erro.shape[1])), saida_Y2_calc_erro))
            entrada_E3_calc_erro = pesos['W3'] @ saida_Y2_with_bias_calc_erro
            saida_Y3_calc_erro = entrada_E3_calc_erro # A saída final é linear para o erro

            erro_anterior = erro_atual
            erro_atual = np.mean((d_train - saida_Y3_calc_erro.flatten())**2)
            erros_por_epoca[topologia_idx][treinamento_idx].append(erro_atual)

            if abs(erro_atual - erro_anterior) < epsilon and epoca > 1:
                p_finais[topologia_idx][treinamento_idx] = {
                    'W1': np.copy(pesos['W1']),
                    'W2': np.copy(pesos['W2']),
                    'W3': np.copy(pesos['W3'])
                }
                epocas_atingidas[topologia_idx][treinamento_idx] = epoca
                print('Convergência!')
                break

            if epoca % 5000 == 0: # Imprimir a cada 5000 épocas
                print(f"Época: {epoca}, Erro: {erro_atual:.8f}")

        if epoca == epocas_max:
            print(f"Atingiu o número máximo de épocas ({epocas_max}) sem convergência.")
            p_finais[topologia_idx][treinamento_idx] = {
                'W1': np.copy(pesos['W1']),
                'W2': np.copy(pesos['W2']),
                'W3': np.copy(pesos['W3'])
            }
            epocas_atingidas[topologia_idx][treinamento_idx] = epoca

        print(f'O número de épocas foi de {epoca} épocas')
        print(f'Erro Médio Quadrático foi de {erro_atual:.8e}')

print("\n--- TREINAMENTO CONCLUÍDO ---\n")

with open("dados_treinamento_TDNN.pkl",'wb') as arquivo:
    pickle.dump([epocas_max, p_finais, erros_por_epoca, x_train, x_test, atraso], arquivo)