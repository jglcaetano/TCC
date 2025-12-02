import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle 

# --- Funções de Ativação ---

def sigmoide(x, b):
    x = np.array(x)
    return 1 / (1 + np.exp(-b * x))

def dsigmoide(x, b):
    s = sigmoide(x, b)
    return b * s * (1 - s)

# --- Dados ---

file_path = "dados_agua.csv"
if not os.path.exists(file_path):
    print(f"Erro: O arquivo '{file_path}' não foi encontrado. Certifique-se de que ele está no mesmo diretório do script.")
    exit()

dados = pd.read_csv(file_path, header=None)
x_atributos = dados.iloc[:, 0].values.reshape(-1, 1)

# Divide os dados em treino e teste (seus novos valores)
x_treino_raw = x_atributos[:263, :]
x_teste_raw = x_atributos[263:, :]

# Normalização Min-Max para [0, 1] (mantido como você tinha no Python)
min_val = np.min(x_treino_raw)
max_val = np.max(x_treino_raw)

range_val = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8

x_train = (x_treino_raw - min_val) / range_val
x_test = (x_teste_raw - min_val) / range_val

print(f"Shape de x_train normalizado: {x_train.shape}")
print(f"Shape de x_test normalizado: {x_test.shape}")

# --- Parametros ---

n = 0.01  # Taxa de aprendizado
epsilon = 10**-6  # Precisão
b = 0.5  # Parametro de inclinação da sigmoide
epocas_max = 50000 # Valor máximo de épocas

# Quantidade de neurônios por camada (seus novos valores)
neuronios = [20, 50, 70]
neuronios_camada2 = [35, 65, 85] # Segunda camada escondida

# Atraso da entrada (seus novos valores)
atraso = [5, 15, 20]

# --- Variáveis para armazenar resultados ---
p_finais = [[None, None, None] for _ in range(3)] # 3 topologias, 3 treinamentos
erros_por_epoca = [[[] for _ in range(3)] for _ in range(3)] # Erros por época para cada treinamento
epocas_atingidas = [[0, 0, 0] for _ in range(3)] # Épocas atingidas

# --- Treinamento ---
print('\n--- Iniciando a Fase de Treinamento ---\n')

for topologia_idx in range(3):
    print(f'\n\n--- TOPOLOGIA {topologia_idx + 1} --- \n\n')

    np_atraso = atraso[topologia_idx]

    X_train_windowed = []
    d_train = [] # D_train é o target, que também é normalizado no [0,1]

    for i in range(np_atraso - 1, len(x_train)):
        janela = x_train[i - (np_atraso - 1) : i + 1].flatten()
        X_train_windowed.append(janela)
        d_train.append(x_train[i]) # Target também vem de x_train, então está em [0,1]

    X_train_windowed = np.array(X_train_windowed)
    d_train = np.array(d_train).flatten()


    X_train_with_bias = np.hstack((-np.ones((X_train_windowed.shape[0], 1)), X_train_windowed))

    # Número de neurônios por camada
    n0 = X_train_with_bias.shape[1] + 1 # Camada de Entrada (com bias + o termo feedback)
    n1 = neuronios[topologia_idx]
    n2_hidden = neuronios_camada2[topologia_idx]
    n3 = 1

    print(f'Neste treinamento foram utilizados:')
    print(f'{n0} Neurônios na Camada de entrada (com bias e feedback)')
    print(f'{n1} Neurônios na Camada Escondida 1')
    print(f'{n2_hidden} Neurônios na Camada Escondida 2')

    for treinamento_idx in range(3):
        print(f'\n--- TREINAMENTO {treinamento_idx + 1} --- \n')

        # Inicializando os pesos (mantido randn, mas você pode experimentar Xavier se tiver problemas de novo)
        pesos = {
            'W1': np.random.randn(n1, n0),
            'W2': np.random.randn(n2_hidden, n1 + 1),
            'W3': np.random.randn(n3, n2_hidden + 1)
        }

        # Seu código MATLAB NARX não usava momentum, então não implementamos p_anterior.
        # Se quiser adicionar momentum, descomente e implemente.

        epoca = 0
        erro_atual_epoch = float('inf')

        erros_por_epoca[topologia_idx][treinamento_idx] = []

        # --- Loop de ÉPOCAS ---
        while epoca < epocas_max:
            erro_acumulado_amostras = 0

            # Reset do feedback para cada época no TREINAMENTO, conforme o MATLAB
            feedback_val_train = 0.0

            # --- Loop de AMOSTRAS (treinamento amostra a amostra) ---
            for amostra_idx in range(X_train_with_bias.shape[0]):
                current_input_features = X_train_with_bias[amostra_idx, :].reshape(-1, 1)
                current_d = d_train[amostra_idx]

                # Construir a entrada para a camada 1, incluindo o termo feedback
                input_to_L1 = np.vstack((current_input_features, [[feedback_val_train]])) # Garante que feedback_val_train seja 2D

                # FORWARD
                # Camada 1
                entrada_E1 = pesos['W1'] @ input_to_L1
                saida_Y1 = sigmoide(entrada_E1, b)

                # Camada 2
                input_to_L2 = np.vstack((-np.ones((1, 1)), saida_Y1))
                entrada_E2 = pesos['W2'] @ input_to_L2
                saida_Y2 = sigmoide(entrada_E2, b)

                # Camada 3 (Saída)
                input_to_L3 = np.vstack((-np.ones((1, 1)), saida_Y2))
                entrada_E3 = pesos['W3'] @ input_to_L3
                saida_Y3 = sigmoide(entrada_E3, b) # Usando sigmoide na saída

                # BACKPROPAGATION
                erro_amostra = (current_d - saida_Y3.item())
                delta3 = erro_amostra * dsigmoide(entrada_E3, b).item()
                delta3 = np.array([[delta3]])

                pesos['W3'] = pesos['W3'] + n * delta3 @ input_to_L3.T

                delta_aux2 = pesos['W3'].T @ delta3
                delta2 = delta_aux2[1:, :] * dsigmoide(entrada_E2, b)

                pesos['W2'] = pesos['W2'] + n * delta2 @ input_to_L2.T

                delta_aux1 = pesos['W2'].T @ delta2
                delta1 = delta_aux1[1:, :] * dsigmoide(entrada_E1, b)
                pesos['W1'] = pesos['W1'] + n * delta1 @ input_to_L1.T

                erro_acumulado_amostras += erro_amostra**2

                # **** Se quiser realimentação da rede (NARX closed-loop no treino), descomente abaixo: ****
                # feedback_val_train = saida_Y3.item()

            epoca += 1

            erro_anterior_epoch = erro_atual_epoch
            erro_atual_epoch = erro_acumulado_amostras / X_train_with_bias.shape[0]
            erros_por_epoca[topologia_idx][treinamento_idx].append(erro_atual_epoch)

            if abs(erro_atual_epoch - erro_anterior_epoch) < epsilon and epoca > 1:
                p_finais[topologia_idx][treinamento_idx] = {
                    'W1': np.copy(pesos['W1']),
                    'W2': np.copy(pesos['W2']),
                    'W3': np.copy(pesos['W3'])
                }
                epocas_atingidas[topologia_idx][treinamento_idx] = epoca
                print('Convergência!')
                break

            if epoca % 1000 == 0:
                print(f"Época: {epoca}, Erro: {erro_atual_epoch:.8f}")

        if epoca == epocas_max:
            print(f"Atingiu o número máximo de épocas ({epocas_max}) sem convergência.")
            p_finais[topologia_idx][treinamento_idx] = {
                'W1': np.copy(pesos['W1']),
                'W2': np.copy(pesos['W2']),
                'W3': np.copy(pesos['W3'])
            }
            epocas_atingidas[topologia_idx][treinamento_idx] = epoca

        print(f'O número de épocas foi de {epoca} épocas')
        print(f'Erro Médio Quadrático foi de {erro_atual_epoch:.8e}')

print('\n\n--- O TREINAMENTO ACABOU ---\n')

# Salva os dados de treinamento no arquivo (como você fez)
with open("dados_treinamento_NARX4.pkl", 'wb') as arquivo:
    pickle.dump([epocas_max, p_finais, erros_por_epoca, x_train, x_test, atraso, b, min_val, max_val], arquivo)