from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt


''' 
--------------------------------------------------------------------------------
                       PARTE 1: GERAÇÃO DA BASE DE DADOS 
--------------------------------------------------------------------------------

Houve a alteração do dataset make_moons para make_circles do scikit-learn. 
Assim, o presente trabalho se trata de um problema de classificação entre elementos 
que estão dispersos como que em 2 círculos, um dentro do outro. Foi adicionado um
ruído para que os círculos não ficassem exatos.

D = 2 * y - 1 é utilizado para converter os rótulos, deixando-os entre -1 a 1 que
é a saída da tanh (função de ativação escolhida).

Foram separadas as 100 primeiras amostras para treinar a rede e as 100 seguintes 
para medir o resultado.

'''

X, y = make_circles(n_samples=500, noise=0.06, factor=0.5, random_state=42)

D = 2 * y - 1

N_TREINO = 100      
N_TESTE = 100  



''' 
--------------------------------------------------------------------------------
        PARTE 2: CÁLCULO DAS FUNÇÕES DE ATIVAÇÃO E RESPECTIVAS DERIVADAS 
--------------------------------------------------------------------------------

A seguir foram definidas 3 funções de ativação para realizar a comparação:
sigmoide, tanh e ReLU. A partir da comparação de resultados, após os testes, 
a função de ativação tanh foi priorizada por obter melhores resultados. 

Logo, ela será utilizada nas camadas ocultas. No neurônio de saída também foi 
utilizada a tanh, com os rótulos convertidos para -1 e 1.

Também são definidas as derivadas de cada função, necessárias para o cálculo dos 
gradientes no backward. As derivadas estão escritas em função da saída do neurônio.

'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def tanh_derivada(y):
    return 1 - y ** 2

def relu(x):
    return max(0.0, x)

def relu_derivada(y):
    return 1.0 if y > 0 else 0.0


''' 
--------------------------------------------------------------------------------
      PARTE 3: REDE NEURAL: 2 ENTRADAS > 3 OCULTOS > 2 OCULTOS > 1 SAÍDA
--------------------------------------------------------------------------------

A estrutura da rede foi definida por meio de testes e simulações através do 
Playground TensorFlow (https://playground.tensorflow.org). Assim, foi possível 
identificar que a estrutura com 2 camadas ocultas, uma com 3 neurônios e outra 
com 2 neurônios, chegava a uma classificação bem sucedida.

Então, tendo uma proposta como base, o passo seguinte foi implementar as funções 
da rede neural com base no exercicio4.ipynb disponibilizado em aula.

A função rodar_rede_neural realiza apenas o forward e retorna a classe prevista 
(0 ou 1). Já a função rede_neural realiza o forward, calcula o erro e a loss, e 
em seguida o backward, retornando os gradientes de cada peso e bias, que serão 
usados no treinamento.

'''

def rodar_rede_neural(x, w0, b0, w1, b1, w2, b2, f):
    
    # Camada oculta 1: 3 neurônios
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y0 = f(v0)

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y1 = f(v1)

    s20 = w0[2, 0] * x[0]
    s21 = w0[2, 1] * x[1]
    s22 = s20 + s21
    v2 = s22 + b0[2]
    y2 = f(v2)

    # Camada oculta 2: 2 neurônios
    s30 = w1[0, 0] * y0
    s31 = w1[0, 1] * y1
    s32 = w1[0, 2] * y2
    s33 = s30 + s31 + s32
    v3 = s33 + b1[0]
    y3 = f(v3)

    s40 = w1[1, 0] * y0
    s41 = w1[1, 1] * y1
    s42 = w1[1, 2] * y2
    s43 = s40 + s41 + s42
    v4 = s43 + b1[1]
    y4 = f(v4)

    # Saída: 1 neurônio, com ativação tanh
    s50 = w2[0] * y3
    s51 = w2[1] * y4
    s52 = s50 + s51
    v5 = s52 + b2[0]
    y5 = tanh(v5)
    return 1 if y5 > 0 else 0


def rede_neural(x, d, w0, b0, w1, b1, w2, b2, f, df):
    
    # Forward
    # Camada oculta 1: 3 neurônios
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y0 = f(v0)

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y1 = f(v1)

    s20 = w0[2, 0] * x[0]
    s21 = w0[2, 1] * x[1]
    s22 = s20 + s21
    v2 = s22 + b0[2]
    y2 = f(v2)

    # Camada oculta 2: 2 neurônios
    s30 = w1[0, 0] * y0
    s31 = w1[0, 1] * y1
    s32 = w1[0, 2] * y2
    s33 = s30 + s31 + s32
    v3 = s33 + b1[0]
    y3 = f(v3)

    s40 = w1[1, 0] * y0
    s41 = w1[1, 1] * y1
    s42 = w1[1, 2] * y2
    s43 = s40 + s41 + s42
    v4 = s43 + b1[1]
    y4 = f(v4)

    # Saída: 1 neurônio, com ativação tanh
    s50 = w2[0] * y3
    s51 = w2[1] * y4
    s52 = s50 + s51
    v5 = s52 + b2[0]
    y5 = tanh(v5)

    e = y5 - d
    L = 1/2 * (e ** 2)


    # Backward
    grad_w0 = np.zeros(w0.shape)
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    grad_b0 = np.zeros(b0.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_b2 = np.zeros(b2.shape)

    grad_L = 1
    grad_e = grad_L * e

    # Saída
    grad_y5 = grad_e
    grad_v5 = grad_y5 * tanh_derivada(y5)
    grad_b2[0] = grad_v5
    grad_s52 = grad_v5
    grad_s50 = grad_s52
    grad_s51 = grad_s52
    grad_w2[0] = grad_s50 * y3
    grad_w2[1] = grad_s51 * y4
    grad_y3 = grad_s50 * w2[0]
    grad_y4 = grad_s51 * w2[1]

    # Camada oculta 2: 2 neurônios
    grad_v3 = grad_y3 * df(y3)
    grad_v4 = grad_y4 * df(y4)
    grad_b1[0] = grad_v3
    grad_b1[1] = grad_v4

    grad_s33 = grad_v3
    grad_s30 = grad_s33
    grad_s31 = grad_s33
    grad_s32 = grad_s33
    grad_w1[0, 0] = grad_s30 * y0
    grad_w1[0, 1] = grad_s31 * y1
    grad_w1[0, 2] = grad_s32 * y2

    grad_s43 = grad_v4
    grad_s40 = grad_s43
    grad_s41 = grad_s43
    grad_s42 = grad_s43
    grad_w1[1, 0] = grad_s40 * y0
    grad_w1[1, 1] = grad_s41 * y1
    grad_w1[1, 2] = grad_s42 * y2

    # Cada neurônio da camada 1 alimenta os 2 neurônios da camada 2,então o gradiente dele é a soma dos dois caminhos
    grad_y0 = grad_s30 * w1[0, 0] + grad_s40 * w1[1, 0]
    grad_y1 = grad_s31 * w1[0, 1] + grad_s41 * w1[1, 1]
    grad_y2 = grad_s32 * w1[0, 2] + grad_s42 * w1[1, 2]

    # Camada oculta 1: 3 neurônios
    grad_v0 = grad_y0 * df(y0)
    grad_v1 = grad_y1 * df(y1)
    grad_v2 = grad_y2 * df(y2)
    grad_b0[0] = grad_v0
    grad_b0[1] = grad_v1
    grad_b0[2] = grad_v2

    grad_s02 = grad_v0
    grad_s12 = grad_v1
    grad_s22 = grad_v2

    grad_s00 = grad_s02
    grad_s01 = grad_s02
    grad_s10 = grad_s12
    grad_s11 = grad_s12
    grad_s20 = grad_s22
    grad_s21 = grad_s22
    grad_w0[0, 0] = grad_s00 * x[0]
    grad_w0[0, 1] = grad_s01 * x[1]
    grad_w0[1, 0] = grad_s10 * x[0]
    grad_w0[1, 1] = grad_s11 * x[1]
    grad_w0[2, 0] = grad_s20 * x[0]
    grad_w0[2, 1] = grad_s21 * x[1]
    return grad_w0, grad_b0, grad_w1, grad_b1, grad_w2, grad_b2, L


''' 
--------------------------------------------------------------------------------
         PARTE 4: AVALIAÇÃO DAS MÉTRICAS DE ACURÁCIA, LOSS E ERRO
--------------------------------------------------------------------------------

Foi definida também uma função para retornar a acurácia, loss e erro, permitindo
a avaliação posterior.

'''

def avalia(inicio, fim, w0, b0, w1, b1, w2, b2, f, df):
    acc = 0
    loss = 0
    for i in range(inicio, fim):
        out = rodar_rede_neural(X[i], w0, b0, w1, b1, w2, b2, f)
        if out == y[i]:
            acc += 1
        *_, L = rede_neural(X[i], D[i], w0, b0, w1, b1, w2, b2, f, df)
        loss += L
    n = fim - inicio
    acuracia = acc / n
    erro = 1 - acuracia
    return acuracia, loss / n, erro


''' 
--------------------------------------------------------------------------------
                    PARTE 5: TREINAMENTO DA REDE NEURAL
--------------------------------------------------------------------------------

O passo seguinte foi realizar o treinamento da rede. Em cada época, são somados 
os gradientes das 100 amostras de treino e os pesos são atualizados por gradiente 
descendente. O número de épocas (8000) também foi ajustado por tentativa e erro, 
com o objetivo de melhorar a acurácia e reduzir o erro.

Também é armazenada a loss de cada época (soma das losses das amostras de treino)
 para possibilitar a plotagem de um gráfico ao final da execução.

Durante o treinamento, a cada 1000 épocas também são impressos na tela a época e loss
para acompanhamento da variação dos dados.

'''

def treina(f, df, taxa, epocas, seed=0, mostrar=False):
    # Inicialização aleatória (seed fixa para comparar as ativações de forma justa)
    np.random.seed(seed)
    w0 = np.random.randn(3, 2)
    w1 = np.random.randn(2, 3)
    w2 = np.random.randn(2)
    b0 = np.zeros(3)
    b1 = np.zeros(2)
    b2 = np.zeros(1)
    pesos_iniciais = (w0.copy(), b0.copy(), w1.copy(), b1.copy(), w2.copy(), b2.copy())
    historico_loss = []     

    # Gradiente descendente
    for i in range(epocas):
        loss = 0

        grad_w0 = np.zeros(w0.shape)
        grad_w1 = np.zeros(w1.shape)
        grad_w2 = np.zeros(w2.shape)
        grad_b0 = np.zeros(b0.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_b2 = np.zeros(b2.shape)

        for k in range(N_TREINO):
            g_w0, g_b0, g_w1, g_b1, g_w2, g_b2, L = rede_neural(X[k], D[k], w0, b0, w1, b1, w2, b2, f, df)

            grad_w0 += g_w0
            grad_w1 += g_w1
            grad_w2 += g_w2
            grad_b0 += g_b0
            grad_b1 += g_b1
            grad_b2 += g_b2
            loss += L

        w0 -= taxa * grad_w0
        w1 -= taxa * grad_w1
        w2 -= taxa * grad_w2
        b0 -= taxa * grad_b0
        b1 -= taxa * grad_b1
        b2 -= taxa * grad_b2

        historico_loss.append(loss)


        if mostrar and i % 1000 == 0:
            print("Época: ", i, "Loss:", loss)

    return pesos_iniciais, (w0, b0, w1, b1, w2, b2), historico_loss


''' 
--------------------------------------------------------------------------------
                    PARTE 6: PLOTAGEM DE GRÁFICOS
--------------------------------------------------------------------------------

Nesta etapa foram realizadas as plotagens dos gráficos para análise da fronteira
de classificação inicial e também após o treinamento.

Outro gráfico que também será utilizado é o de exibição da variação da loss ao 
longo das épocas.

'''

def plota_fronteira(ax, pesos, f, titulo, acuracia):
    inicio, fim = N_TREINO, N_TREINO + N_TESTE
    Xt, yt = X[inicio:fim], y[inicio:fim]

    # grade de pontos cobrindo o gráfico; a rede classifica cada um deles
    xx, yy = np.meshgrid(np.linspace(-1.4, 1.4, 300), np.linspace(-1.4, 1.4, 300))
    zz = np.array([rodar_rede_neural((a, b), *pesos, f)
      for a, b in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=['purple', 'green'], alpha=0.2)
    ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=1.5)
    ax.scatter(Xt[yt == 0, 0], Xt[yt == 0, 1], c='purple', label='Classe 0 (externo)')
    ax.scatter(Xt[yt == 1, 0], Xt[yt == 1, 1], c='green', label='Classe 1 (interno)')
    ax.set_title(f"{titulo}\n Acurácia {acuracia:.0%}")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)




def plota_loss(historicos, titulo, arquivo):
    plt.figure(figsize=(8, 5))
    for nome, historico in historicos.items():
        plt.plot(historico, label=nome)
    plt.yscale('log')       
    plt.xlabel('época')
    plt.ylabel(f'loss (soma nas {N_TREINO} amostras de treino)')
    plt.title(titulo)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(arquivo)
    plt.show()


''' 
--------------------------------------------------------------------------------
                 PARTE 7: FUNÇÃO PRINCIPAL - ORQUESTRAÇÃO
--------------------------------------------------------------------------------

Ao final, são definidas a taxa de aprendizagem, as épocas e o intervalo de dados
de teste. Também é impressa uma visão geral das alterações realizadas na rede neural
proposta no exercício.

Em seguida, é realizada uma comparação da fronteira de acordo com a classificação
inicial e após o treinamento para dar um panorama de como foi o desempenho da 
rede neural.

Também é plotado o gráfico da loss para a função de ativação selecionada (tanh)
e depois realizada a comparação com diferentes funções de ativação, utilizando
as funções definidas inicialmente para sigmoide, tanh e ReLU e suas respectivas
derivadas.

Por fim, a partir dos dados obtidos, é possível observar que a rede neural com a 
estrutura 2 ENTRADAS > 3 OCULTOS > 2 OCULTOS > 1 SAÍDA conseguiu classificar 
corretamente todos os dados de teste com as três funções de ativação. A tanh 
obteve a menor loss, indicando maior confiança nas previsões, enquanto a 
sigmoide foi a que convergiu mais lentamente.

'''


def main():
    taxa = 0.01
    epocas = 8000
    teste = (N_TREINO, N_TREINO + N_TESTE)

    print("---------------- DADOS GERAIS DA REDE NEURAL COM ALTERAÇÕES ----------------")
    print("Neurônios de entrada:       2")
    print("Camadas ocultas:            2 (alterado)") # alterado de 1 para 2 camadas ocultas
    print("Neurônios na oculta 1:      3 (alterado)") # alterado de 2 para 3 neurônios
    print("Neurônios na oculta 2:      2 (alterado)") # alterado para 
    print("Neurônios de saída:         1")
    print("Ativação das ocultas:       tanh (alterado)") # alterada função de ativação nas camadas ocultas
    print("Ativação da saída:          tanh (alterado)") # alterada função de ativação no neurônio de saída
    print("Taxa de aprendizado:       ", taxa)
    print()

    iniciais, finais, historico = treina(tanh, tanh_derivada, taxa, epocas, mostrar=True)

    print()
    print("---------------- ANÁLISE DA FRONTEIRA DE CLASSIFICAÇÃO ANTES E DEPOIS DA APLICAÇÃO ----------------")
    acc_antes, loss_antes, erro_antes = avalia(*teste, *iniciais, tanh, tanh_derivada)
    acc_depois, loss_depois, erro_depois = avalia(*teste, *finais, tanh, tanh_derivada)
    print(f"\nAntes do treino:  Acurácia {acc_antes:.0%} | Loss {loss_antes:.4f} | Erro {erro_antes:.0%}")
    print(f"Depois do treino: Acurácia {acc_depois:.0%} | Loss {loss_depois:.4f} | Erro {erro_depois:.0%}")
    print()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    plota_fronteira(ax1, iniciais, tanh, "Antes do treinamento", acc_antes)
    plota_fronteira(ax2, finais, tanh, "Depois do treinamento", acc_depois)
    fig.suptitle("Rede 2-3-2-1 com tanh")
    plt.tight_layout()
    plt.savefig('img/comparacao_fronteira_antes_depois.png')
    plt.show()
    print()
    
    print("---------------- PLOTAGEM DA LOSS DA FUNÇÃO DE ATIVAÇÃO SELECIONADA: TANH ----------------")
    plota_loss({'tanh': historico}, 'Loss ao longo das épocas: Rede 2-3-2-1 com tanh', 'img/loss_tanh.png')

    print()
    
    print("---------------- PLOTAGEM DA COMPARAÇÃO DE LOSS ENTRE DIFERENTES FUNÇÕES DE ATIVAÇÃO ----------------")
    # comparação: mesma rede e mesmos pesos iniciais, mudando só a ativação das ocultas
    historicos = {}
    for nome, f, df in [('Sigmoide:', sigmoid, sigmoid_derivada),
                        ('Tanh:', tanh, tanh_derivada),
                        ('ReLU:', relu, relu_derivada)]:
        _, pesos, historicos[nome] = treina(f, df, taxa, epocas)
        acc, loss, erro = avalia(*teste, *pesos, f, df)
        print(f"{nome:9s} Acurácia {acc:.0%} | Loss {loss:.4f} | Erro {erro:.0%}")
    



    plota_loss(historicos, 'Loss ao longo das épocas: Comparação das ativações', 'img/loss_comparacao_sig_tanh_relu.png')


main()