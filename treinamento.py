# -*- coding: utf-8 -*-
"""
MÓDULO DE TREINAMENTO - Clusterização K-Means da base Boston Housing
Avaliação do 2º bimestre - Data Science

Etapas:
  1. Carregar a base
  2. Separar os atributos categóricos dos quantitativos
  3. Tratar os valores ausentes (NA)
  4. Normalizar a base (MinMaxScaler nos quantitativos e one-hot nos categóricos)
  5. Determinar o número ótimo de grupos com o método do cotovelo (distâncias)
  6. Treinar o K-Means com o número ótimo de grupos e salvar o modelo

Obs.: não há necessidade de balancear a base. A clusterização é não supervisionada,
portanto não existe uma classe a ser balanceada.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # gera o gráfico direto em arquivo, sem abrir janela
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from pickle import dump

PASTA_MODELOS = 'modelos'
PASTA_RESULTADOS = 'resultados'
os.makedirs(PASTA_MODELOS, exist_ok=True)
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# 1 CARREGAR OS DADOS
dados = pd.read_csv('dados/HousingData.csv')
print(f'Base carregada: {dados.shape[0]} residências x {dados.shape[1]} atributos')
ausentes = dados.isna().sum()
print('Valores ausentes por coluna:')
print(ausentes[ausentes > 0].to_string())

# 2 SEPARAR OS DADOS CATEGÓRICOS DOS QUANTITATIVOS
# CHAS é a única coluna categórica da base (1 = o imóvel faz limite com o rio Charles,
# 0 = não faz). As demais colunas são quantitativas. RAD é um índice ordinal de acesso
# a rodovias (quanto maior, mais acesso), por isso é tratado como quantitativo.
colunas_categoricas = ['CHAS']
colunas_quantitativas = [c for c in dados.columns if c not in colunas_categoricas]

dados_num = dados[colunas_quantitativas]
dados_cat = dados[colunas_categoricas]

# 3 TRATAR OS VALORES AUSENTES
# O K-Means não aceita valores ausentes (NA). Cada coluna quantitativa recebe a sua
# mediana e cada coluna categórica recebe a sua moda (valor mais frequente).
# Os imputadores são salvos para que a inferência trate uma nova residência da mesma forma.
imputador_num = SimpleImputer(strategy='median').fit(dados_num)
imputador_cat = SimpleImputer(strategy='most_frequent').fit(dados_cat)
dump(imputador_num, open(f'{PASTA_MODELOS}/imputador_num.model', 'wb'))
dump(imputador_cat, open(f'{PASTA_MODELOS}/imputador_cat.model', 'wb'))

dados_num = pd.DataFrame(imputador_num.transform(dados_num), columns=colunas_quantitativas)
dados_cat = pd.DataFrame(imputador_cat.transform(dados_cat), columns=colunas_categoricas).astype(int)

# 4 NORMALIZAR OS DADOS
# 4.1 Quantitativos: o MinMaxScaler leva cada coluna para a escala de 0 a 1, assim nenhum
#     atributo domina o cálculo das distâncias só por ter valores maiores (ex.: TAX x NOX).
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)
# 4.2 Salvar o modelo normalizador para uso na inferência
dump(modelo_normalizador, open(f'{PASTA_MODELOS}/normalizador_boston.model', 'wb'))
# 4.3 Normalizar os dados quantitativos
dados_num_normalizados = pd.DataFrame(modelo_normalizador.transform(dados_num),
                                      columns=colunas_quantitativas)

# 4.4 Categóricos: one-hot encoding (get_dummies). Cada categoria vira uma coluna 0/1,
#     ou seja, já fica na mesma escala dos quantitativos normalizados. Não se aplica
#     MinMaxScaler em categoria, pois os códigos não têm ordem nem distância entre si.
dados_cat_normalizados = pd.get_dummies(dados_cat, columns=colunas_categoricas, dtype=int)
colunas_dummies = list(dados_cat_normalizados.columns)

# 4.5 Recompor a base normalizada (quantitativos + categóricos)
dados_normalizados = pd.concat([dados_num_normalizados, dados_cat_normalizados], axis=1)
colunas_modelo = list(dados_normalizados.columns)

# 4.6 Salvar a estrutura das colunas, para a inferência montar a nova residência na mesma ordem
estrutura_colunas = {
    'colunas_originais': list(dados.columns),
    'colunas_quantitativas': colunas_quantitativas,
    'colunas_categoricas': colunas_categoricas,
    'colunas_dummies': colunas_dummies,
    'colunas_modelo': colunas_modelo,
}
dump(estrutura_colunas, open(f'{PASTA_MODELOS}/estrutura_colunas.model', 'wb'))

# 5 DETERMINAR O NÚMERO ÓTIMO DE GRUPOS - MÉTODO DO COTOVELO (DISTÂNCIAS)
# Para cada quantidade de grupos k treina-se um K-Means e mede-se a inércia: a soma das
# distâncias quadráticas de cada residência ao centroide do seu grupo. A inércia sempre
# diminui quando k aumenta; o "cotovelo" é o ponto a partir do qual aumentar k deixa de
# trazer ganho relevante na redução das distâncias.
K_MINIMO, K_MAXIMO = 1, 15
valores_k = list(range(K_MINIMO, K_MAXIMO + 1))
inercias = []
for k in valores_k:
    kmeans_k = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_k.fit(dados_normalizados)
    inercias.append(kmeans_k.inertia_)


def encontrar_cotovelo(valores_k, inercias):
    """Localiza o cotovelo de forma automática: é o ponto da curva k x inércia mais
    distante da reta que liga o primeiro ao último ponto (eixos normalizados para 0..1)."""
    x = np.array(valores_k, dtype=float)
    y = np.array(inercias, dtype=float)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    ponto_inicial = np.array([x[0], y[0]])
    ponto_final = np.array([x[-1], y[-1]])
    reta = (ponto_final - ponto_inicial) / np.linalg.norm(ponto_final - ponto_inicial)
    vetores = np.column_stack([x, y]) - ponto_inicial
    projecoes = np.outer(vetores @ reta, reta)
    distancias_reta = np.linalg.norm(vetores - projecoes, axis=1)
    return valores_k[int(np.argmax(distancias_reta))]


k_otimo = encontrar_cotovelo(valores_k, inercias)

print('\nMétodo do cotovelo (k x inércia):')
for k, inercia in zip(valores_k, inercias):
    print(f'  k = {k:2d}   inércia = {inercia:8.3f}' + ('   <-- cotovelo' if k == k_otimo else ''))
print(f'Número ótimo de grupos: {k_otimo}')

# Gráfico do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(valores_k, inercias, marker='o', label='Inércia (soma das distâncias quadráticas)')
plt.axvline(k_otimo, color='red', linestyle='--', label=f'Cotovelo: k = {k_otimo}')
plt.xticks(valores_k)
plt.xlabel('Número de grupos (k)')
plt.ylabel('Inércia')
plt.title('Método do cotovelo - K-Means na base Boston Housing')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PASTA_RESULTADOS}/metodo_cotovelo.png', dpi=120)
pd.DataFrame({'k': valores_k, 'inercia': inercias}).to_csv(f'{PASTA_RESULTADOS}/metodo_cotovelo.csv', index=False)

# 6 TREINAR E SALVAR O MODELO
kmeans = KMeans(n_clusters=k_otimo, n_init=10, random_state=42)
modelo_kmeans = kmeans.fit(dados_normalizados)
dump(modelo_kmeans, open(f'{PASTA_MODELOS}/kmeans_boston.model', 'wb'))
print(f'\nModelo K-Means treinado com {k_otimo} grupos e salvo em {PASTA_MODELOS}/kmeans_boston.model')

# Resumo dos grupos encontrados
dados['GRUPO'] = modelo_kmeans.labels_
dados.to_csv(f'{PASTA_RESULTADOS}/base_com_grupos.csv', index=False)
print('\nResidências por grupo:')
print(dados['GRUPO'].value_counts().sort_index().to_string())

# Centroides desnormalizados (escala original), para conferência
# (o clip remove resíduos de ponto flutuante, como -0.000, mantendo o centroide na faixa observada)
centroides = pd.DataFrame(modelo_kmeans.cluster_centers_, columns=colunas_modelo)
centroides_naturais = pd.DataFrame(
    np.clip(modelo_normalizador.inverse_transform(centroides[colunas_quantitativas]),
            modelo_normalizador.data_min_, modelo_normalizador.data_max_),
    columns=colunas_quantitativas)
for coluna in colunas_categoricas:
    dummies = [d for d in colunas_dummies if d.startswith(coluna + '_')]
    # no centroide, cada dummy guarda a proporção da categoria dentro do grupo;
    # a categoria do centroide é a de maior proporção
    centroides_naturais[coluna] = (centroides[dummies].idxmax(axis=1)
                                   .str.replace(coluna + '_', '', regex=False).astype(int))
centroides_naturais = centroides_naturais[estrutura_colunas['colunas_originais']]
centroides_naturais.index.name = 'GRUPO'
centroides_naturais['RESIDENCIAS'] = dados['GRUPO'].value_counts().sort_index().values
centroides_naturais.to_csv(f'{PASTA_RESULTADOS}/centroides.csv')
print('\nCentroides de cada grupo (dados desnormalizados):')
print(centroides_naturais.round(3).to_string())
