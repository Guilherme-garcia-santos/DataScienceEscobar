# -*- coding: utf-8 -*-
"""
MÓDULO DE INFERÊNCIA - Clusterização K-Means da base Boston Housing
Avaliação do 2º bimestre - Data Science

Etapas:
  1. Receber os dados de uma nova residência
  2. Normalizar os dados da nova residência (mesmos imputadores e normalizador do treinamento)
  3. Inferir a qual grupo (cluster) a nova residência pertence
  4. Exibir os dados desnormalizados do centroide do grupo inferido

Uso:
  python inferencia.py               -> usa a residência de exemplo definida abaixo
  python inferencia.py --interativo  -> pergunta cada atributo no terminal
"""
import sys
import numpy as np
import pandas as pd
from pickle import load

PASTA_MODELOS = 'modelos'

# Carregar os modelos salvos pelo módulo de treinamento
imputador_num = load(open(f'{PASTA_MODELOS}/imputador_num.model', 'rb'))
imputador_cat = load(open(f'{PASTA_MODELOS}/imputador_cat.model', 'rb'))
normalizador = load(open(f'{PASTA_MODELOS}/normalizador_boston.model', 'rb'))
kmeans = load(open(f'{PASTA_MODELOS}/kmeans_boston.model', 'rb'))
estrutura_colunas = load(open(f'{PASTA_MODELOS}/estrutura_colunas.model', 'rb'))

colunas_originais = estrutura_colunas['colunas_originais']
colunas_quantitativas = estrutura_colunas['colunas_quantitativas']
colunas_categoricas = estrutura_colunas['colunas_categoricas']
colunas_dummies = estrutura_colunas['colunas_dummies']
colunas_modelo = estrutura_colunas['colunas_modelo']

DESCRICAO = {
    'CRIM': 'taxa de criminalidade per capita',
    'ZN': '% de terrenos residenciais com lotes acima de 25.000 pés²',
    'INDUS': '% de área ocupada por negócios não varejistas',
    'CHAS': 'faz limite com o rio Charles (1 = sim, 0 = não)',
    'NOX': 'concentração de óxido nítrico (partes por 10 milhões)',
    'RM': 'número médio de cômodos por residência',
    'AGE': '% de unidades construídas antes de 1940',
    'DIS': 'distância ponderada até 5 centros de emprego de Boston',
    'RAD': 'índice de acesso a rodovias radiais',
    'TAX': 'imposto sobre a propriedade (por US$ 10.000)',
    'PTRATIO': 'proporção aluno/professor',
    'B': '1000(Bk - 0,63)², sendo Bk a proporção de população negra',
    'LSTAT': '% da população de baixa renda',
    'MEDV': 'valor mediano das residências (em US$ 1.000)',
}

# 1 RECEBER OS DADOS DE UMA NOVA RESIDÊNCIA
# Residência de exemplo (altere os valores à vontade). Use None para um atributo
# desconhecido: ele será preenchido com a mediana (quantitativo) ou a moda (categórico)
# aprendidas no treinamento.
nova_residencia = {
    'CRIM': 0.1, 'ZN': 20.0, 'INDUS': 6.0, 'CHAS': 0, 'NOX': 0.45, 'RM': 6.5,
    'AGE': 40.0, 'DIS': 5.0, 'RAD': 4, 'TAX': 300, 'PTRATIO': 17.0, 'B': 390.0,
    'LSTAT': 8.0, 'MEDV': 26.0,
}


def receber_residencia_interativa():
    """Pergunta cada atributo no terminal. Enter em branco = valor desconhecido."""
    residencia = {}
    print('Informe os dados da nova residência (Enter em branco se não souber o valor):')
    for coluna in colunas_originais:
        while True:
            texto = input(f'  {coluna} - {DESCRICAO[coluna]}: ').strip().replace(',', '.')
            if texto == '':
                residencia[coluna] = None
                break
            try:
                residencia[coluna] = float(texto)
                break
            except ValueError:
                print('    Valor inválido, digite um número.')
    return residencia


# 2 NORMALIZAR OS DADOS DA NOVA RESIDÊNCIA
def normalizar_residencia(residencia):
    """Aplica à nova residência exatamente o mesmo tratamento feito no treinamento:
    imputação de ausentes, MinMaxScaler nos quantitativos e one-hot nos categóricos.
    Usa-se sempre transform() (nunca fit), para manter a escala aprendida no treinamento."""
    df = pd.DataFrame([residencia]).reindex(columns=colunas_originais).astype(float)

    # 2.1 quantitativos: imputar ausentes e normalizar com o modelo normalizador salvo
    num = pd.DataFrame(imputador_num.transform(df[colunas_quantitativas]), columns=colunas_quantitativas)
    num = pd.DataFrame(normalizador.transform(num), columns=colunas_quantitativas)

    # 2.2 categóricos: imputar ausentes e aplicar one-hot com as mesmas colunas do treinamento
    #     (o reindex garante que todas as dummies existam, inclusive a da categoria não escolhida)
    cat = pd.DataFrame(imputador_cat.transform(df[colunas_categoricas]), columns=colunas_categoricas).astype(int)
    cat = pd.get_dummies(cat, columns=colunas_categoricas, dtype=int).reindex(columns=colunas_dummies, fill_value=0)

    # 2.3 recompor na mesma ordem de colunas usada no treinamento
    return pd.concat([num, cat], axis=1)[colunas_modelo]


# 4 DESNORMALIZAR UM CENTROIDE
def desnormalizar_centroide(centroide):
    """Converte um centroide (que está na escala normalizada) de volta para a escala original."""
    centroide = pd.DataFrame([centroide], columns=colunas_modelo)

    # quantitativos: inverse_transform do MinMaxScaler
    # (o clip remove resíduos de ponto flutuante, como -0.000, mantendo o centroide na faixa observada)
    num = pd.DataFrame(np.clip(normalizador.inverse_transform(centroide[colunas_quantitativas]),
                               normalizador.data_min_, normalizador.data_max_),
                       columns=colunas_quantitativas)

    # categóricos: no centroide cada dummy guarda a proporção da categoria dentro do grupo
    # (ex.: CHAS_0 = 0,93 e CHAS_1 = 0,07). A categoria do centroide é a de maior proporção:
    # monta-se o one-hot dessa categoria e desfaz-se com from_dummies.
    one_hot = pd.DataFrame(0, index=[0], columns=colunas_dummies)
    for coluna in colunas_categoricas:
        dummies = [d for d in colunas_dummies if d.startswith(coluna + '_')]
        one_hot.loc[0, centroide[dummies].iloc[0].idxmax()] = 1
    cat = pd.from_dummies(one_hot, sep='_').astype(int)

    return pd.concat([num, cat], axis=1)[colunas_originais].iloc[0]


if __name__ == '__main__':
    # 1 receber os dados
    if '--interativo' in sys.argv:
        nova_residencia = receber_residencia_interativa()

    print('\n=== NOVA RESIDÊNCIA (dados originais) ===')
    for coluna in colunas_originais:
        valor = nova_residencia.get(coluna)
        print(f'  {coluna:8s} = {"desconhecido" if valor is None else valor}')

    # 2 normalizar
    residencia_normalizada = normalizar_residencia(nova_residencia)
    print('\n=== NOVA RESIDÊNCIA (dados normalizados) ===')
    print(residencia_normalizada.round(4).to_string(index=False))

    # 3 inferir o grupo
    grupo = int(kmeans.predict(residencia_normalizada)[0])
    distancias = kmeans.transform(residencia_normalizada)[0]
    print(f'\n=== GRUPO INFERIDO: {grupo} ===')
    print('Distância da nova residência a cada centroide (escala normalizada):')
    for i, distancia in enumerate(distancias):
        print(f'  grupo {i}: {distancia:.4f}' + ('   <-- mais próximo' if i == grupo else ''))

    # 4 exibir o centroide desnormalizado
    centroide = desnormalizar_centroide(kmeans.cluster_centers_[grupo])
    print(f'\n=== CENTROIDE DO GRUPO {grupo} (dados desnormalizados) ===')
    for coluna in colunas_originais:
        if coluna in colunas_categoricas:
            valor = f'{int(centroide[coluna]):>10d}'
        else:
            valor = f'{centroide[coluna]:>10.3f}'
        print(f'  {coluna:8s} = {valor}   {DESCRICAO[coluna]}')
