#!/usr/bin/env python
# coding: utf-8

# ![imovel.png](attachment:imovel.png)

# # 1. Introdução
# 
# Para o presente projeto, fora selecionado o *Dataset* disponível em [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/), que contém os dados históricos do mercado imobiliário entre o ano de 2012 e 2013, do distrito de Sindian, na cidade de New Taipei em Taiwan. Cada registro descreve a transação de uma propriedade imobiliária. O conjunto de dados contém 414 registros de vendas de imóveis. **Serão aplicados os algoritmos de Regressão Linear e Random Forest para prever o custo do metro quadrado das casas.**

# ## 1.1. Descrição das Variáveis
# 
# | **Variável** |                                 **Descrição**                             |
# |:------------:|:-------------------------------------------------------------------------:|
# |    **No**    |                    Código de identificação do registro.                   |
# |    **X1**    |                         Data de transação da casa.                        |
# |    **X2**    |                           Idade da casa em anos.                          |
# |    **X3**    | Distância da casa até a estação de metrô mais próxima. (medida em metros) |
# |    **X4**    |                    Quantidade de lojas próximas à casa.                   |
# |    **X5**    |                      Coordenadas de latitude da casa.                     |
# |    **X6**    |                     Coordenadas de longitude da casa.                     |
# |     **Y**    |                 Preço do metro quadrado da casa. (variável *target*)      |

# ## <center> ----------------------------------------------------------------------------------------- </center>
# 

# # 2. Importação das Bibliotecas

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## <center> ----------------------------------------------------------------------------------------- </center>
# 

# # 3. Carregamento do Dataset

# In[2]:


df = pd.read_csv('real_valuation.csv', sep=';')


# ## <center> ----------------------------------------------------------------------------------------- </center>
# 

# # 4. Visualização Geral dos Dados do DataFrame

# In[3]:


# Visualização das primeiras linhas do DataFrame (DF)
df.head(3)


# In[4]:


# Tamanho da base de dados (414 linhas por 8 colunas)
df.shape


# In[5]:


# Informações básicas do DF
df.info()


# In[6]:


# Visualização das variáveis númericas do DF (Apenas a X4)
df.describe()


# In[7]:


# Visualização das variáveis categóricas do DF
df.describe(include='O')


# ## <center> ----------------------------------------------------------------------------------------- </center>
# 

# # 5. Tratamento dos Dados

# ## 5.1. Exclusão da Coluna "No"

# In[8]:


df.drop(['No'], axis=1, inplace=True)


# In[9]:


df.head(3)


# ## 5.2. Renomeação das Colunas do DataFrame

# In[10]:


colunas = ['DateTransaction', 'AgeHouse', 'DistanceMRT', 'NumberStores', 'Lat', 'Long', 'Price']
df.columns = colunas


# In[11]:


# Visualização das primeiras linhas do DF após o tratamento
df.head(3)


# ## 5.3. Transformação das Variáveis em Float

# In[12]:


for coluna in colunas:   
    df[coluna] = df[coluna].astype(str)
    df[coluna] = df[coluna].str.replace(',', '.')
    df[coluna] = df[coluna].astype(float)


# ## <center> ----------------------------------------------------------------------------------------- </center>
# 

# # 6. Exploração dos Dados

# In[13]:


# Visualização geral da correlação das variáveis
sns.pairplot(df);


# In[14]:


# Visualização da correlação das colunas no dataframe de forma mais detalhada
plt.figure(figsize=(15, 10))
sns.set(font_scale=1.25)
sns.heatmap(df.corr(), annot=True, cbar=False, linewidths=2.5, cmap='Reds_r');


# ## <center> ----------------------------------------------------------------------------------------- </center>
# 

# # 7. Machine Learning

# ## 7.1. Linear Regression

# ### 7.1.1. Linear Regression Simples - Selecionando a variável com maior correspondência (DistanceMRT) e aplicando uma Regressão Linear Simples.

# In[15]:


# Divisão das variáveis entre explanatória e target
X_regressaoSimples = pd.DataFrame(df['DistanceMRT'])
y_regressaoSimples = df['Price']


# In[16]:


# Divisão entre treino e teste
X_regressaoTreinamento, X_regressaoTeste, y_regressaoTreinamento, y_regressaoTeste = train_test_split(X_regressaoSimples,
                                                                                                      y_regressaoSimples,
                                                                                                      test_size=0.2,
                                                                                                      random_state=0)


# In[17]:


# Construção e treinamento do modelo
modeloRegressaoSimples = LinearRegression()
modeloRegressaoSimples.fit(X_regressaoTreinamento, y_regressaoTreinamento);


# In[18]:


# Valores Resultantes
valorB0 = modeloRegressaoSimples.intercept_
valorB1 = modeloRegressaoSimples.coef_

print(f'Valor do Intercept (B0): {valorB0}\nValor de B1: {valorB1[0]}')


# In[19]:


# Realizando a previsão na base de testes
previsaoModeloRegressaoSimples = modeloRegressaoSimples.predict(X_regressaoTeste)


# In[20]:


# Verificando as métricas
MAE = metrics.mean_absolute_error(y_regressaoTeste, previsaoModeloRegressaoSimples)
MSE = metrics.mean_squared_error(y_regressaoTeste, previsaoModeloRegressaoSimples)
RMSE = np.sqrt(MSE)

print(f'MAE: {MAE}')
print(f'MSE: {MSE}')
print(f'RMSE: {RMSE}')


# In[21]:


plt.figure(figsize=(15, 10))
plt.scatter(X_regressaoTreinamento, y_regressaoTreinamento, color='blue')
plt.plot(X_regressaoTreinamento, modeloRegressaoSimples.predict(X_regressaoTreinamento), color='red');


# In[22]:


casa20Metros = modeloRegressaoSimples.predict([[20]])
casa1000Metros = modeloRegressaoSimples.predict([[1000]])
casa6000Metros = modeloRegressaoSimples.predict([[6000]])

print(f'Uma casa à distância de 20m da estação de metro custará cerca de {np.round(casa20Metros, 2)[0]} o m².')
print(f'Uma casa à distância de 1km da estação de metro custará cerca de {np.round(casa1000Metros, 2)[0]} o m².')
print(f'Uma casa à distância de 6km da estação de metro custará cerca de {np.round(casa6000Metros, 2)[0]} o m².')


# In[23]:


test_residuals = y_regressaoTeste - previsaoModeloRegressaoSimples
plt.figure(figsize=(15, 10))
sns.scatterplot(x=y_regressaoTeste, y=test_residuals, color='g')
plt.axhline(y=0, color='r', ls='-');


# ### 7.1.2. Linear Regression Multiple

# In[24]:


# Divisão das variáveis entre explanatórias e target
X_regressao = df.drop(['DateTransaction', 'Price'], axis=1)
y_regressao = df['Price']


# In[25]:


# Escalonamento dos valores
scaler = preprocessing.StandardScaler().fit(X_regressao)
X_regressao = scaler.transform(X_regressao)


# In[26]:


# Divisão entre treino e teste
X_regressaoTreinamento, X_regressaoTeste, y_regressaoTreinamento, y_regressaoTeste = train_test_split(X_regressao,
                                                                                                      y_regressao,
                                                                                                      test_size=0.2,
                                                                                                      random_state=0)


# In[27]:


# Construção e treinamento do modelo
modeloRegressaoMultipla = LinearRegression()
modeloRegressaoMultipla.fit(X_regressaoTreinamento, y_regressaoTreinamento);


# In[28]:


# Valores Resultantes
valorB0 = modeloRegressaoMultipla.intercept_
valorB1 = modeloRegressaoMultipla.coef_

print(f'Valor do Intercept (B0): {valorB0}')
print(f'Valores dos coeficientes: {valorB1}')


# In[29]:


# Realizando a previsão na base de testes
previsaoModeloRegressaoMultipla = modeloRegressaoMultipla.predict(X_regressaoTeste)


# In[30]:


# Verificando as métricas
MAE = metrics.mean_absolute_error(y_regressaoTeste, previsaoModeloRegressaoMultipla)
MSE = metrics.mean_squared_error(y_regressaoTeste, previsaoModeloRegressaoMultipla)
RMSE = np.sqrt(MSE)

print(f'MAE: {MAE}')
print(f'MSE: {MSE}')
print(f'RMSE: {RMSE}')


# In[31]:


plt.figure(figsize=(15, 10))
sns.scatterplot(x=y_regressaoTeste, y=previsaoModeloRegressaoMultipla)
plt.xlabel('Y-Test')
plt.ylabel('Y-Pred');


# In[32]:


test_residuals = y_regressaoTeste - previsaoModeloRegressaoMultipla
plt.figure(figsize=(15, 10))
sns.scatterplot(x=y_regressaoTeste, y=test_residuals, color='g')
plt.axhline(y=0, color='r', ls='-');


# # 7.2. Random Forest

# ### 7.2.1. Random Forest Simples - Selecionando a variável com maior correspondência (DistanceMRT)

# In[33]:


# Divisão das variáveis entre explanatória e target
X = np.array(df['DistanceMRT']).reshape(-1, 1)
y = np.array(df['Price']).reshape(-1, 1)


# In[34]:


# Dividindo o Dataset entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[35]:


# Construção e treinamento do modelo
RandomForestModel = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
RandomForestModel.fit(X_train, y_train);


# In[36]:


# Verificando o score sobre a base de teste
RandomForestModel.score(X_test, y_test)


# In[37]:


# Realizando a previsão na base de testes
previsoes = RandomForestModel.predict(X_test)


# In[38]:


# Verificando as métricas
MAE = metrics.mean_absolute_error(y_test, previsoes)
MSE = metrics.mean_squared_error(y_test, previsoes)
RMSE = np.sqrt(MSE)

print(f'MAE: {MAE}')
print(f'MSE: {MSE}')
print(f'RMSE: {RMSE}')


# In[39]:


casa20Metros = RandomForestModel.predict([[20]])
casa1000Metros = RandomForestModel.predict([[1000]])
casa6000Metros = RandomForestModel.predict([[6000]])

print(f'Uma casa à distância de 20m da estação de metro custará cerca de {np.round(casa20Metros, 2)[0]} o m².')
print(f'Uma casa à distância de 1km da estação de metro custará cerca de {np.round(casa1000Metros, 2)[0]} o m².')
print(f'Uma casa à distância de 6km da estação de metro custará cerca de {np.round(casa6000Metros, 2)[0]} o m².')


# ### 7.2.2. Random Forest

# In[40]:


# Divisão das variáveis entre explanatórias e target
X_all = df.drop(['DateTransaction', 'Price'], axis=1)
y_all = df['Price']


# In[41]:


# Dividindo o Dataset entre treino e teste
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size = 0.2, random_state = 0)


# In[42]:


# Construção e treinamento do modelo
RandomForestModelAll = RandomForestRegressor(n_estimators = 250, max_depth=5, random_state=0)
RandomForestModelAll.fit(X_train_all, y_train_all);


# In[43]:


# Verificando o score sobre a base de teste
RandomForestModelAll.score(X_test_all, y_test_all)


# In[44]:


# Realizando a previsão na base de testes
previsoes_all = RandomForestModelAll.predict(X_test_all)


# In[45]:


# Verificando as métricas

MAE = metrics.mean_absolute_error(y_test_all, previsoes_all)
MSE = metrics.mean_squared_error(y_test_all, previsoes_all)
RMSE = np.sqrt(MSE)

print(f'MAE: {MAE}')
print(f'MSE: {MSE}')
print(f'RMSE: {RMSE}')


# In[46]:


casa1 = [5.6,90.45606,9.0,24.97433,121.54310]
casa1Predict = RandomForestModelAll.predict([casa1])[0]

print(f'A casa 1 custará cerca de {np.round(casa1Predict, 2)}')


# ## <center> ----------------------------------------------------------------------------------------- </center>
# 

# # 8. Conclusões
# 
# * A variável "DistanceMRT" (Distância da casa até a estação de metrô mais próxima. (medida em metros)) e a variável "Price" foram as que tiveram uma maior correlação. Uma correlação moderada negativa, ou seja, quanto **maior** a distância até uma estação de metrô **menor** será o preço do imóvel;
# 
# 
# * O algoritmo que obteve o melhor desempenho em relação às métricas foi o Random Forest, utilizando todas as variáveis exceto a 'DateTransaction', com os seguintes parâmetros: (n_estimators = 250, max_depth=5), alcançado um valor de erro absoluto em cerca de 5;
# 
# 
# * O algoritmo que obteve o pior resultado em relação às métricas doi o Linear Regression, utilizando apenas a variável "DistanceMRT", embora seja a variável com a maior correlação (negativa), a correlação não é forte o suficiente para alcançarmos resultados satisfatórios, uma vez que o seu MSE foi de quase 90.

# ## <center> ----------------------------------------------------------------------------------------- </center>
# 

# |  	| Contatos 	|  	|
# |:---:	|:---:	|:---:	|
# | <img width=40 align='center' alt='Thiago Ferreira' src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" /> 	| <img width=40 align='center' src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" /> 	| <img width=40 align='center' src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/facebook/facebook-original.svg" /> 	|
# | [Linkedin](https://www.linkedin.com/in/tferreirasilva/) 	| [Github](https://github.com/ThiagoFerreiraWD) 	| [Facebook](https://www.facebook.com/thiago.ferreira.50746) 	|
# |  	| Autor: **Thiago Ferreira** 	|  	|
