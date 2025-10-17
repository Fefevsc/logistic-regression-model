# logistic_regression_full.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -----------------------------
# 1. Carregar a base de dados
# -----------------------------
df = pd.read_csv("clientes_realista.csv")
print("Primeiras linhas da base:")
print(df.head())

# -----------------------------
# 2. Análise exploratória simples
# -----------------------------
print("\nEstatísticas descritivas:")
print(df.describe())

# Proporção de clientes que compraram
print("\nProporção de Comprou:")
print(df["Comprou"].value_counts(normalize=True))

# Histograma de Tempo_no_site
sns.histplot(df["Tempo_no_site"], bins=10, kde=True)
plt.title("Distribuição do Tempo no Site")
plt.xlabel("Tempo no site (min)")
plt.ylabel("Frequência")
plt.show()

# -----------------------------
# 3. Pré-processamento
# -----------------------------
# Separar variáveis independentes e alvo
X = pd.get_dummies(df.drop("Comprou", axis=1), drop_first=True)
y = df["Comprou"]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 4. Treinamento da Regressão Logística
# -----------------------------
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# Coeficientes do modelo
coef_df = pd.DataFrame({"Variável": X.columns, "Coeficiente": modelo.coef_[0]})
print("\nCoeficientes do modelo:")
print(coef_df)

# -----------------------------
# 5. Avaliação do modelo
# -----------------------------
y_pred = modelo.predict(X_test)
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Visualizar matriz de confusão
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# -----------------------------
# 6. Curva logística (ex: Tempo_no_site)
# -----------------------------
# Selecionar índice da variável Tempo_no_site
idx_tempo = X.columns.get_loc("Tempo_no_site")
x_vals = np.linspace(df["Tempo_no_site"].min(), df["Tempo_no_site"].max(), 100)
prob = 1 / (1 + np.exp(-(modelo.intercept_[0] + modelo.coef_[0][idx_tempo] * x_vals)))

plt.plot(x_vals, prob)
plt.title("Curva Logística - Probabilidade de Compra vs Tempo no Site")
plt.xlabel("Tempo no Site (min)")
plt.ylabel("Probabilidade de Compra")
plt.show()

# -----------------------------
# 7. Teste com um novo cliente
# -----------------------------
novo_cliente = pd.DataFrame({
    "Idade": [30],
    "Tempo_no_site": [12.5],
    "Numero_de_visitas": [10],
    "Valor_medio_carrinho": [500],
    "Tempo_de_cadastro_meses": [24],
    "Categoria_preferida_Beleza": [0],
    "Categoria_preferida_Casa": [0],
    "Categoria_preferida_Eletrônicos": [1],
    "Categoria_preferida_Esportes": [0],
    "Categoria_preferida_Roupas": [0]
})

# Garantir que todas as colunas do modelo estão presentes
for col in X.columns:
    if col not in novo_cliente.columns:
        novo_cliente[col] = 0

# Reordenar colunas para coincidir com X
novo_cliente = novo_cliente[X.columns]


probabilidade = modelo.predict_proba(novo_cliente)[0][1]
print(f"\nProbabilidade de compra para o novo cliente: {probabilidade:.2%}")