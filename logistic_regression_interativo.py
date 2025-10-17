import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Carregar a base de dados
# -----------------------------
df = pd.read_csv("clientes_realista.csv")
print("Base carregada com sucesso!")

# -----------------------------
# 2. Pré-processamento
# -----------------------------
X = pd.get_dummies(df.drop("Comprou", axis=1), drop_first=True)
y = df["Comprou"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 3. Treinamento do modelo
# -----------------------------
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)
print("Modelo treinado com sucesso!")

# -----------------------------
# 4. Avaliação rápida
# -----------------------------
y_pred = modelo.predict(X_test)
print("\nAcurácia:", accuracy_score(y_test, y_pred))

# -----------------------------
# 5. Interação com o usuário
# -----------------------------
print("\n--- Teste interativo de novo cliente ---")
idade = float(input("Idade: "))
tempo_no_site = float(input("Tempo no site (min): "))
numero_visitas = int(input("Número de visitas: "))
valor_medio_carrinho = float(input("Valor médio do carrinho: "))
tempo_cadastro = int(input("Tempo de cadastro (meses): "))
categoria = input("Categoria preferida (Eletrônicos, Roupas, Beleza, Casa, Esportes): ")

# Criar dataframe do cliente
novo_cliente = pd.DataFrame({
    "Idade": [idade],
    "Tempo_no_site": [tempo_no_site],
    "Numero_de_visitas": [numero_visitas],
    "Valor_medio_carrinho": [valor_medio_carrinho],
    "Tempo_de_cadastro_meses": [tempo_cadastro],
    "Categoria_preferida_Beleza": [1 if categoria=="Beleza" else 0],
    "Categoria_preferida_Casa": [1 if categoria=="Casa" else 0],
    "Categoria_preferida_Eletrônicos": [1 if categoria=="Eletrônicos" else 0],
    "Categoria_preferida_Esportes": [1 if categoria=="Esportes" else 0],
    "Categoria_preferida_Roupas": [1 if categoria=="Roupas" else 0]
})

# Garantir que todas as colunas do modelo estão presentes
for col in X.columns:
    if col not in novo_cliente.columns:
        novo_cliente[col] = 0

# Reordenar colunas para coincidir com X
novo_cliente = novo_cliente[X.columns]

# Prever probabilidade
probabilidade = modelo.predict_proba(novo_cliente)[0][1]
print(f"\nProbabilidade de compra para este cliente: {probabilidade:.2%}")
