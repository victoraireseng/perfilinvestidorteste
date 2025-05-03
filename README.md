# Importações necessárias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Geração de Dados Simulados (1000 amostras)
np.random.seed(42)
n_samples = 1000

# Variáveis macroeconômicas (features)
inflacao = np.random.normal(5, 2, n_samples)  # Inflação média de 5% com desvio 2%
juros = np.random.normal(7, 1.5, n_samples)   # Taxa de juros média de 7%
pib = np.random.normal(2, 1, n_samples)       # Crescimento do PIB em %
volatilidade = np.random.uniform(10, 30, n_samples)  # Índice de volatilidade (VIX)

# Perfil do investidor (target: 0=Conservador, 1=Moderado, 2=Agressivo)
# Probabilidades baseadas em cenários econômicos
perfil = []
for i in range(n_samples):
    if inflacao[i] > 6 and juros[i] > 8:
        perfil.append(0)  # Conservador em cenário de alta inflação/juros
    elif pib[i] > 3 and volatilidade[i] < 15:
        perfil.append(2)  # Agressivo em cenário de alto PIB e baixa volatilidade
    else:
        perfil.append(1)  # Moderado nos demais casos

# Criar DataFrame
data = pd.DataFrame({
    'Inflacao': inflacao,
    'Juros': juros,
    'PIB': pib,
    'Volatilidade': volatilidade,
    'Perfil': perfil
})

# 2. Pré-processamento
X = data.drop('Perfil', axis=1)
y = data['Perfil']

# Dividir em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Treinamento do Modelo (Decision Tree)
clf = DecisionTreeClassifier(
    criterion='gini',       # Critério de divisão (gini ou entropy)
    max_depth=4,            # Profundidade máxima para evitar overfitting
    min_samples_split=20,   # Mínimo de amostras para dividir um nó
    class_weight='balanced' # Ponderação para classes desbalanceadas
)
clf.fit(X_train, y_train)

# 4. Avaliação do Modelo
y_pred = clf.predict(X_test)

print("\n--- Métricas de Classificação ---")
print(classification_report(y_test, y_pred, target_names=['Conservador', 'Moderado', 'Agressivo']))

print("\n--- Matriz de Confusão ---")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    columns=['Pred Conservador', 'Pred Moderado', 'Pred Agressivo'],
    index=['Real Conservador', 'Real Moderado', 'Real Agressivo']
))

# 5. Análise das Regras de Decisão
print("\n--- Regras da Árvore (Exemplo) ---")
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)

# 6. Importância das Variáveis
print("\n--- Importância das Variáveis ---")
importance = pd.DataFrame({
    'Variável': X.columns,
    'Importância': clf.feature_importances_
}).sort_values('Importância', ascending=False)
print(importance)

# 7. Visualização da Árvore
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Conservador', 'Moderado', 'Agressivo'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árvore de Decisão - Perfil de Investidor", fontsize=14)
plt.show()
