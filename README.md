# SPCI — Sequential Predictive Conformal Inference (Time Series)

> Implementação em Python com API simples (estilo *statsmodels*) do método **SPCI** — *Sequential Predictive Conformal Inference for Time Series* (Xu & Xie).  
> Constrói **intervalos de predição conformais** sequenciais, **adaptados no tempo**, a partir de:
> 1) um **preditor pontual** obtido por *bagging* (resíduos **LOO/OOB** no treino), e  
> 2) uma **regressão de quantis condicional** sobre janelas de **resíduos passados**, recalibrada a cada passo.

---

## Ideia do método (conforme o artigo)

Para prever \(Y_t\) dado \(X_t\):

1. **Preditor pontual \(\hat f\)** (bagging): treine \(B\) modelos-base via **bootstrap** e, para cada ponto do treino, compute **previsões *leave-one-out*** (agregando apenas os modelos que **não** viram aquele ponto). Os **resíduos LOO** são \(\hat\varepsilon_i = y_i - \hat f_{-i}(x_i)\).  
2. **Janela de resíduos**: mantenha um histórico \(\ldots,\hat\varepsilon_{t-w},\dots,\hat\varepsilon_{t-1}\) de tamanho \(w\).
3. **Quantis condicionais**: ajuste uma **regressão de quantis** (ex.: **QRF – Random Forest Quantílico**) para aprender o mapeamento
   \[ [\hat\varepsilon_{u+w-1},\dots,\hat\varepsilon_u] \longmapsto \hat\varepsilon_{u+w} \]
   e **estime** \(Q_t(p)\), o \(p\)-quantil **condicional** do próximo resíduo, **dado a janela atual**.
4. **Escolha adaptativa de \(\beta\)**: construa o intervalo
   \[ C_{t-1}(X_t) = \big[\ \hat f(X_t) + Q_t(\beta)\ ,\ \hat f(X_t) + Q_t(1-\alpha+\beta)\ \big], \]
   escolhendo \(\beta \in [0,\alpha]\) que **minimiza a largura** \(\,Q_t(1-\alpha+\beta)-Q_t(\beta)\,\).  
   (Quando a distribuição do erro é assimétrica, deslocar \(\beta\) reduz a largura mantendo cobertura \(1-\alpha\).)
5. **Atualização online**: após observar \(y_t\), **atualize** o histórico de resíduos com \( \hat\varepsilon_t = y_t - \hat f(X_t) \) (ou padronizado por \(\hat\sigma(X_t)\) quando modelado), e repita.

> No paper, este fluxo aparece como **Algoritmo 1** (predição 1-passo), com fórmulas (10)–(11) para a escolha de \(\beta\) e (13) para a construção das **features** de regressão quantílica (janelas dos resíduos).

---

## Instalação

O pacote é puro Python; para usar **QRF** (Random Forest Quantílico) — recomendado — instale uma das opções:

```bash
pip install sklearn-quantile    # (recomendado)
# ou
pip install skranger            # alternativa com quantis
```

Se não houver backend de QRF disponível, a biblioteca usa um **fallback KNN** simples para quantis.

---

## Exemplo rápido (RF base + QRF condicional)

```python
import numpy as np
from spci import SPCI

# dados sintéticos (tabulares para ilustração)
rng = np.random.default_rng(0)
n = 300
X = rng.normal(size=(n, 5))
beta = np.array([1.2, -0.8, 0.0, 0.5, 0.0])
y = X @ beta + rng.standard_t(df=4, size=n) * 0.7  # erros assimétricos/heavy-tail

# split
X_tr, y_tr = X[:220], y[:220]
X_te, y_te = X[220:], y[220:]

# SPCI (preditor pontual RF com bagging interno; QRF para quantis condicionais)
m = SPCI(base_model="rf", B=30, alpha=0.1, w=30, bins=7, random_state=0)
m.fit(X_tr, y_tr)

# Predição sequencial com feedback (atualiza resíduos on-line)
res = m.predict_interval(X_te, y_true=y_te)
lb, ub, center = res["lower"], res["upper"], res["center"]

print("Cobertura empírica:", np.mean((y_te >= lb) & (y_te <= ub)))
print("Largura média:", np.mean(ub - lb))
```

---

## API essencial

```python
from spci import SPCI

SPCI(
  base_model="rf",   # "rf" (RF interno), "mlp" (usa PyTorch) ou um estimador sklearn-like
  B=30,              # nº de modelos no ensemble (bagging) para resíduos LOO
  alpha=0.1,         # 1 - cobertura alvo
  w=20,              # tamanho da janela de resíduos para a regressão de quantis
  bins=5,            # nº de candidatos β em [0, α] para minimizar a largura
  fit_sigmaX=False,  # se base_model="mlp": ativa MLP separada para σ(X) (heterocedasticidade)
  random_state=0
)
.fit(X_train, y_train)                 # treina ensemble e calcula resíduos LOO
.predict_interval(X_future, y_true)    # constrói intervalos; se y_true é passado, atualiza resíduos online
```

### Parâmetros principais
- **`base_model`**:  
  - `"rf"`: usa RandomForestRegressor como preditor pontual;  
  - `"mlp"`: usa MLP (PyTorch) e, com `fit_sigmaX=True`, ajusta **σ(X)** (rede MLP com saída positiva) para **padronizar resíduos**;  
  - estimador sklearn-like (p.ex. `LinearRegression()`), desde que tenha `.fit/.predict`.
- **`w`** (*window*): tamanho da janela de resíduos para as **features** da QRF (Eq. 13).  
- **`bins`**: nº de pontos na malha \([0,\alpha]\) para buscar \(\beta\) que minimiza a largura (Eqs. 10–11).  
- **`B`**: nº de modelos no *bagging* para gerar resíduos **LOO** (via amostras bootstrap).

---

## Exemplo: uso com estimador do scikit-learn (LinearRegression)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from spci import SPCI

rng = np.random.default_rng(1)
X = rng.normal(size=(240, 4))
y = X @ np.array([0.9, -0.4, 0.0, 0.2]) + rng.normal(scale=0.6, size=240)

X_tr, y_tr = X[:180], y[:180]
X_te, y_te = X[180:], y[180:]

# passa o estimador sklearn como base_model
m = SPCI(base_model=LinearRegression(), B=40, alpha=0.1, w=25, bins=5, random_state=1)
m.fit(X_tr, y_tr)

res = m.predict_interval(X_te, y_true=y_te)

print("Cobertura:", ((y_te >= res["lower"]) & (y_te <= res["upper"])).mean())
```

---

### Multi-step (H passos à frente)

Quando você tem um bloco futuro de tamanho `H` e **não** observa `y_true` durante a previsão:

```python
H = 12
X_future = X[-H:]
res = spci.predict_interval(X_future)   # multi-step (h=1..H)
# res["lower"][h-1], res["center"][h-1], res["upper"][h-1] para cada horizonte
```

---

### Usando seu modelo base treinado

**Recomendado (fiel ao método):** reusar apenas os **hiperparâmetros**, deixando o SPCI treinar o ensemble bootstrap:

```python
from sklearn.base import clone

trained = ...                     # seu estimador já treinado
base_unfitted = clone(trained)    # clona hiperparâmetros, zera os pesos

spci = SPCI(base_model=base_unfitted, B=30, alpha=0.1, w=20, qrf_backend="auto", random_state=0)
spci.fit(X_train, y_train)
res = spci.predict_interval(X_test, y_true=y_test)
```

> **Observação:** se você **precisar** usar exatamente o modelo já treinado sem re-treinar, dá para montar um *wrapper* “congelado” e usar `B=1`. Funciona, mas **não** produz resíduos LOO e **enfraquece as garantias**. Prefira o procedimento acima.
---


## Exemplo (opcional): heterocedasticidade com MLP e $$\sigma(X)$$

> Requer **PyTorch**: `pip install torch`

```python
import numpy as np
from spci import SPCI

rng = np.random.default_rng(2)
n = 500
X = rng.normal(size=(n, 6))
sigma = 0.3 + 0.8 * (X[:, 0] > 0)       # variância condicional dependente de X
y = (X @ np.array([0.7,0,0.3,0,0,0])) + rng.normal(scale=sigma)

X_tr, y_tr = X[:400], y[:400]
X_te, y_te = X[400:], y[400:]

m = SPCI(base_model="mlp", fit_sigmaX=True, B=20, alpha=0.1, w=30, bins=7, random_state=0)
m.fit(X_tr, y_tr)

res = m.predict_interval(X_te, y_true=y_te)

print("Cobertura:", ((y_te >= res['lower']) & (y_te <= res['upper'])).mean())
```

---

## Notas práticas

- **QRF vs. KNN**: para resultados como no artigo, use **QRF** (`sklearn-quantile` ou `skranger`). O **KNN** é apenas um *fallback* leve.  
- **Escolha de `w`**: 20–100 costuma funcionar; ajuste pela dependência temporal dos resíduos.  
- **`bins`** (β): mais pontos \(\Rightarrow\) busca mais fina, porém mais custo; 5–11 é um bom ponto de partida.  
- **Cobertura simultânea (multi-passos)**: para vários passos à frente, a formulação do paper sugere modelos por horizonte (Apênd. B.3). Esta implementação prioriza o **caso 1-passo** (Alg. 1).  
- **Desbalanceamento/Drift**: use janelas menores (`w`) para reagir mais rápido a mudanças; considere ponderação temporal no QRF se necessário (custom).

---

## Referências

Se este pacote for útil na sua pesquisa, **por favor, cite**:

### Este pacote
**Texto (ABNT/APA):**  
> *SPCI (Python): Sequential Predictive Conformal Inference.* Repositório GitHub, 2025. Disponível em: **https://github.com/Jonas-Santos-Siqueira/SPCI**.

**BibTeX (substitua os campos em maiúsculas):**
```bibtex
@software{SPCI_Python,
  title   = {SPCI (Python): Sequential Predictive Conformal Inference},
  author  = {Autor do repositório},
  year    = {2025},
  version = {X.Y.Z},
  url     = {URL_DO_REPO},
  note    = {Python package}
}
```

### Artigo oficial (SPCI)
**Texto (APA):**  
> Xu, C., & Xie, Y. (2023). *Sequential Predictive Conformal Inference for Time Series*.

**BibTeX (mínimo):**
```bibtex
@article{XuXieSPCI2023,
  title   = {Sequential Predictive Conformal Inference for Time Series},
  author  = {Xu, Chen and Xie, Yao},
  year    = {2023}
}
```

> **Opcional:** como SPCI se apoia no EnbPI, considere citar também:  
> Xu, C., & Xie, Y. (2021). *Conformal Prediction Interval for Dynamic Time-Series* (EnbPI).
```bibtex
@inproceedings{XuXieEnbPI2021,
  title     = {Conformal Prediction Interval for Dynamic Time-Series},
  author    = {Xu, Chen and Xie, Yao},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  year      = {2021}
}
```

---
