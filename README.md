# SPCI — Sequential Predictive Conformal Inference

Este repositório implementa o **SPCI** conforme o artigo dos autores (SPCI: *Sequential Predictive Conformal Inference for Time Series*):
- **Resíduos LOO (leave-one-out) via bagging**: treinamos `B` modelos base em *bootstrap* e, para cada ponto de treino, agregamos **apenas** os modelos que **não** viram o ponto (OOB) para obter os resíduos de calibração.
- **Regressão de quantis condicional nos resíduos passados** usando **Quantile Random Forest (QRF)**. O modelo aprende `Q̂_t(p | ε̂_{t-1},…,ε̂_{t-w})` com uma janela AR de resíduos de tamanho `w`.
- **Otimização de β ∈ [0, α]**: a cada passo, escolhemos `β̂` que **minimiza a largura** do intervalo `[Q̂_t(β), Q̂_t(1−α+β)]`, mantendo cobertura alvo `1−α` (equações (10)–(11) do paper).
- **Atualização online**: quando `y_t` é observado, atualizamos a lista de resíduos e **recalibramos** os intervalos dinamicamente.
- **Multi-step**: sem `y_true`, para um bloco futuro de tamanho `H` calculamos intervalos para horizontes `h=1..H` usando **modelos quantílicos horizonte-específicos** (estratégia “divide-and-conquer” do paper).

> **Backends de QRF suportados:** `sklearn-quantile` (preferido) ou `skranger`. Se nenhum estiver disponível, há um **fallback KNN** (apenas para smoke test). Para resultados fiéis ao artigo, instale um backend QRF.

---

## Sumário
- [Instalação](#instalação)
- [Ideia do método (resumo do artigo)](#ideia-do-método-resumo-do-artigo)
- [API](#api)
- [Exemplos de uso](#exemplos-de-uso)
  - [Online (1 passo à frente, com atualização)](#online-1-passo-à-frente-com-atualização)
  - [Multi-step (H passos à frente)](#multi-step-h-passos-à-frente)
  - [Usando seu modelo base treinado](#usando-seu-modelo-base-treinado)
- [Smoke tests](#smoke-tests)
- [Práticas recomendadas](#práticas-recomendadas)
- [Limitações e notas](#limitações-e-notas)
- [Citação](#citação)

---

## Instalação

Crie um ambiente e instale as dependências. **Escolha um backend QRF**:

```bash
python -m venv .venv && source .venv/bin/activate     # no Windows: .venv\Scripts\activate
pip install -U pip wheel setuptools numpy scipy scikit-learn

# Backend QRF (escolha UM; recomendado)
pip install sklearn-quantile            # preferido
# ou
pip install skranger                    # alternativa

# Instale o pacote (modo editável)
pip install -e .
```

> Se nenhum backend QRF estiver instalado, o pacote usará **KNN quantile** como fallback — útil para teste rápido, mas **não é** a configuração fiel ao artigo.

---

## Ideia do método (resumo do artigo)

1) **Preditor pontual f̂(X)** via **ensemble bootstrap** (`B` modelos). No treino, para cada observação `i`, agregamos as previsões **apenas** dos modelos que **não** usaram `i` (OOB) ⇒ obtemos o **resíduo LOO** `ε̂_i = y_i − f̂_{−i}(X_i)`.

2) **Quantis condicionais do resíduo futuro**: montamos uma **janela de resíduos** tamanho `w` e treinamos uma **QRF** para prever `ε̂_{t+1}` a partir de `[ε̂_{t}, ε̂_{t-1}, …, ε̂_{t-w+1}]`. Isso captura dependência temporal e heterocedasticidade.

3) **Intervalo com β ótimo**: em vez de usar um intervalo central simétrico, escolhemos `β̂ ∈ [0, α]` que **minimiza a largura** `Q̂_t(1−α+β) − Q̂_t(β)`, garantindo cobertura `1−α` e **estreitando** o intervalo quando há assimetria nos erros.

4) **Atualização online**: após prever `Y_t`, quando `y_t` é observado, **atualizamos** a lista de resíduos com `ε̂_t` (e opcionalmente usamos uma janela deslizante), **reajustando** a QRF a cada passo.

5) **Multi-step** (`H>1`): treinamos **modelos quantílicos por horizonte** (para `h=1..H`) sobre pares defasados de resíduos e produzimos `H` intervalos simultaneamente.

---

## API

```python
from spci import SPCI

SPCI(
    base_model="rf",     # (obj sklearn-like ou "rf") preditor pontual f̂; usado no ensemble bootstrap
    B=30,                # nº de modelos bootstrap para f̂
    alpha=0.1,           # 1 - cobertura alvo
    w=20,                # janela (lags) de resíduos para a QRF
    bins=5,              # nº de candidatos de β em [0, α] (malha para min. de largura)
    qrf_backend="auto",  # "auto" usa QRF (sklearn-quantile/skranger); "knn" força fallback KNN
    random_state=0
)
```

- **`.fit(X, y)`**: treina o ensemble bootstrap e calcula resíduos LOO de calibração.
- **`.predict_interval(X_new, y_true=None)`**:
  - Se `y_true` for `None`: modo **multi-step** — para `H=len(X_new)`, retorna `H` intervalos (horizonte-específicos).
  - Se `y_true` for fornecido: modo **online (1 passo)** — atualiza os resíduos a cada passo.

**Retorno:** `dict` com `{"lower": np.ndarray, "upper": np.ndarray, "center": np.ndarray}` (tamanhos `(H,)`).

---

## Exemplos de uso

### Online (1 passo à frente, com atualização)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from spci import SPCI

rng = np.random.default_rng(0)
n = 400
X = rng.normal(size=(n, 5))
beta = rng.normal(size=(5,))
y = X @ beta + 0.4*rng.standard_t(df=5, size=n)

spci = SPCI(base_model=LinearRegression(),
            B=30, alpha=0.1, w=20, bins=7, qrf_backend="auto", random_state=0)

spci.fit(X[:320], y[:320])
res = spci.predict_interval(X[320:], y_true=y[320:])   # online (atualiza resíduos)

lower, upper, center = res["lower"], res["upper"], res["center"]
print("Cobertura empírica:", np.mean((y[320:] >= lower) & (y[320:] <= upper)))
```

### Multi-step (H passos à frente)

Quando você tem um bloco futuro de tamanho `H` e **não** observa `y_true` durante a previsão:

```python
H = 12
X_future = X[-H:]
res = spci.predict_interval(X_future)   # multi-step (h=1..H)
# res["lower"][h-1], res["center"][h-1], res["upper"][h-1] para cada horizonte
```

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

## Smoke tests

- **QRF real (recomendado):** garante que você está de fato usando `sklearn-quantile`/`skranger`.
  ```bash
  python smoke_test_qrf.py
  ```

- **Fallback KNN (apenas para ambientes sem QRF):**
  ```bash
  python smoke_test_knn.py
  ```

Cada script imprime cobertura empírica e largura média do intervalo.

---

## Práticas recomendadas

- **Escolha de `w` (tamanho da janela):** use valores que capturem a memória relevante dos resíduos (e.g., 20–100). Valide por *backtesting*.
- **`B` (ensemble bootstrap):** 20–50 costuma estabilizar o centro f̂ e os resíduos LOO.
- **Grade de `β` (`bins`):** 5–11 pontos entre `[0, α]` é um bom começo; aumente se notar assimetria forte.
- **QRF backend:** para fidelidade ao paper, **instale `sklearn-quantile`** (ou `skranger`). O fallback KNN é para teste apenas.
- **Reprodutibilidade:** fixe `random_state` e, se possível, fixe seeds globais das libs usadas.
- **Validação:** monitore cobertura realizada vs. alvo `1−α`; ajuste `w`, `bins` e hiperparâmetros do preditor pontual.

---

## Limitações e notas

- **Sigma(X) heterocedástico (MLP):** esta *build* QRF-first **não** inclui a estimação explícita de `σ(X)` por MLP (variante presente em outra build fiel com PyTorch). Posso integrar essa trilha aqui se desejar (mantendo fidelidade).
- **Backends QRF de terceiros:** implementações podem diferir sutilmente no cálculo de quantis; versões distintas podem levar a pequenas variações nos números.
- **Inputs:** a API espera `X` como `array (n, d)` e `y` como `(n,)` (ou `(n,1)`).

---

## Referências

Xu, Chen, and Yao Xie. "Sequential predictive conformal inference for time series." International Conference on Machine Learning. PMLR, 2023.
