class: middle, center, title-slide

$$
\gdef\s{{\bm{s}}}
\gdef\x{{\bm{x}}}
\gdef\y{\bm{y}}
\gdef\cA{\mathcal{A}}
\gdef\cP{\mathcal{P}}
\gdef\cX{\mathcal{X}}
\gdef\cY{\mathcal{Y}}
\gdef\RR{\mathbb{R}}
\gdef\EE{\mathbb{E}}
\gdef\ebm{{\mathrm{EBM}}}
\gdef\arm{{\mathrm{ARM}}}
$$

# Autoregressive Language Models are Secretly Energy-Based Models

## Insights into the Lookahead Capabilities of Next-Token Prediction

<br><br>
<small>Mathieu Blondel, Michaël Sander, Germain Vivier-Ardisson, Tianlin Liu, Vincent Roulet</small>

---

# Outline

* Energy-based models (EBMs)
* Autoregressive models (ARMs)
* Bijection between ARM logits and EBM logits
* Optimality of teacher forcing

---

{{OUTLINE}}

---

## Energy-based models (EBMs)

EBMs are sequence-level Gibbs / Boltzmann distributions
$$
p\_R^\ebm(\y|\x) \coloneqq \frac{\exp(R(\x, \y))}{\sum_{\y' \in \cY} \exp(R(\x, \y'))}
$$
$R$ scores the affinity between the prompt $\x$ and the **entire** response $\y$.

---

## Pros and cons of EBMs 

* **Pros**

  * Can score the entire response given the prompt.

  * Can use bidirectional (non-causal) Transformer.

  * Have the ability to plan ahead the response.

* **Cons**

  * Intractable normalization constant.

  * Difficult to sample from (needs to resort to MCMC).

  * Difficult to train from prompt-response pairs (log-probabilities are intractable).

---

## Sequence-level log-partition

EBMs can be rewritten as
$$
p\_R^\ebm(\y|\x)  = \exp(R(\x, \y) - A\_R^\ebm(\x))
$$
where we used the sequence-level log-partition
$$
A\_R^\ebm(\x)) \coloneqq \log \sum_{\y \in \cY} \exp(R(\x, \y))
$$

---

## Why studying the equivalence between EBMs and ARMs?

<br/>

**Any strictly positive distribution can be written as an EBM**

Indeed, for all $p(\y|\x) > 0$,
$$
R(\x, \y) \coloneqq \log p(\y|\x)
\implies
p(\y|\x) = p^\ebm\_R(\y|\x)
$$

<br/>

**The optimal solution of maxent RL is an EBM**

$$
p^\star \coloneqq \argmax\_{p \in \cP(\cY|\cX)} 
\EE\_X \EE\_{Y \sim p} \left[R(X, Y) - \mathrm{KL}(p(\cdot|X), p\_{\mathrm{ref}}(\cdot|X))\right]
$$

$$
R\_{\mathrm{ref}}(\x, \y) \coloneqq \log p\_{\mathrm{ref}}(\y|\x)
$$

<br/>

$$
\implies
p^\star = p^\ebm\_{R + R\_{\mathrm{ref}}}
$$

---

{{OUTLINE}}

---

## Autoregressive models (ARMs)

Autoregressive models are factorized as
$$
p^\arm\_q(\y|\x) \coloneqq \prod\_{t=1}^T \pi\_q(y\_t | \x \oplus \y\_{< t})
$$

where

$$
\pi\_q(\y\_t | \s\_t) \coloneqq \frac{\exp(q(\s\_t, y\_t))}{\sum\_{j \in \cA} \exp(q(\s\_t, j)}
$$

$q$ scores the next token $y\_t$ given the context (prefix) $\s\_t$.

---

## Pros and cons of ARMs 

* **Pros**

  * Easy to sample from (by ancestral sampling).

  * Easy to train from input-output pairs (log-probabilities are intractable).

* **Cons**

  * On first sight, they look myopic (next-token prediction).

  * Must causal Transformers.


---

{{OUTLINE}}

---

## Future work

* How much do chain-of-thoughts help with estimating $V\_q$?
