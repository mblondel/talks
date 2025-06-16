class: middle, center, title-slide

$$
\gdef\muv{\bm{\mu}}
\gdef\thetav{\bm{\theta}}
\gdef\x{\bm{x}}
\gdef\y{\bm{y}}
\gdef\w{\bm{w}}
\gdef\cC{\mathcal{C}}
\gdef\cL{\mathcal{L}}
\gdef\cW{\mathcal{W}}
\gdef\cX{\mathcal{X}}
\gdef\cY{\mathcal{Y}}
\gdef\RR{\mathbb{R}}
\gdef\EE{\mathbb{E}}
\gdef\LSE{{\mathrm{LSE}}}
$$

# Joint Learning of Energy-based Models and their Partition Function

<br><br>
Michaël Sander, Vincent Roulet, Tianlin Liu, Mathieu Blondel

---

# Outline

* What are energy-based models? (EBMs)
* Inference with EBMs
* Existing approaches for learning EBMs
* Proposed learning approach
* Experiments

---

{{OUTLINE}}

---

## Problem setup: probabilistic structured prediction

.center[**Goal:** learn a conditional probability distribution $p(\y|\x)$]

$\cX$: input space <br>
$\cY$: combinatorially-large discrete output space

<br>

**Examples**

* Multi-label classification: $\cY$ is the power set of $[k]$, with $|\cY| = 2^k$
* Label ranking: $\cY$ is the set of permutations, with $|\cY| = k!$


---

## Probabilistic energy-based models (EBMs)

Turn a function into a probability distribution without factorization assumptions.
$$
p\_g(\y|\x) \coloneqq \frac{q(\y|\x)\exp(g(\x, \y))}{\sum\_{\y' \in \cY}q(\y'|\x)\exp(g(\x, \y'))}
$$

<br>

$g(\x, \y)$: scalar-valued function, $q(\y|\x)$ prior distribution

<br><br>

EBMs ar also used in the unsupervised setting to learn a distribution $p\_g(\x)$.

---

## Model decomposition

$$
g(\x, \y) \coloneqq \Phi(h(\x), \y)
$$

$\Phi(\thetav, \y)$ coupling function <br>
$\thetav \coloneqq h(\x)$ is a model function (e.g., neural net) producing the logits $\thetav$.

<br>

**Bilinear coupling**
$$
\Phi(\thetav, \y) \coloneqq \langle \thetav, \y \rangle
$$

<br>

**Gibbs distribution**

EBMs coincide with the Gibbs distribution (exponential family) with
**natural parameters** $\thetav$ and **base measure** $q$.
$$
p\_g(\y|\x) \coloneqq \frac{q(\y|\x)\exp(\langle \thetav, \y \rangle)}{\sum\_{\y' \in \cY}q(\y'|\x)\exp(\langle \thetav, \y' \rangle)}
$$

---

{{OUTLINE}}

---

## Three inference problems

* Computing the mode
$$
\y^\star\_g(\x) 
\coloneqq \argmax\_{\y \in \cY} p\_g(\y|\x) 
$$

<br>

* Sampling
$$
\y \sim p\_g(\cdot|\x)
$$

<br>

* Computing the mean
$$
\muv\_g(\x) 
\coloneqq \EE\_{\y \sim p\_g(\cdot|\x)}[\y]
\in \mathrm{conv}(\cY)
$$

<br>

.center[Each problem requires a specific oracle for a given output set $\cY$.]

---

## Computing the mode: linear coupling case


If $g(\x, \y) = \langle h(\x), \y \rangle$ 
and $q$ is uniform, then

<br>

$$
\y^\star\_g(\x) 
= \argmax\_{\y \in \cY} \langle h(\x), \y \rangle
\subseteq \argmax\_{\y \in \mathrm{conv}(\cY)} \langle h(\x), \y \rangle
$$

<br>

This is a **linear program**. 
Dedicated oracles exist for specific $\cY$.

---

## Computing the mode: nonlinear coupling case

More generally we can solve the **relaxed problem**
$$
\argmax\_{\muv \in \cC} g(\x, \muv) \approx \y^\star\_g(\x)
$$

$\cC$ is a convex superset of $\cY$, for example, $\cC = \mathrm{conv}(\cY)$.

$g(\x, \muv)$ is well-defined on $\cX \times \cC$ instead of
$\cX \times \cY$.

Typically, rounding the solution from $\cC$ to $\cY$ is necessary.

**Example**

$\argmax\_{\y \in \\{0,1\\}^k} g(\x, \y)$
can be relaxed into
$\argmax\_{\muv \in [0,1]^k} g(\x, \muv)$

---

## Sampling

How to sample
$$\y \sim p\_g(\cdot|\x)$$
where
$$p(\y|\x) \propto q(\y|\x) \exp(g(\x, \y))$$

<br>

Designing a sampler is typically case-by-case.

* Continuous $\cY$: Langevin
* Binary $\cY$: Gibbs sampling

---

{{OUTLINE}}

---

## The challenge of MLE

$$
\begin{aligned}
\cL\_{\mathrm{MLE}}(g) 
&\coloneqq \EE\_{(\x,\y)} \left[-\log p\_g(\y|\x)\right]\\\\
&= \EE\_{\x} \left[\LSE\_g(\x)\right] - \EE\_{(\x,\y)} \left[g(\x, \y) -\log(q(\y|\x))\right]
\end{aligned}
$$

**Problem**

The log-partition function is intractable in general.

$$
\LSE\_g(\x) \coloneqq \log \sum_{\y' \in \cY} q(\y'|\x) \exp(g(\x,\y'))
$$

---

{{OUTLINE}}

---

{{OUTLINE}}
