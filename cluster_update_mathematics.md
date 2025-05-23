# Mathematical Details of Cluster Updates in Hierarchical Partially-Ordered MCMC


## 1. Chinese Restaurant Process (CRP) Prior

For the clustering component, we use a Chinese Restaurant Process (CRP) prior, which is a Dirichlet Process with concentration parameter $\alpha$. It provides a prior distribution over partitions of elements.

### Mathematical Formulation

The probability of a new element being assigned to an existing cluster $k$ is proportional to:

$$
P(\text{assign to cluster } k) \propto (n_k - \alpha)
$$

And the probability of creating a new cluster is proportional to:

$$
P(\text{new cluster}) \propto (\theta + \alpha \cdot |\text{clusters}|)
$$

Where:

- $n_k$ is the number of elements already in cluster $k$
- $\alpha$ is the discount parameter (typically between 0 and 1)
- $\theta$ is the base concentration parameter
- $|\text{clusters}|$ is the current number of clusters

### The CRP:

The Chinese Restaurant Process can be derived from the Pitman-Yor process, which is a generalization of the Dirichlet process with parameters $0 \leq \alpha < 1$ (discount parameter) and $\theta > -\alpha$ (concentration parameter).

Starting with a sequential construction where customers (data points) arrive one at a time:

1. Customer 1 always sits at the first table.
2. When the $(n+1)$-th customer arrives, they choose:
   - An existing table $k$ with probability proportional to $(n_k - \alpha)$
   - A new table with probability proportional to $(\theta + \alpha \cdot K)$

Let $X_i$ be the table assignment for customer $i$, and $K_n$ be the number of distinct tables after $n$ customers. The conditional probabilities are:

$$
P(X_{n+1} = k | X_1,...,X_n) = \frac{n_k - \alpha}{n + \theta}, \quad \text{for existing table } k
$$

$$
P(X_{n+1} \neq X_i \text{ for all } i \leq n | X_1,...,X_n) = \frac{\theta + \alpha K_n}{n + \theta}
$$

### 2.2 Clustered Labeled Model

The data came in groups $Y_a$ with assessor labels $a \in A$. If we simply have a collection of $N$ lists $Y = (Y_1, Y_2, \ldots, Y_N)$ with $Y_i \in C_{S_i}$ and we think there may be some latent group structure, we add $N$ unknown label parameters $c = (c_1, \ldots, c_N)$ with $c_i \in A$ for each $i \in [N]$.

Take a two dimensional Dirichlet distribution for the grouping, assuming the number of groups $A$ is fixed and the probability $i$ is assigned to group $a \in A$ is $w_a$ with $w = (w_1, \ldots, w_A)$ and $w \sim \text{Dirichlet} (\frac{\gamma}{A}, \ldots, \frac{\gamma}{A})$ (this parameterization recovers the Chinese Restaurant Process (CRP) with parameter $\gamma > 0$ for non-empty groups as $A \to \infty$).

Let

$$
N_a(c) = \sum_{i=1}^{N} I_{c_i=a}
$$

Upon integrating $w$, the prior for $c$ is

$$
\pi_C (c|\gamma) = \frac{\Gamma(\gamma)}{\Gamma (\frac{\gamma}{A})^A} \frac{\prod_{a=1}^{A} \Gamma (N_a(c) + \frac{\gamma}{A})}{\Gamma(N + \gamma)}
$$

The likelihood now depends on the list-labels $c$, that is

$$
\prod_{i=1}^{N} p_{S_i} (Y_i|h(c_i)[S_i])
$$

where $h(c_i) = h(\eta(c_i)(U^{(c_i)}, \beta))$. The posterior becomes

$$
\pi_S (\rho, \beta, \tau, U, c|Y) \propto \pi_R(\rho)\pi_B (\beta)\pi_T (\tau)\pi(U|\rho, \tau)\pi_C (c|\gamma)p_S (Y|c, h(U, \beta))
$$

with $\pi_C$ given above and $p_S (Y|c, h(U, \beta))$ as defined earlier.

#### The CRP Example:

**Hyper-parameters:**


| Quantity                   | Value        |
| -------------------------- | ------------ |
| Number of items            | $N = 5$      |
| Number of (assumed) groups | $A = 3$      |
| Dirichlet concentration    | $\gamma = 1$ |

**Current seating configuration:**

$c = (1, 1, 2, 3, 2) \Rightarrow N_1 = 2, N_2 = 2, N_3 = 1$.

**Dirichlet–multinomial prior weight:**

With $\gamma = 1$ the prior weight of this configuration, derived from the equation above, is

$$
\pi_C(c | \gamma = 1) = \frac{\Gamma(\frac{\gamma}{A})^A}{\Gamma(\gamma)} \frac{\Gamma(N_1 + \frac{\gamma}{A}) \Gamma(N_2 + \frac{\gamma}{A}) \Gamma(N_3 + \frac{\gamma}{A})}{\Gamma(N + \gamma)} \approx 5.49 \times 10^{-4}
$$

(Here $N = \sum_{a=1}^{A} N_a = 5$.)

#### Algorithm Steps:

1. Iterate over $i=1...N$, removing each item from its current cluster
2. Calculate the probability weights for assigning the item to each existing cluster or a new one
3. Sample a new cluster assignment according to these weights
4. Update the cluster assignments and related data structures

## 3. Cluster Update - Metropolis-Hastings Step

The cluster update follows a Gibbs sampling approach within a Metropolis-Hastings framework:

1. Choose a random list/order $e$
2. Remove it from its current cluster
3. Calculate the probability of assigning it to each existing cluster or creating a new one
4. Sample a new cluster assignment
5. Accept or reject according to Metropolis-Hastings criteria

### Mathematical Details

#### Step 1-2: Remove List from Current Cluster

For a selected list $e$ currently in cluster $k$, we decrement $n_k$ and potentially remove the cluster if it becomes empty:

$$
n_k = n_k - 1
$$

#### Step 3: Calculate Assignment Probabilities

For each existing cluster $j$, the log-weight is:

$$
\log w_j = \log(n_j - \alpha) + \log p(y_e | U_a^j)
$$

Where:

- $n_j$ is the count of lists in cluster $j$ (after removal of $e$)
- $\log p(y_e | U_a^j)$ is the log-likelihood of observing order $y_e$ given the latent utilities $U_a^j$ of cluster $j$

For a new cluster, the log-weight is:

$$
\log w_{new} = \log(\theta + \alpha \cdot |\text{clusters}|) + \log p(y_e | U_a^{new})
$$

Where:

- $U_a^{new}$ are newly sampled latent utilities for a potential new cluster
- $\log p(y_e | U_a^{new})$ is the likelihood of the observed order under these new utilities

#### Step 4: Sample New Assignment

We normalize the weights and sample a new cluster assignment:

$$
P(k_{new} = j) = \frac{\exp(\log w_j)}{\sum_i \exp(\log w_i)}
$$

#### Step 5: Update Data Structures

If a new cluster is created, we:

1. Generate new latent positions $U_a^{new}$ from the hierarchical prior:

   $$
   U_a^{new} \sim \mathcal{N}(\tau U_0, (1-\tau^2)\Sigma_\rho)
   $$
2. Recalculate the partial orders based on these utilities

### Derivation of the Acceptance Rate

For the Metropolis-Hastings algorithm, the acceptance probability for a proposed move from state $S$ to state $S'$ is:

$$
\alpha(S \to S') = \min\left(1, \frac{\pi(S') q(S|S')}{\pi(S) q(S'|S)}\right)
$$

where $\pi(S)$ is the target distribution and $q(S'|S)$ is the proposal distribution.

For our cluster update, where we're moving a list $e$ from cluster $k$ to cluster $j$:

1. **Target Distribution Ratio**:

   $$
   \frac{\pi(S')}{\pi(S)} = \frac{\pi(c', U|Y)}{\pi(c, U|Y)} = \frac{p(Y|c', U) \cdot \pi_C(c'|\alpha, \theta)}{p(Y|c, U) \cdot \pi_C(c|\alpha, \theta)}
   $$

   Let's examine each component:

   a. **Likelihood Ratio**: Only the likelihood of list $e$ changes when we move it from cluster $k$ to $j$:

   $$
   ac{p(Y|c', U)}{p(Y|c, U)} = \frac{p(y_e|U_j)}{p(y_e|U_k)}
   $$

   b. **Prior Ratio**: For the CRP prior when moving from cluster $k$ to an existing cluster $j$:

   $$
   ac{\pi_C(c'|\alpha, \theta)}{\pi_C(c|\alpha, \theta)} = \frac{(n'_j - \alpha) \cdot (n'_k - \alpha \cdot \mathbb{I}[n'_k > 0])}{(n_j - \alpha) \cdot (n_k - \alpha)}
   $$

   Where $n'_j = n_j + 1$ and $n'_k = n_k - 1$, and $\mathbb{I}$ is the indicator function.

   For a move to a new cluster:

   $$
   ac{\pi_C(c'|\alpha, \theta)}{\pi_C(c|\alpha, \theta)} = \frac{(\theta + \alpha \cdot K') \cdot (n'_k - \alpha \cdot \mathbb{I}[n'_k > 0])}{(n_k - \alpha) \cdot (\theta + \alpha \cdot K)}
   $$

   Where $K'$ is the new number of clusters.
2. **Proposal Distribution Ratio**:

   Since we sample the new cluster assignment based on normalized weights calculated from the CRP prior and likelihood:

   $$
   q(c'|c) = \frac{\exp(\log w_j)}{\sum_i \exp(\log w_i)}
   $$

   $$
   q(c|c') = \frac{\exp(\log w'_k)}{\sum_i \exp(\log w'_i)}
   $$

   Where the weights incorporate both prior and likelihood components.
3. **Final Acceptance Probability**:

   For a move from cluster $k$ to an existing cluster $j$:

   $$
   \alpha(c \to c') = \min\left(1, \frac{p(y_e|U_j) \cdot (n_j + 1 - \alpha) \cdot (n_k - 1 - \alpha \cdot \mathbb{I}[n_k > 1]) \cdot q(c|c')}{p(y_e|U_k) \cdot (n_j - \alpha) \cdot (n_k - \alpha) \cdot q(c'|c)}\right)
   $$

   For a move to a new cluster:

   $$
   \alpha(c \to c') = \min\left(1, \frac{p(y_e|U_{new}) \cdot (\theta + \alpha \cdot (K+1)) \cdot (n_k - 1 - \alpha \cdot \mathbb{I}[n_k > 1]) \cdot q(c|c')}{p(y_e|U_k) \cdot (n_k - \alpha) \cdot (\theta + \alpha \cdot K) \cdot q(c'|c)}\right)
   $$

   If we are using a Gibbs sampler (sampling directly from the conditional posterior), the proposal matches the target conditional distribution and the acceptance rate simplifies to 1.

## 5. Overall Algorithm

For each iteration of the MCMC:

1. With probability $p_{cluster}$, perform a cluster update
2. Sample a list $e$ randomly
3. Remove $e$ from its current cluster
4. Calculate the weight for each existing cluster and a new cluster
5. Sample a new cluster assignment based on these weights
6. Update all data structures accordingly
7. Accept or reject the move based on Metropolis-Hastings ratio

This framework allows the model to automatically discover the appropriate number of clusters in the data, with each cluster representing a distinct pattern of preferences.
