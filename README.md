# Bayesian Consistency Networks (BCN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python implementation of Bayesian Consistency Networks for resolving contradictions in binary claims from multiple sources using belief propagation and variational EM.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Overview

Bayesian Consistency Networks (BCN) is a probabilistic graphical model that serves as a robust framework for resolving contradictions in binary claims. It acts as a 'probabilistic referee' that:

### Key Features

- **Soft Logical Constraints**: Incorporates flexible rules between propositions:
  - **Exclusion (A ⊥ B)**: Penalizes A ∧ B (mutual exclusion)
  - **Entailment (A ⇒ B)**: Directional implication that specifically penalizes A=1 ∧ B=0
  - **Equivalence (A ⇔ B)**: Penalizes A ⊕ B (XOR)
  - **Cardinality / Top-k**: At most *k* of a set may be true
  
- **Stable Learning**: Uses damping (default 0.5) to ensure stable, oscillation-free updates by moving only partway each iteration
  
- **Contradiction Scoring**: Each constraint reports a score (0-1) indicating how strained it is under current beliefs, aiding in debugging and trust analysis
  
- **Source Reliability**: Automatically estimates and adjusts source reliability (sensitivity/specificity) during inference
- **Correlated-Source Down-weighting**: Optionally reduce influence of near-duplicate sources

1. Infers the most likely truth values of propositions given noisy observations from multiple sources
2. Incorporates soft logical constraints (exclusion, entailment, equivalence) to resolve contradictions
3. Estimates source reliability parameters (sensitivity and specificity) during inference
4. Provides contradiction scores to identify the most problematic constraints

## Installation

```bash
git clone https://github.com/yourusername/bayesian-consistency-networks.git
cd bayesian-consistency-networks
pip install -e .
```

## Usage

### Practical Example: Fact-Checking News

Here's how BCN can resolve conflicting claims about a news event:

```python
from bcn import BayesianConsistencyNetwork

# Initialize with 2 claims and 2 sources
bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)

# Source 0 (reliable but not perfect) says:
# - Claim 0: "The event happened on Monday" (True)
# - Claim 1: "The event was canceled" (False)
bcn.add_observation(0, 0, 1)  # Source 0, Claim 0 = true
bcn.add_observation(0, 1, 1)  # Source 0, Claim 1 = true

# Source 1 (less reliable) says:
# - Claim 0: "The event happened on Monday" (True)
# - Claim 1: "The event was not canceled" (True)
bcn.add_observation(1, 0, 1)  # Source 1, Claim 0 = true
bcn.add_observation(1, 1, 0)  # Source 1, Claim 1 = false

# Add constraint: These claims are mutually exclusive
# (An event can't be both canceled and not canceled)
bcn.add_constraint('exclusion', [0, 1], strength=2.0)

# Run inference
bcn.run_inference()

# Check results
for i, prop in enumerate(bcn.propositions):
    print(f"Claim {i}: P(True) = {prop.belief:.3f}")

# Check contradiction scores
scores = bcn.get_contradiction_scores()
print(f"Contradiction score: {scores[0]:.3f} (closer to 1 means more contradictory)")

# Check source reliability
for i, source in enumerate(bcn.sources):
    print(f"Source {i}: sensitivity={source.sensitivity:.3f}, specificity={source.specificity:.3f}")
```

### Basic Example

```python
from bcn import BayesianConsistencyNetwork

# Create a BCN with 2 propositions and 2 sources
bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)

# Add observations (source_idx, prop_idx, value)
bcn.add_observation(0, 0, 1)  # Source 0 says prop 0 is true
bcn.add_observation(0, 1, 1)  # Source 0 says prop 1 is true
bcn.add_observation(1, 0, 1)  # Source 1 says prop 0 is true
bcn.add_observation(1, 1, 0)  # Source 1 says prop 1 is false

# Add constraints
bcn.add_constraint('exclusion', [0, 1], strength=2.0)  # Props 0 and 1 cannot both be true

# Run inference
bcn.run_inference()

# Get beliefs
for i, prop in enumerate(bcn.propositions):
    print(f"Proposition {i}: P(True) = {prop.belief:.3f}")

# Get contradiction scores
scores = bcn.get_contradiction_scores()
print(f"Contradiction scores: {scores}")
```

### Cardinality / Top-k Example

```python
from bcn import BayesianConsistencyNetwork

# Enable correlation_penalty to down-weight similar sources
bcn = BayesianConsistencyNetwork(
    n_propositions=3, n_sources=1, correlation_penalty=True
)

for i in range(3):
    bcn.add_observation(0, i, 1)

# At most one proposition can be true (top-1)
bcn.add_constraint('topk', [0, 1, 2], strength=2.0, cardinality=1)
bcn.run_inference()
print([p.belief for p in bcn.propositions])
print(bcn.get_metrics())
```

### Running Tests

```bash
python -m pytest tests/
```

## How It Works

BCN combines belief propagation with variational EM in an iterative process:

1. **Message Passing**: Each source's claims are weighted by their reliability
2. **Constraint Satisfaction**: Soft rules push beliefs toward consistency
3. **Source Reliability Update**: Sources are re-weighted based on agreement with consensus
4. **Damping**: Updates are smoothed (default: 0.5) to prevent oscillation
5. **Convergence**: Process repeats until beliefs stabilize

### Technical Details

#### Key Components

1. **Propositions**: Binary variables representing claims that can be true or false
2. **Sources**: Information providers with unknown reliability (sensitivity and specificity)
3. **Observations**: Binary claims made by sources about propositions
4. **Constraints**: Soft logical rules between propositions:
   - **Exclusion (A ⊥ B)**: Penalizes A ∧ B (mutual exclusion)
   - **Entailment (A ⇒ B)**: Directional; penalizes A=1 ∧ B=0
   - **Equivalence (A ⇔ B)**: Penalizes A ⊕ B (XOR)

#### Inference Process

1. **Initialization**: Start with uniform beliefs and source parameters
2. **E-step**: Update beliefs using current source parameters
3. **M-step**: Update source parameters using current beliefs
4. **Damping**: Apply smoothing to prevent oscillation (default: 0.5)
5. **Convergence**: Stop when beliefs change by less than tolerance

## Advanced Usage

### Tuning Parameters

```python
# Run inference with custom parameters
bcn.run_inference(
    max_iter=100,      # Max BP iterations per E-step
    tol=1e-4,          # Convergence tolerance for BP
    max_em_iter=10,    # Max EM iterations
    em_tol=1e-3,       # Convergence tolerance for EM
    damping=0.5        # Damping factor (0.5 = half of update applied each step)
)

# Get detailed diagnostics
contradiction_scores = bcn.get_contradiction_scores()  # Per-constraint scores [0,1)
for i, score in enumerate(contradiction_scores):
    print(f"Constraint {i} contradiction: {score:.3f}")
```

### Custom Priors

You can specify custom Beta priors for source reliability parameters:

```python
from bcn import BayesianConsistencyNetwork, Source

# Create a BCN with custom source priors
bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)

# Customize source parameters
bcn.sources[0] = Source(
    sensitivity=0.9,  # Initial guess
    specificity=0.9,   # Initial guess
    alpha_a=2.0,      # Beta prior for sensitivity
    beta_a=2.0,
    alpha_b=2.0,      # Beta prior for specificity
    beta_b=2.0
)
```

### Tuning Parameters

```python
# Run inference with custom parameters
bcn.run_inference(
    max_iter=100,      # Max BP iterations per E-step
    tol=1e-4,          # Convergence tolerance for BP
    max_em_iter=10,    # Max EM iterations
    em_tol=1e-3,       # Convergence tolerance for EM
    damping=0.5        # Damping factor for BP updates
)
```

## References

1. Pasternack, J., & Roth, D. (2010). "Latent Credibility Analysis."
2. Pasternack, J., & Roth, D. (2013). "Making Better Informed Trust Decisions with Generalized Fact-Finding."
3. Zhao, B., Rubinstein, B. I., Gemmell, J., & Han, J. (2012). "A Bayesian Approach to Discovering Truth from Conflicting Sources for Data Integration." 

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
