# Bayesian Consistency Networks (BCN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python implementation of Bayesian Consistency Networks for resolving contradictions in binary claims from multiple sources using belief propagation and variational EM.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Overview

Bayesian Consistency Networks (BCN) is a probabilistic graphical model that:

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

### Running Tests

```bash
python -m pytest tests/
```

## Model Details

### Key Components

1. **Propositions**: Binary variables representing claims that can be true or false
2. **Sources**: Information providers with unknown reliability (sensitivity and specificity)
3. **Observations**: Binary claims made by sources about propositions
4. **Constraints**: Soft logical rules between propositions (exclusion, entailment, equivalence)

### Inference

The model uses:
- **Belief Propagation (BP)**: For approximate inference over the factor graph
- **Variational EM**: For jointly estimating source parameters and beliefs
- **Damped Updates**: For numerical stability during belief propagation

## Advanced Usage

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

1. Pasternack, J., & Roth, D. (2010). "Latent Credibility Analysis." WWW.
2. Pasternack, J., & Roth, D. (2013). "Making Better Informed Trust Decisions with Generalized Fact-Finding." IJCAI.
3. Zhao, B., Rubinstein, B. I., Gemmell, J., & Han, J. (2012). "A Bayesian Approach to Discovering Truth from Conflicting Sources for Data Integration." PVLDB.

## License

MIT
