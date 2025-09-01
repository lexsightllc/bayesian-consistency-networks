"""Tests for cardinality constraints in Bayesian Consistency Networks."""

import pytest

from bcn import BayesianConsistencyNetwork


def test_cardinality_k1_lowers_sum():
    """k=1 constraint should push sum of beliefs below ~1.3."""
    print("\n=== Test: Cardinality k=1 ===")

    bcn = BayesianConsistencyNetwork(n_propositions=3, n_sources=1)

    for i in range(3):
        bcn.add_observation(0, i, 1)

    bcn.add_constraint("cardinality", [0, 1, 2], strength=2.0, cardinality=1)

    bcn.run_inference(max_iter=50, damping=0.7)

    beliefs = [p.belief for p in bcn.propositions]
    total = sum(beliefs)

    print(f"\nBeliefs: {[f'{b:.3f}' for b in beliefs]}")
    assert total < 1.3, "Sum of beliefs should be pushed below 1.3"


def test_cardinality_with_exclusion():
    """Test interaction between cardinality and exclusion constraints."""
    print("\n=== Test: Cardinality with Exclusion ===")

    bcn = BayesianConsistencyNetwork(n_propositions=3, n_sources=2)

    # Add observations that would make all propositions likely true
    for i in range(3):
        bcn.add_observation(0, i, 1)
        bcn.add_observation(1, i, 1)

    # Add both cardinality and pairwise exclusions
    bcn.add_constraint("cardinality", [0, 1, 2], strength=2.0, cardinality=1)
    bcn.add_constraint("exclusion", [0, 1], strength=2.0)
    bcn.add_constraint("exclusion", [1, 2], strength=2.0)
    bcn.add_constraint("exclusion", [0, 2], strength=2.0)

    # Run inference
    bcn.run_inference(max_iter=50, damping=0.7)

    # Get results
    beliefs = [p.belief for p in bcn.propositions]
    scores = bcn.get_contradiction_scores()

    print(f"\nBeliefs: {[f'{b:.3f}' for b in beliefs]}")
    print(f"Contradiction scores: {[f'{s:.3f}' for s in scores]}")

    # Verify that at most one belief is high
    high_beliefs = sum(1 for b in beliefs if b > 0.7)
    assert high_beliefs <= 1, "At most one proposition should have high belief"

    # The cardinality constraint should be satisfied (since exclusions enforce it)
    assert scores[0] < 0.1, "Cardinality constraint should be satisfied"


def test_cardinality_k0_enforces_exclusion():
    """Cardinality k=0 should force both beliefs low."""
    bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=1)

    bcn.add_observation(0, 0, 1)
    bcn.add_observation(0, 1, 1)

    bcn.add_constraint("cardinality", [0, 1], strength=2.0, cardinality=0)

    bcn.run_inference(max_iter=50, damping=0.7)

    beliefs = [p.belief for p in bcn.propositions]

    assert all(b < 0.5 for b in beliefs)


if __name__ == "__main__":
    test_cardinality_constraint()
    test_cardinality_with_exclusion()
