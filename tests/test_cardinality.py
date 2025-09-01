"""Tests for cardinality constraints in Bayesian Consistency Networks."""

import pytest
from bcn import BayesianConsistencyNetwork

def test_cardinality_constraint():
    """Test that cardinality constraints work as expected."""
    print("\n=== Test: Cardinality Constraint ===")
    
    # Create a scenario where we have 3 propositions and want at most 2 to be true
    bcn = BayesianConsistencyNetwork(n_propositions=3, n_sources=2)
    
    # Add some observations that would make all 3 propositions likely true
    bcn.add_observation(0, 0, 1)  # Source 0: prop 0 is true
    bcn.add_observation(0, 1, 1)  # Source 0: prop 1 is true
    bcn.add_observation(1, 1, 1)  # Source 1: prop 1 is true
    bcn.add_observation(1, 2, 1)  # Source 1: prop 2 is true
    
    # Add a cardinality constraint: at most 2 of the 3 can be true
    bcn.add_constraint('cardinality', [0, 1, 2], strength=2.0, cardinality=2)
    
    # Run inference
    bcn.run_inference(max_iter=50, damping=0.7)
    
    # Get beliefs and scores
    beliefs = [p.belief for p in bcn.propositions]
    scores = bcn.get_contradiction_scores()
    
    print(f"\nBeliefs: {[f'{b:.3f}' for b in beliefs]}")
    print(f"Contradiction score: {scores[0]:.3f}")
    
    # Verify that the constraint is somewhat violated (since we have evidence for all 3)
    assert scores[0] > 0.1, "Cardinality constraint should be somewhat violated"
    
    # Verify that the beliefs are pushed down by the constraint
    assert sum(beliefs) <= 2.5, "Expected sum of beliefs to be pushed below 2.5"

def test_cardinality_with_exclusion():
    """Test interaction between cardinality and exclusion constraints."""
    print("\n=== Test: Cardinality with Exclusion ===")
    
    bcn = BayesianConsistencyNetwork(n_propositions=3, n_sources=2)
    
    # Add observations that would make all propositions likely true
    for i in range(3):
        bcn.add_observation(0, i, 1)
        bcn.add_observation(1, i, 1)
    
    # Add both cardinality and pairwise exclusions
    bcn.add_constraint('cardinality', [0, 1, 2], strength=2.0, cardinality=1)
    bcn.add_constraint('exclusion', [0, 1], strength=2.0)
    bcn.add_constraint('exclusion', [1, 2], strength=2.0)
    bcn.add_constraint('exclusion', [0, 2], strength=2.0)
    
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

if __name__ == "__main__":
    test_cardinality_constraint()
    test_cardinality_with_exclusion()
