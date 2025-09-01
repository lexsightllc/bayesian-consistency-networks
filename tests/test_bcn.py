"""Test script for the improved Bayesian Consistency Network."""

import numpy as np
from bcn import BayesianConsistencyNetwork

def test_simple_contradiction():
    """Test with simple contradiction and exclusion constraint."""
    print("\n=== Test 1: Simple Contradiction with Exclusion ===")
    bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)
    
    # Add observations
    bcn.add_observation(0, 0, 1)  # Source 0: prop 0 is true
    bcn.add_observation(0, 1, 1)  # Source 0: prop 1 is true
    bcn.add_observation(1, 0, 1)  # Source 1: prop 0 is true
    bcn.add_observation(1, 1, 0)  # Source 1: prop 1 is false
    
    # Add exclusion constraint (can't both be true)
    bcn.add_constraint('exclusion', [0, 1], strength=2.0)
    
    # Run inference
    bcn.run_inference()
    
    # Print results
    print("\nResults:")
    for i, prop in enumerate(bcn.propositions):
        print(f"  Proposition {i}: {prop.belief:.3f}")
    
    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
              f"Specificity: {source.specificity:.3f}")
    
    scores = bcn.get_contradiction_scores()
    print(f"\nContradiction scores: {[f'{s:.3f}' for s in scores]}")

def test_entailment():
    """Test with entailment constraint."""
    print("\n=== Test 2: Entailment (p ⇒ q) ===")
    bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)
    
    # Add observations
    bcn.add_observation(0, 0, 1)  # Source 0: p is true
    bcn.add_observation(1, 1, 0)  # Source 1: q is false
    
    # Add entailment constraint (p ⇒ q)
    bcn.add_constraint('entailment', [0, 1], strength=2.0)
    
    # Run inference
    bcn.run_inference()
    
    # Print results
    print("\nResults:")
    for i, prop in enumerate(bcn.propositions):
        print(f"  Proposition {i}: {prop.belief:.3f}")
    
    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
              f"Specificity: {source.specificity:.3f}")
    
    scores = bcn.get_contradiction_scores()
    print(f"\nContradiction scores: {[f'{s:.3f}' for s in scores]}")

def demo_equivalence():
    """Demo of equivalence constraint (A ⇔ B)."""
    print("\n=== Demo 1: Equivalence (A ⇔ B) ===")
    print("Sources:")
    print("  Source 0: A=1, B=0 (contradiction)")
    print("  Source 1: A=1, B=1 (consistent with equivalence)")
    
    bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)
    
    # Add observations
    bcn.add_observation(0, 0, 1)  # Source 0: A=1
    bcn.add_observation(0, 1, 0)  # Source 0: B=0
    bcn.add_observation(1, 0, 1)  # Source 1: A=1
    bcn.add_observation(1, 1, 1)  # Source 1: B=1
    
    # Add equivalence constraint
    bcn.add_constraint('equivalence', [0, 1], strength=2.0)
    
    # Run inference
    bcn.run_inference()
    
    # Print results
    print("\nResults:")
    print(f"  Belief A: {bcn.propositions[0].belief:.3f}")
    print(f"  Belief B: {bcn.propositions[1].belief:.3f}")
    
    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
              f"Specificity: {source.specificity:.3f}")

def demo_exactly_one_of_three():
    """Demo of 'exactly one of three' using pairwise exclusions."""
    print("\n=== Demo 2: Exactly-One-of-Three ===")
    print("Sources:")
    print("  Source 0 (strong): C0=1, C1=0, C2=0")
    print("  Source 1 (mediocre): C0=1, C1=1, C2=0")
    print("  Source 2 (noisy): C0=0, C1=1, C2=1")
    
    bcn = BayesianConsistencyNetwork(n_propositions=3, n_sources=3)
    
    # Add observations for Source 0
    bcn.add_observation(0, 0, 1)  # C0=1
    bcn.add_observation(0, 1, 0)  # C1=0
    bcn.add_observation(0, 2, 0)  # C2=0
    
    # Add observations for Source 1
    bcn.add_observation(1, 0, 1)  # C0=1
    bcn.add_observation(1, 1, 1)  # C1=1
    bcn.add_observation(1, 2, 0)  # C2=0
    
    # Add observations for Source 2
    bcn.add_observation(2, 0, 0)  # C0=0
    bcn.add_observation(2, 1, 1)  # C1=1
    bcn.add_observation(2, 2, 1)  # C2=1
    
    # Add pairwise exclusion constraints
    bcn.add_constraint('exclusion', [0, 1], strength=2.0)
    bcn.add_constraint('exclusion', [0, 2], strength=2.0)
    bcn.add_constraint('exclusion', [1, 2], strength=2.0)
    
    # Add a weak prior to encourage at least one true
    for prop in bcn.propositions:
        prop.prior = 0.6
    
    # Run inference
    bcn.run_inference()
    
    # Print results
    print("\nResults:")
    for i, prop in enumerate(bcn.propositions):
        print(f"  Belief C{i}: {prop.belief:.3f}")
    
    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
              f"Specificity: {source.specificity:.3f}")
    
    scores = bcn.get_contradiction_scores()
    print(f"\nContradiction scores: {[f'{s:.3f}' for s in scores]}")

if __name__ == "__main__":
    test_simple_contradiction()
    test_entailment()
    demo_equivalence()
    demo_exactly_one_of_three()
