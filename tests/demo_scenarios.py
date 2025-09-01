"""
BCN Demonstration Scenarios

This file contains practical demonstrations of Bayesian Consistency Networks
with different constraint types and source configurations.
"""
import numpy as np
from bcn import BayesianConsistencyNetwork

def demo_equivalence():
    """Demo of equivalence constraint (A ⇔ B).
    
    Scenario:
    - Constraint: A and B should have the same truth value
    - Source 0: says A=1, B=0 (contradiction)
    - Source 1: says A=1, B=1 (consistent with equivalence)
    """
    print("\n=== Equivalence Constraint Demo (A ⇔ B) ===")
    print("Sources:")
    print("  Source 0: A=1, B=0")
    print("  Source 1: A=1, B=1")
    
    # Create BCN with 2 propositions and 2 sources
    bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)
    
    # Add observations
    bcn.add_observation(0, 0, 1)  # Source 0: A=1
    bcn.add_observation(0, 1, 0)  # Source 0: B=0
    bcn.add_observation(1, 0, 1)  # Source 1: A=1
    bcn.add_observation(1, 1, 1)  # Source 1: B=1
    
    # Add equivalence constraint
    bcn.add_constraint('equivalence', [0, 1], strength=2.0)
    
    # Run inference
    bcn.run_inference(max_iter=30, max_em_iter=5)
    
    # Print results
    print("\nResults:")
    print(f"  Belief A: {bcn.propositions[0].belief:.3f}")
    print(f"  Belief B: {bcn.propositions[1].belief:.3f}")
    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
              f"Specificity: {source.specificity:.3f}")

def demo_exactly_one_of_three():
    """Demo of 'exactly one of three' using pairwise exclusions.
    
    Scenario:
    - Claims: C0, C1, C2 (e.g., "which city is the capital?")
    - Constraint: At most one true (pairwise exclusion between each pair)
    - Sources:
      - S0 (strong): C0=1, C1=0, C2=0
      - S1 (mediocre): C0=1, C1=1, C2=0
      - S2 (noisier): C0=0, C1=1, C2=1
    """
    print("\n=== Exactly-One-of-Three Demo ===")
    print("Sources:")
    print("  Source 0: C0=1, C1=0, C2=0")
    print("  Source 1: C0=1, C1=1, C2=0")
    print("  Source 2: C0=0, C1=1, C2=1")
    
    # Create BCN with 3 propositions and 3 sources
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
    bcn.add_constraint('exclusion', [0, 1], strength=2.0)  # C0 and C1 can't both be true
    bcn.add_constraint('exclusion', [0, 2], strength=2.0)  # C0 and C2 can't both be true
    bcn.add_constraint('exclusion', [1, 2], strength=2.0)  # C1 and C2 can't both be true
    
    # Add a weak prior to encourage at least one true (optional)
    for prop in bcn.propositions:
        prop.prior = 0.6  # Slight bias toward true
    
    # Run inference with more iterations for convergence
    bcn.run_inference(max_iter=50, max_em_iter=5)
    
    # Print results
    print("\nResults:")
    for i, prop in enumerate(bcn.propositions):
        print(f"  Belief C{i}: {prop.belief:.3f}")
    
    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
              f"Specificity: {source.specificity:.3f}")

if __name__ == "__main__":
    demo_equivalence()
    demo_exactly_one_of_three()
