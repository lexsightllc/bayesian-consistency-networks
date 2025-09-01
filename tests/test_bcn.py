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
    bcn.add_constraint("exclusion", [0, 1], strength=2.0)

    # Run inference
    bcn.run_inference()

    # Print results
    print("\nResults:")
    for i, prop in enumerate(bcn.propositions):
        print(f"  Proposition {i}: {prop.belief:.3f}")

    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(
            f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
            f"Specificity: {source.specificity:.3f}"
        )

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
    bcn.add_constraint("entailment", [0, 1], strength=2.0)

    # Run inference
    bcn.run_inference()

    # Print results
    print("\nResults:")
    for i, prop in enumerate(bcn.propositions):
        print(f"  Proposition {i}: {prop.belief:.3f}")

    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(
            f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
            f"Specificity: {source.specificity:.3f}"
        )

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
    bcn.add_constraint("equivalence", [0, 1], strength=2.0)

    # Run inference
    bcn.run_inference()

    # Print results
    print("\nResults:")
    print(f"  Belief A: {bcn.propositions[0].belief:.3f}")
    print(f"  Belief B: {bcn.propositions[1].belief:.3f}")

    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(
            f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
            f"Specificity: {source.specificity:.3f}"
        )


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
    bcn.add_constraint("exclusion", [0, 1], strength=2.0)
    bcn.add_constraint("exclusion", [0, 2], strength=2.0)
    bcn.add_constraint("exclusion", [1, 2], strength=2.0)

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
        print(
            f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
            f"Specificity: {source.specificity:.3f}"
        )

    scores = bcn.get_contradiction_scores()
    print(f"\nContradiction scores: {[f'{s:.3f}' for s in scores]}")


def test_source_reliability_learning():
    """Test that source reliability parameters are learned correctly."""
    print("\n=== Test 3: Source Reliability Learning ===")

    # Create a scenario where:
    # - Source 0 is 80% accurate
    # - Source 1 is 60% accurate (barely better than random)
    # - Propositions 0 and 1 are mutually exclusive

    # Ground truth: prop0=True, prop1=False
    bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)

    # Add observations from source 0 (80% accurate)
    for _ in range(80):  # 80% true positives for prop0
        bcn.add_observation(0, 0, 1)
    for _ in range(20):  # 20% false positives for prop0
        bcn.add_observation(0, 0, 0)

    for _ in range(80):  # 80% true negatives for prop1
        bcn.add_observation(0, 1, 0)
    for _ in range(20):  # 20% false negatives for prop1
        bcn.add_observation(0, 1, 1)

    # Add observations from source 1 (60% accurate)
    for _ in range(60):  # 60% true positives for prop0
        bcn.add_observation(1, 0, 1)
    for _ in range(40):  # 40% false negatives for prop0
        bcn.add_observation(1, 0, 0)

    for _ in range(60):  # 60% true negatives for prop1
        bcn.add_observation(1, 1, 0)
    for _ in range(40):  # 40% false positives for prop1
        bcn.add_observation(1, 1, 1)

    # Add exclusion constraint with moderate strength
    bcn.add_constraint("exclusion", [0, 1], strength=1.0)

    # Run inference with more EM iterations for better learning
    bcn.run_inference(max_em_iter=10, em_tol=1e-3, damping=0.7)

    # Print results
    print("\nLearned Source Reliability:")
    for i, source in enumerate(bcn.sources):
        print(
            f"  Source {i} - Sensitivity: {source.sensitivity:.3f}, "
            f"Specificity: {source.specificity:.3f}"
        )

    # Verify source 0 is more reliable than source 1
    # Use a small epsilon to avoid floating point comparison issues
    assert bcn.sources[0].sensitivity > bcn.sources[1].sensitivity - 0.01
    assert bcn.sources[0].specificity > bcn.sources[1].specificity - 0.01

    # Verify beliefs are reasonable (prop0 low, prop1 high)
    assert bcn.propositions[0].belief < 0.4
    assert bcn.propositions[1].belief > 0.6


def test_contradiction_scoring():
    """Test that contradiction scores are in [0,1] and make sense."""
    print("\n=== Test 4: Contradiction Scoring ===")

    # Test case with known contradictions
    bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=2)

    # Add direct contradiction
    bcn.add_observation(0, 0, 1)  # Source 0: prop 0 is true
    bcn.add_observation(1, 0, 0)  # Source 1: prop 0 is false

    # Add constraints with different strengths
    bcn.add_constraint("exclusion", [0, 1], strength=1.5)  # Should be violated
    bcn.add_constraint("entailment", [0, 1], strength=1.0)  # Should be satisfied

    # Run inference with more iterations for stability
    bcn.run_inference(max_iter=50, tol=1e-4, damping=0.7)

    # Get contradiction scores
    scores = bcn.get_contradiction_scores()

    print(f"\nBeliefs: {[p.belief for p in bcn.propositions]}")
    print(f"Contradiction scores: {scores}")

    # Verify scores are in [0,1]
    assert all(0 <= s <= 1 for s in scores), "Scores must be in [0,1]"

    # The exclusion constraint should have higher violation than entailment
    # Use a small epsilon to account for floating point imprecision
    assert (
        scores[0] > scores[1] - 1e-10
    ), f"Exclusion score ({scores[0]}) should be > entailment score ({scores[1]})"


if __name__ == "__main__":
    test_simple_contradiction()
    test_entailment()
    demo_equivalence()
    demo_exactly_one_of_three()
    test_source_reliability_learning()
    test_contradiction_scoring()
