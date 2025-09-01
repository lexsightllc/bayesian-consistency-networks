"""
Bayesian Consistency Networks (BCN) - Improved Implementation

A refined implementation with:
- Corrected entailment directionality
- Stable EM updates for source parameters
- Numerically robust LLRs
- Normalized contradiction scores in [0,1]
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

# Small constant for numerical stability
EPSILON = 1e-10

def stable_sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def stable_log(x: float) -> float:
    """Numerically stable log with clipping."""
    return math.log(max(x, EPSILON))

@dataclass
class Source:
    """Represents an information source with reliability parameters."""
    sensitivity: float  # true positive rate (a_s)
    specificity: float  # true negative rate (b_s)
    alpha_a: float = 1.0  # Beta prior for sensitivity
    beta_a: float = 1.0
    alpha_b: float = 1.0  # Beta prior for specificity
    beta_b: float = 1.0

@dataclass
class Proposition:
    """Represents a proposition with a prior probability."""
    prior: float  # π_p: prior probability that the proposition is true
    belief: float  # Current belief b(T_p = 1)
    neighbors: Set[int] = field(default_factory=set)  # Indices of connected propositions

@dataclass
class Constraint:
    """Represents a soft logical constraint between propositions."""
    constraint_type: str  # 'exclusion', 'entailment', or 'equivalence'
    prop_indices: List[int]  # Indices of constrained propositions
    strength: float  # γ: strength of the constraint

class BayesianConsistencyNetwork:
    """
    Improved Bayesian Consistency Network for contradiction resolution.
    
    Key improvements:
    - Correct entailment directionality (p ⇒ q)
    - Stable EM updates for source parameters
    - Numerically robust LLRs with clipping
    - Contradiction scores normalized to [0,1]
    """
    
    def __init__(self, n_propositions: int, n_sources: int):
        """Initialize the BCN with given number of propositions and sources."""
        self.propositions = [
            Proposition(prior=0.5, belief=0.5) 
            for _ in range(n_propositions)
        ]
        self.sources = [
            Source(sensitivity=0.8, specificity=0.8) 
            for _ in range(n_sources)
        ]
        self.constraints: List[Constraint] = []
        self.observations: Dict[Tuple[int, int], int] = {}
    
    def add_observation(self, source_idx: int, prop_idx: int, value: int) -> None:
        """Record an observation from a source about a proposition."""
        if value not in {0, 1}:
            raise ValueError("Observation value must be 0 or 1")
        self.observations[(source_idx, prop_idx)] = value
    
    def add_constraint(self, constraint_type: str, prop_indices: List[int], 
                      strength: float = 1.0) -> None:
        """Add a soft logical constraint between propositions.
        
        Args:
            constraint_type: Type of constraint. One of:
                - 'exclusion' (A ⊥ B): Penalizes A ∧ B (mutual exclusion)
                - 'entailment' (A ⇒ B): Directional; penalizes A=1 ∧ B=0
                - 'equivalence' (A ⇔ B): Penalizes A ⊕ B (XOR)
            prop_indices: List of exactly two proposition indices [i, j]
            strength: Strength of the constraint (higher = stronger)
            
        Raises:
            ValueError: If constraint_type is invalid or prop_indices has wrong length
        """
        if constraint_type not in {'exclusion', 'entailment', 'equivalence'}:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        if len(prop_indices) != 2:
            raise ValueError("Currently only binary constraints are supported")
        
        constraint = Constraint(
            constraint_type=constraint_type,
            prop_indices=prop_indices,
            strength=strength
        )
        self.constraints.append(constraint)
        
        # Update neighbors for message passing
        i, j = prop_indices
        self.propositions[i].neighbors.add(j)
        self.propositions[j].neighbors.add(i)
    
    def _compute_observation_llr(self, source_idx: int, prop_idx: int) -> float:
        """Compute log-likelihood ratio for an observation."""
        if (source_idx, prop_idx) not in self.observations:
            return 0.0
            
        source = self.sources[source_idx]
        y = self.observations[(source_idx, prop_idx)]
        
        # Clip probabilities to avoid numerical issues
        a = max(min(source.sensitivity, 1 - EPSILON), EPSILON)
        b = max(min(source.specificity, 1 - EPSILON), EPSILON)
        
        if y == 1:
            return stable_log(a / (1 - b))
        else:
            return stable_log((1 - a) / b)
    
    def _compute_constraint_message(self, constraint: Constraint, target_idx: int) -> float:
        """Compute the LLR message from a constraint to a target proposition."""
        if constraint.constraint_type == 'exclusion':
            # p ∧ q is penalized
            p, q = constraint.prop_indices
            u = self.propositions[q].belief if p == target_idx else self.propositions[p].belief
            u = max(min(u, 1 - EPSILON), EPSILON)  # Clip to avoid numerical issues
            return stable_log(u * math.exp(-constraint.strength) + (1 - u))
            
        elif constraint.constraint_type == 'entailment':
            p, q = constraint.prop_indices
            if target_idx == p:  # Message to p (antecedent)
                u = max(min(self.propositions[q].belief, 1 - EPSILON), EPSILON)
                return stable_log(u + (1 - u) * math.exp(-constraint.strength))
            else:  # Message to q (consequent)
                u = max(min(self.propositions[p].belief, 1 - EPSILON), EPSILON)
                return -stable_log((1 - u) + u * math.exp(-constraint.strength))
                
        elif constraint.constraint_type == 'equivalence':
            p, q = constraint.prop_indices
            u = self.propositions[q].belief if p == target_idx else self.propositions[p].belief
            u = max(min(u, 1 - EPSILON), EPSILON)  # Clip to avoid numerical issues
            return stable_log((u + (1 - u) * math.exp(-constraint.strength)) /
                            ((1 - u) + u * math.exp(-constraint.strength)))
        
        return 0.0
    
    def _update_source_parameters(self) -> None:
        """Update source reliability parameters using expected counts."""
        for s, source in enumerate(self.sources):
            tp = fp = tn = fn = 0.0
            
            # Count expected true/false positives/negatives
            for (src_idx, prop_idx), y in self.observations.items():
                if src_idx != s:
                    continue
                    
                b = max(min(self.propositions[prop_idx].belief, 1 - EPSILON), EPSILON)
                if y == 1:
                    tp += b      # True positive
                    fp += (1 - b)  # False positive
                else:
                    fn += b      # False negative
                    tn += (1 - b)  # True negative
            
            # Update with Beta posterior mean (clipped for stability)
            source.sensitivity = max(min(
                (source.alpha_a + tp - 1) / (source.alpha_a + source.beta_a + tp + fn - 2),
                1 - EPSILON
            ), EPSILON)
            
            source.specificity = max(min(
                (source.alpha_b + tn - 1) / (source.alpha_b + source.beta_b + tn + fp - 2),
                1 - EPSILON
            ), EPSILON)
    
    def belief_propagation_step(self, damping: float = 0.5) -> float:
        """Perform one step of belief propagation."""
        max_delta = 0.0
        
        for i, prop in enumerate(self.propositions):
            # Prior term
            prior_llr = stable_log(prop.prior / (1 - prop.prior))
            
            # Observation terms
            obs_llr = sum(
                self._compute_observation_llr(s, i)
                for s in range(len(self.sources))
                if (s, i) in self.observations
            )
            
            # Constraint terms
            constraint_llr = sum(
                self._compute_constraint_message(constraint, i)
                for constraint in self.constraints
                if i in constraint.prop_indices
            )
            
            # Total LLR and new belief
            total_llr = prior_llr + obs_llr + constraint_llr
            new_belief = stable_sigmoid(total_llr)
            
            # Apply damping
            old_belief = prop.belief
            prop.belief = damping * new_belief + (1 - damping) * old_belief
            
            # Track maximum change
            max_delta = max(max_delta, abs(prop.belief - old_belief))
        
        return max_delta
    
    def run_inference(self, max_iter: int = 100, tol: float = 1e-4, 
                     max_em_iter: int = 10, em_tol: float = 1e-3,
                     damping: float = 0.5) -> None:
        """Run variational EM to infer beliefs and source parameters.
        
        Args:
            max_iter: Maximum BP iterations per E-step
            tol: Convergence tolerance for BP (belief change)
            max_em_iter: Maximum EM iterations
            em_tol: Convergence tolerance for EM (parameter change)
            damping: Damping factor (0.0-1.0) for belief updates. 
                   Lower values make updates more stable but slower.
                   
        The algorithm alternates between:
        1. E-step: Update beliefs using current source parameters
        2. M-step: Update source parameters using current beliefs
        
        Convergence is reached when either:
        - Beliefs change by less than `tol` (BP convergence), or
        - Parameters change by less than `em_tol` (EM convergence)
        """
        for em_step in range(max_em_iter):
            # E-step: Run BP to convergence
            for bp_step in range(max_iter):
                max_delta = self.belief_propagation_step(damping)
                if max_delta < tol:
                    break
            
            # M-step: Update source parameters
            old_params = [(s.sensitivity, s.specificity) for s in self.sources]
            self._update_source_parameters()
            
            # Check for EM convergence
            param_diff = max(
                abs(s.sensitivity - old_s) + abs(s.specificity - old_p)
                for (old_s, old_p), s in zip(old_params, self.sources)
            )
            if param_diff < em_tol:
                break
    
    def get_contradiction_scores(self) -> List[float]:
        """Compute contradiction scores for each constraint in [0,1].
        
        Returns:
            List of scores, one per constraint, where:
            - 0: Constraint is perfectly satisfied
            - 1: Constraint is maximally violated
            - Values in between indicate partial constraint satisfaction
            
        The scores are computed as 1 - exp(-strength * violation_probability),
        which maps the raw violation probability through a saturating function
        to produce scores in [0,1). This makes the scores more interpretable
        and less sensitive to the absolute scale of the constraint strengths.
        """
        scores = []
        
        for constraint in self.constraints:
            if constraint.constraint_type == 'exclusion':
                # P(p ∧ q) for exclusion
                p, q = constraint.prop_indices
                violation_prob = (self.propositions[p].belief * 
                                self.propositions[q].belief)
                score = 1 - math.exp(-constraint.strength * violation_prob)
                
            elif constraint.constraint_type == 'entailment':
                # P(p ∧ ¬q) for p ⇒ q
                p, q = constraint.prop_indices
                violation_prob = (self.propositions[p].belief * 
                                (1 - self.propositions[q].belief))
                score = 1 - math.exp(-constraint.strength * violation_prob)
                
            elif constraint.constraint_type == 'equivalence':
                # P(p ≠ q) for p ⇔ q
                p, q = constraint.prop_indices
                violation_prob = (self.propositions[p].belief * 
                                (1 - self.propositions[q].belief) +
                                (1 - self.propositions[p].belief) * 
                                self.propositions[q].belief)
                score = 1 - math.exp(-constraint.strength * violation_prob)
            
            scores.append(score)
        
        return scores
