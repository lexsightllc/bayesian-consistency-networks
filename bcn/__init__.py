"""
Bayesian Consistency Networks (BCN) - Improved Implementation

A refined implementation with:
- Corrected entailment directionality
- Stable EM updates for source parameters
- Numerically robust LLRs
- Normalized contradiction scores in [0,1]
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

__version__ = "1.2.0"

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
    neighbors: Set[int] = field(
        default_factory=set
    )  # Indices of connected propositions


@dataclass
class Constraint:
    """Represents a soft logical constraint between propositions.

    Attributes:
        constraint_type: Type of constraint. One of:
            - 'exclusion' (A ⊥ B): At most one proposition can be true
            - 'entailment' (A ⇒ B): If A is true, B must be true
            - 'equivalence' (A ⇔ B): A and B must have the same truth value
            - 'cardinality': At most 'cardinality' propositions can be true
        prop_indices: Indices of constrained propositions
        strength: Strength of the constraint (higher = stronger)
        cardinality: Maximum number of propositions that can be true (for 'cardinality' type)
    """

    constraint_type: str
    prop_indices: List[int]
    strength: float = 1.0
    cardinality: Optional[int] = None  # For 'cardinality' constraint type


class BayesianConsistencyNetwork:
    """
    Improved Bayesian Consistency Network for contradiction resolution.

    Key improvements:
    - Correct entailment directionality (p ⇒ q)
    - Stable EM updates for source parameters
    - Numerically robust LLRs with clipping
    - Contradiction scores normalized to [0,1]
    """

    def __init__(
        self,
        n_propositions: int,
        n_sources: int,
        *,
        correlation_penalty: bool = False,
        correlation_threshold: float = 0.8,
    ):
        """Initialize the BCN with given number of propositions and sources."""
        self.propositions = [
            Proposition(prior=0.5, belief=0.5) for _ in range(n_propositions)
        ]
        self.sources = [
            Source(sensitivity=0.8, specificity=0.8) for _ in range(n_sources)
        ]
        self.constraints: List[Constraint] = []
        self.observations: Dict[Tuple[int, int], int] = {}

        # Correlated-source down-weighting
        self.correlation_penalty = correlation_penalty
        self.correlation_threshold = correlation_threshold
        self.redundancy_weights = [1.0 for _ in range(n_sources)]

        # Metrics storage
        self._last_mean_delta: float = 0.0
        self._last_max_delta: float = 0.0
        self._last_mean_violation: Dict[str, float] = {}

    def add_observation(self, source_idx: int, prop_idx: int, value: int) -> None:
        """Record an observation from a source about a proposition."""
        if value not in {0, 1}:
            raise ValueError("Observation value must be 0 or 1")
        self.observations[(source_idx, prop_idx)] = value

    def add_constraint(
        self,
        constraint_type: str,
        prop_indices: List[int],
        strength: float = 1.0,
        cardinality: Optional[int] = None,
    ) -> None:
        """Add a soft logical constraint between propositions.

        Args:
            constraint_type: Type of constraint. One of:
                - 'exclusion' (A ⊥ B): At most one proposition can be true
                - 'entailment' (A ⇒ B): If A is true, B must be true
                - 'equivalence' (A ⇔ B): A and B must have the same truth value
                - 'cardinality': At most 'cardinality' propositions can be true
            prop_indices: List of proposition indices
            strength: Strength of the constraint (higher = stronger)
            cardinality: For 'cardinality' type, the maximum number of true propositions

        Raises:
            ValueError: If constraint_type is invalid or parameters are inconsistent
        """
        if constraint_type == "topk":
            constraint_type = "cardinality"
            # 'topk' is kept as an alias for 'cardinality' for backward compatibility

        if constraint_type == "cardinality":
            if cardinality is None or cardinality < 0:
                raise ValueError("Cardinality must be a non-negative integer")
            if len(prop_indices) < 2:
                raise ValueError(
                    "Cardinality constraints require at least 2 propositions"
                )
        elif constraint_type in {"exclusion", "entailment", "equivalence"}:
            if len(prop_indices) != 2:
                raise ValueError(
                    f"{constraint_type} constraint requires exactly 2 propositions"
                )
            if cardinality is not None:
                raise ValueError(
                    f"Cardinality parameter not supported for {constraint_type} constraint"
                )
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        constraint = Constraint(
            constraint_type=constraint_type,
            prop_indices=prop_indices,
            strength=strength,
            cardinality=cardinality if constraint_type == "cardinality" else None,
        )
        self.constraints.append(constraint)

        # Update neighbors for message passing
        if constraint_type == "cardinality":
            # For cardinality constraints, connect all pairs of variables
            from itertools import combinations

            for i, j in combinations(prop_indices, 2):
                self.propositions[i].neighbors.add(j)
                self.propositions[j].neighbors.add(i)
        else:
            # For binary constraints, just connect the two variables
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
            llr = stable_log(a / (1 - b))
        else:
            llr = stable_log((1 - a) / b)

        return self.redundancy_weights[source_idx] * llr

    def _compute_constraint_message(
        self, constraint: Constraint, target_idx: int
    ) -> float:
        """Compute the LLR message from a constraint to a target proposition.

        Args:
            constraint: The constraint to compute the message for
            target_idx: Index of the target proposition

        Returns:
            The log-likelihood ratio message from the constraint to the target
        """
        if constraint.constraint_type == "exclusion":
            other_idx = next(p for p in constraint.prop_indices if p != target_idx)
            u = self.propositions[other_idx].belief
            m1 = u * math.exp(-constraint.strength) + (1 - u)
            return stable_log(m1)

        if constraint.constraint_type == "equivalence":
            other_idx = next(p for p in constraint.prop_indices if p != target_idx)
            u = self.propositions[other_idx].belief
            m1 = u + (1 - u) * math.exp(-constraint.strength)
            m0 = (1 - u) + u * math.exp(-constraint.strength)
            return stable_log(m1) - stable_log(m0)

        if constraint.constraint_type == "entailment":
            p_idx, q_idx = constraint.prop_indices
            gamma = constraint.strength
            if target_idx == p_idx:
                u = self.propositions[q_idx].belief
                return stable_log(u + (1 - u) * math.exp(-gamma))
            else:
                u = self.propositions[p_idx].belief
                # Message from p to q when q is the target. The LLR captures
                # log P(q=1) - log P(q=0) under p ⇒ q, where the violation
                # probability is exp(-gamma) when p is true and q is false.
                # LLR = log((1-u)*1 + u*1) - log((1-u)*1 + u*exp(-gamma))
                #     = -log((1-u) + u*exp(-gamma))
                return -stable_log((1 - u) + u * math.exp(-gamma))

        if constraint.constraint_type == "cardinality":
            k = constraint.cardinality
            if k is None:
                return 0.0
            indices = constraint.prop_indices
            mu = sum(self.propositions[i].belief for i in indices if i != target_idx)
            v1 = max(0.0, mu + 1 - k)
            v0 = max(0.0, mu - k)
            return -constraint.strength * (v1 - v0)

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
                    tp += b  # True positive
                    fp += 1 - b  # False positive
                else:
                    fn += b  # False negative
                    tn += 1 - b  # True negative

            # Update with Beta posterior mean (clipped for stability)
            # Posterior mean: (alpha_prior + successes) / (alpha_prior + beta_prior + total)
            # For sensitivity: successes=tp, total=tp+fn
            source.sensitivity = max(
                min(
                    (source.alpha_a + tp)
                    / (source.alpha_a + source.beta_a + tp + fn),
                    1 - EPSILON,
                ),
                EPSILON,
            )

            # For specificity: successes=tn, total=tn+fp
            source.specificity = max(
                min(
                    (source.alpha_b + tn)
                    / (source.alpha_b + source.beta_b + tn + fp),
                    1 - EPSILON,
                ),
                EPSILON,
            )

    def _update_redundancy_weights(self) -> None:
        """Compute redundancy weights for correlated-source down-weighting."""
        if not self.correlation_penalty:
            self.redundancy_weights = [1.0 for _ in self.sources]
            return

        source_sets = []
        for s in range(len(self.sources)):
            props = {
                p
                for (src, p), val in self.observations.items()
                if src == s and val == 1
            }
            source_sets.append(props)

        weights = []
        for s, s_set in enumerate(source_sets):
            count = 0
            for t, t_set in enumerate(source_sets):
                if s == t:
                    continue
                union = s_set | t_set
                if not union:
                    continue
                jaccard = len(s_set & t_set) / len(union)
                if jaccard > self.correlation_threshold:
                    count += 1
            weights.append(1.0 / (1.0 + count))

        self.redundancy_weights = weights

    def _compute_mean_violations(self) -> Dict[str, float]:
        """Compute mean violation measure per constraint type."""
        viols: Dict[str, List[float]] = {}
        for constraint in self.constraints:
            if constraint.constraint_type == "exclusion":
                p, q = constraint.prop_indices
                v = self.propositions[p].belief * self.propositions[q].belief
            elif constraint.constraint_type == "entailment":
                p, q = constraint.prop_indices
                v = self.propositions[p].belief * (1 - self.propositions[q].belief)
            elif constraint.constraint_type == "equivalence":
                p, q = constraint.prop_indices
                v = (
                    self.propositions[p].belief * (1 - self.propositions[q].belief)
                    + (1 - self.propositions[p].belief) * self.propositions[q].belief
                )
            elif constraint.constraint_type == "cardinality":
                k = constraint.cardinality
                if k is None:
                    continue
                total = sum(
                    self.propositions[i].belief for i in constraint.prop_indices
                )
                v = max(0.0, total - k) / len(constraint.prop_indices)
            else:
                continue
            viols.setdefault(constraint.constraint_type, []).append(v)

        return {ctype: sum(vals) / len(vals) for ctype, vals in viols.items() if vals}

    def belief_propagation_step(self, damping: float = 0.5) -> float:
        """Perform one step of belief propagation."""
        max_delta = 0.0
        deltas: List[float] = []

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

            delta = abs(prop.belief - old_belief)
            deltas.append(delta)
            max_delta = max(max_delta, delta)

        if deltas:
            self._last_mean_delta = sum(deltas) / len(deltas)
            self._last_max_delta = max_delta
        else:
            self._last_mean_delta = self._last_max_delta = 0.0

        self._last_mean_violation = self._compute_mean_violations()

        return max_delta

    def run_inference(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        max_em_iter: int = 10,
        em_tol: float = 1e-3,
        damping: float = 0.5,
    ) -> None:
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
            if self.correlation_penalty:
                self._update_redundancy_weights()

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

        The scores are computed as 1 - exp(-strength * violation_measure),
        which maps the raw violation measure through a saturating function
        to produce scores in [0,1). This makes the scores more interpretable
        and less sensitive to the absolute scale of the constraint strengths.
        """
        scores = []

        for constraint in self.constraints:
            if constraint.constraint_type == "exclusion":
                # P(p ∧ q) for exclusion
                p, q = constraint.prop_indices
                violation_measure = (
                    self.propositions[p].belief * self.propositions[q].belief
                )

            elif constraint.constraint_type == "entailment":
                # P(p ∧ ¬q) for p ⇒ q
                p, q = constraint.prop_indices
                violation_measure = self.propositions[p].belief * (
                    1 - self.propositions[q].belief
                )

            elif constraint.constraint_type == "equivalence":
                # P(p ≠ q) for p ⇔ q
                p, q = constraint.prop_indices
                violation_measure = (
                    self.propositions[p].belief * (1 - self.propositions[q].belief)
                    + (1 - self.propositions[p].belief) * self.propositions[q].belief
                )

            elif constraint.constraint_type == "cardinality":
                # For cardinality constraint: measure how much the expected number
                # of true variables exceeds k, normalized by the number of variables
                k = constraint.cardinality
                if k is None:
                    continue
                total_belief = sum(
                    self.propositions[i].belief for i in constraint.prop_indices
                )
                excess = max(0, total_belief - k)
                violation_measure = excess / len(constraint.prop_indices)

            score = 1 - math.exp(-constraint.strength * violation_measure)
            scores.append(score)

        return scores

    def get_metrics(
        self, ground_truth: Optional[Iterable[int]] = None
    ) -> Dict[str, float]:
        """Return diagnostics from the last BP step.

        Args:
            ground_truth: Optional iterable of true proposition values for
                computing Brier score and log-loss.
        """

        metrics: Dict[str, float] = {
            "mean_delta": self._last_mean_delta,
            "max_delta": self._last_max_delta,
        }
        for ctype, val in self._last_mean_violation.items():
            metrics[f"mean_violation_{ctype}"] = val

        if ground_truth is not None:
            beliefs = [p.belief for p in self.propositions]
            gt_list = list(ground_truth)
            if len(gt_list) != len(beliefs):
                raise ValueError(
                    "Ground truth length must match number of propositions"
                )
            brier = sum((b - gt) ** 2 for b, gt in zip(beliefs, gt_list)) / len(gt_list)
            log_loss = -sum(
                gt * stable_log(b) + (1 - gt) * stable_log(1 - b)
                for b, gt in zip(beliefs, gt_list)
            ) / len(gt_list)
            metrics["brier"] = brier
            metrics["log_loss"] = log_loss

        return metrics
