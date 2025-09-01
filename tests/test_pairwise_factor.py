import math

import pytest

from bcn import BayesianConsistencyNetwork, Constraint

US = [0.1 * i for i in range(1, 10)]
GAMMAS = [0.5, 1.0, 2.0]


def _make_network(u_other: float, target_idx: int) -> BayesianConsistencyNetwork:
    bcn = BayesianConsistencyNetwork(n_propositions=2, n_sources=0)
    # set beliefs
    bcn.propositions[1 - target_idx].belief = u_other
    bcn.propositions[target_idx].belief = 0.3
    return bcn


@pytest.mark.parametrize("u", US)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_exclusion_message(u: float, gamma: float):
    bcn = _make_network(u, 0)
    c = Constraint("exclusion", [0, 1], strength=gamma)
    msg = bcn._compute_constraint_message(c, 0)
    expected = math.log(u * math.exp(-gamma) + (1 - u))
    assert math.isclose(msg, expected, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("u", US)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_equivalence_message(u: float, gamma: float):
    bcn = _make_network(u, 0)
    c = Constraint("equivalence", [0, 1], strength=gamma)
    msg = bcn._compute_constraint_message(c, 0)
    m1 = u + (1 - u) * math.exp(-gamma)
    m0 = (1 - u) + u * math.exp(-gamma)
    expected = math.log(m1) - math.log(m0)
    assert math.isclose(msg, expected, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("u", US)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("target_idx", [0, 1])
def test_entailment_message(u: float, gamma: float, target_idx: int):
    bcn = _make_network(u, target_idx)
    c = Constraint("entailment", [0, 1], strength=gamma)
    msg = bcn._compute_constraint_message(c, target_idx)
    if target_idx == 0:
        expected = math.log(u + (1 - u) * math.exp(-gamma))
    else:
        expected = -math.log((1 - u) + u * math.exp(-gamma))
    assert math.isclose(msg, expected, rel_tol=1e-9, abs_tol=1e-12)
