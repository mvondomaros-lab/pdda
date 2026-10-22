import numpy as np
import pytest

from pdda.propagators import (
    PropagationResult,
    SmoluchowskiSolver,
    periodic_grid,
    propagate_density,
    propagate_diffusivity_profiles,
)


def test_solver_preserves_an_off_center_explicit_grid_exactly() -> None:
    grid = np.array([-3.7, -3.45, -3.2, -2.95])
    solver = SmoluchowskiSolver(grid, np.ones(4), np.zeros(4))

    result = solver.propagate(0.0, initial_position=-3.2)

    assert isinstance(result, PropagationResult)
    assert np.array_equal(solver.grid, grid)
    assert np.array_equal(result.positions, grid)
    assert result.densities.shape == (1, grid.size)
    assert np.array_equal(result.densities[0], [0.0, 0.0, 4.0, 0.0])


def test_scalar_and_multiple_times_are_consistent() -> None:
    grid = np.linspace(-1.0, 1.0, 21)
    solver = SmoluchowskiSolver(grid, np.ones_like(grid), 0.3 * grid**2)

    scalar = solver.propagate(0.4, initial_position=grid[6])
    multiple = solver.propagate([0.0, 0.4, 0.7], initial_position=grid[6])

    assert np.array_equal(scalar.times, [0.4])
    assert scalar.densities.shape == (1, grid.size)
    assert np.allclose(scalar.densities[0], multiple.densities[1])
    assert np.allclose(multiple.densities[0, 6], 10.0)


@pytest.mark.parametrize("periodic", [False, True])
def test_boundaries_conserve_mass_and_preserve_positivity(periodic: bool) -> None:
    grid = np.linspace(-1.5, 1.5, 31)
    solver = SmoluchowskiSolver(
        grid, 0.4 + grid**2, 0.7 * np.cos(2.0 * np.pi * grid / 3.0), periodic=periodic
    )
    result = solver.propagate([0.0, 0.02, 0.8], initial_position=grid[1])
    spacing = grid[1] - grid[0]

    assert np.all(result.densities >= 0.0)
    assert np.allclose(result.densities.sum(axis=1) * spacing, 1.0, atol=1e-12)


def test_relaxes_to_discrete_boltzmann_equilibrium() -> None:
    grid = np.linspace(-1.5, 1.5, 31)
    potential = 1.5 * np.cos(2.0 * np.pi * grid / 3.0)
    solver = SmoluchowskiSolver(grid, np.full_like(grid, 0.5), potential)

    result = solver.propagate(1_000.0, initial_position=grid[0])
    expected = np.exp(-potential)
    expected /= expected.sum() * (grid[1] - grid[0])

    assert np.allclose(solver.equilibrium, expected)
    assert np.allclose(result.equilibrium, expected)
    assert np.allclose(result.densities[0], expected, atol=1e-8)


@pytest.mark.parametrize("potential_step", [-1e-5, 1e-5])
def test_small_potential_threshold_preserves_equilibrium(potential_step: float) -> None:
    solver = SmoluchowskiSolver([0.0, 1.0], [1.0, 1.0], [0.0, potential_step])
    result = solver.propagate([0.1, 1.0], initial_density=solver.equilibrium)

    np.testing.assert_allclose(
        result.densities, np.broadcast_to(solver.equilibrium, (2, 2)), atol=1e-14
    )


@pytest.mark.parametrize(
    "grid",
    [
        [0.0],
        [0.0, 0.2, 0.1],
        [0.0, 0.1, 0.3],
        [0.0, np.nan],
    ],
)
def test_rejects_invalid_grids(grid: list[float]) -> None:
    with pytest.raises(ValueError, match="grid"):
        SmoluchowskiSolver(grid, [1.0] * len(grid), [0.0] * len(grid))


@pytest.mark.parametrize("diffusivity", [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0]])
def test_rejects_nonpositive_diffusivity(diffusivity: list[float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        SmoluchowskiSolver([0.0, 0.1, 0.2], diffusivity, [0.0, 0.0, 0.0])


@pytest.mark.parametrize("position", [0.15, np.inf, np.nan])
def test_rejects_invalid_initial_positions(position: float) -> None:
    solver = SmoluchowskiSolver([0.0, 0.1, 0.2], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="exact grid point"):
        solver.propagate(0.1, initial_position=position)


def test_rejects_invalid_times_and_profiles() -> None:
    with pytest.raises(ValueError, match="matching grid"):
        SmoluchowskiSolver([0.0, 0.1], [1.0], [0.0, 0.0])
    solver = SmoluchowskiSolver([0.0, 0.1], [1.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="times"):
        solver.propagate([-0.1])


def test_functional_density_propagation() -> None:
    positions = np.linspace(-1.0, 1.0, 5)
    result = propagate_density(
        positions,
        np.ones_like(positions),
        np.zeros_like(positions),
        [0.0, 0.1],
        initial_position=0.0,
    )

    assert isinstance(result, PropagationResult)
    assert np.array_equal(result.positions, positions)
    assert np.argmax(result.densities[0]) == 2


def test_periodic_grid_and_multi_profile_propagation() -> None:
    grid, spacing = periodic_grid(4.0, 0.3)
    assert grid.size % 2 == 1
    assert np.isclose(spacing * grid.size, 4.0)
    results = propagate_diffusivity_profiles(
        grid,
        np.zeros_like(grid),
        {"slow": np.ones_like(grid), "fast": np.full_like(grid, 2.0)},
        [0.0, 0.1],
    )
    assert tuple(results) == ("slow", "fast")
    assert np.allclose(results["slow"].densities.sum(axis=1) * spacing, 1.0)


def test_coarse_periodic_grid_can_be_used_for_propagation() -> None:
    grid, spacing = periodic_grid(4.0, 10.0)
    assert grid.size == 3
    assert grid[1] == 0.0
    assert spacing * grid.size == pytest.approx(4.0)
    result = propagate_density(grid, np.ones(3), np.zeros(3), 0.1, periodic=True)
    assert result.densities.sum() * spacing == pytest.approx(1.0)
