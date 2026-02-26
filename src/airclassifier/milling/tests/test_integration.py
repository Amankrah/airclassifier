"""
Milling Module Integration Tests
================================

End-to-end tests for the hammer mill simulation.
"""

import pytest
import numpy as np


class TestMillConfig:
    """Tests for MillConfig and related config classes."""

    def test_mill_config_defaults(self):
        """Test MillConfig has reasonable defaults (NIH: 0.75 mm → D50 ~24 µm)."""
        from ..config import MillConfig

        config = MillConfig()
        assert config.rotor_rpm == 3000.0
        assert config.screen_aperture_mm == 0.75
        assert config.hammer_rows >= 1
        assert config.hammers_per_row >= 1

    def test_mill_config_derived_properties(self):
        """Test derived properties compute correctly."""
        from ..config import MillConfig

        config = MillConfig(rotor_rpm=3000)

        # Angular velocity
        omega = config.rotor_angular_velocity
        assert omega > 0
        assert abs(omega - 3000 * 2 * np.pi / 60) < 1e-6

        # Tip speed
        tip_speed = config.hammer_tip_speed
        assert tip_speed > 0

    def test_mill_recipe(self):
        """Test MillRecipe dataclass."""
        from ..config import MillRecipe

        recipe = MillRecipe(
            name="test",
            rotor_rpm=3500,
            screen_aperture_mm=2.0,
            feed_rate_kg_per_hr=400,
        )

        assert recipe.rotor_omega > 0
        assert recipe.throughput_kg_per_s() == 400 / 3600


class TestGeometry:
    """Tests for geometry generation."""

    def test_create_assembly(self):
        """Test creating a machine assembly."""
        from ..geometry import create_hammer_mill_machine

        assembly = create_hammer_mill_machine()
        assert assembly is not None
        assert assembly.config is not None

    def test_build_meshes(self):
        """Test mesh generation."""
        from ..geometry import build_hammer_mill_meshes

        meshes = build_hammer_mill_meshes()

        assert "rotor" in meshes
        assert "hammers" in meshes
        assert "screen" in meshes
        assert "housing" in meshes

        for name, (verts, tris, meta) in meshes.items():
            assert isinstance(verts, np.ndarray)
            assert isinstance(tris, np.ndarray)
            assert len(verts.shape) == 2
            assert verts.shape[1] == 3
            assert len(tris.shape) == 2
            assert tris.shape[1] == 3

    def test_animated_components(self):
        """Test animation metadata."""
        from ..geometry import create_hammer_mill_machine

        assembly = create_hammer_mill_machine()
        animated = assembly.get_animated_components()

        # Rotor and hammers should be animated
        assert "rotor" in animated
        assert "hammers" in animated
        assert animated["rotor"]["animation_type"] == "rotate"


class TestKernels:
    """Tests for Warp kernels (NumPy fallbacks)."""

    def test_transport_step(self):
        """Test transport kernel."""
        from ..kernels import transport_step_np

        n = 10
        positions = np.random.rand(n, 3).astype(np.float32) * 0.1
        velocities = np.random.rand(n, 3).astype(np.float32) * 0.5
        sizes = np.ones(n, dtype=np.float32) * 0.001
        masses = np.ones(n, dtype=np.float32) * 0.0001
        residence_times = np.zeros(n, dtype=np.float32)

        new_pos, new_vel, new_res = transport_step_np(
            positions, velocities, sizes, masses, residence_times,
            chamber_radius=0.22,
            chamber_length=0.40,
            rotor_omega=300.0,
            dt=0.001,
        )

        assert new_pos.shape == positions.shape
        assert new_vel.shape == velocities.shape
        assert np.all(new_res >= 0)

    def test_impact_detection(self):
        """Test impact detection kernel."""
        from ..kernels import impact_detection_np

        n = 10
        # Place particles in the impact zone (between 0.15m and 0.20m radius)
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:, 0] = np.linspace(0.08, 0.26, n)  # Along X at hammer rows
        positions[:, 1] = 0.17  # Y component gives radius ~0.17m (in impact zone)
        positions[:, 2] = 0.0
        velocities = np.random.rand(n, 3).astype(np.float32)
        sizes = np.ones(n, dtype=np.float32) * 0.001
        masses = np.ones(n, dtype=np.float32) * 0.0001

        impact_flags, impact_energies, new_vel = impact_detection_np(
            positions, velocities, sizes, masses,
            rotor_theta=0.0,
            rotor_omega=300.0,
            hammer_tip_radius=0.18,  # Corrected for new geometry
            hammer_width=0.05,
            hammer_rows=4,
            hammers_per_row=4,
            row_start_x=0.08,
            row_spacing=0.06,
        )

        assert len(impact_flags) == n
        assert len(impact_energies) == n
        assert new_vel.shape == velocities.shape

    def test_breakage_step(self):
        """Test breakage kernel."""
        from ..kernels import breakage_step_np

        n = 10
        sizes = np.ones(n, dtype=np.float32) * 0.002  # 2mm particles
        masses = np.ones(n, dtype=np.float32) * 0.001
        impact_flags = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        impact_energies = np.array([0.1, 0.05, 0.02, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        new_sizes, new_masses, break_flags = breakage_step_np(
            sizes, masses, impact_flags, impact_energies,
        )

        assert len(new_sizes) == n
        assert len(new_masses) == n
        # Impacted particles may have broken (smaller sizes)
        # Non-impacted should be unchanged
        for i in range(3, n):
            assert new_sizes[i] == sizes[i]

    def test_screen_passage(self):
        """Test screen passage kernel."""
        from ..kernels import screen_passage_np

        n = 10
        # Place some particles in screen zone (updated for corrected geometry)
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:, 0] = 0.15  # X in screen range
        positions[:, 1] = -0.188  # Y at screen radius (corrected)
        velocities = np.random.rand(n, 3).astype(np.float32)
        sizes = np.linspace(0.0005, 0.002, n).astype(np.float32)  # 0.5mm to 2mm
        masses = np.ones(n, dtype=np.float32) * 0.0001

        passage_flags = screen_passage_np(
            positions, velocities, sizes, masses,
            screen_radius=0.188,  # Corrected for new geometry
            screen_start_angle=np.pi,
            screen_arc_angle=np.pi,
            screen_x_start=0.05,
            screen_x_end=0.35,
            aperture=0.0015,  # 1.5mm aperture
        )

        assert len(passage_flags) == n
        # Small particles (< 1.5mm) should have chance to pass
        # Large particles (> 1.5mm) should not pass


class TestPhysics:
    """Tests for physics solvers."""

    def test_impact_solver(self):
        """Test ImpactSolver."""
        from ..physics import ImpactSolver
        from ..config import MillConfig

        solver = ImpactSolver.from_config(MillConfig())
        solver.set_rotor_state(0.0, 300.0)

        positions = np.random.rand(5, 3).astype(np.float32) * 0.2
        velocities = np.random.rand(5, 3).astype(np.float32)
        sizes = np.ones(5, dtype=np.float32) * 0.001
        masses = np.ones(5, dtype=np.float32) * 0.0001

        impact_flags, impact_energies, new_vel = solver.step(
            positions, velocities, sizes, masses, dt=0.001
        )

        assert len(impact_flags) == 5
        assert solver.stats is not None

    def test_breakage_model(self):
        """Test BreakageModel."""
        from ..physics import BreakageModel

        model = BreakageModel()

        sizes = np.ones(5, dtype=np.float32) * 0.002
        masses = np.ones(5, dtype=np.float32) * 0.001
        impact_flags = np.array([1, 1, 0, 0, 0], dtype=np.int32)
        impact_energies = np.array([0.1, 0.05, 0, 0, 0], dtype=np.float32)

        new_sizes, new_masses, break_flags = model.step_lagrangian(
            sizes, masses, impact_flags, impact_energies
        )

        assert len(new_sizes) == 5
        assert model.stats is not None

    def test_screen_classifier(self):
        """Test ScreenClassifier."""
        from ..physics import ScreenClassifier
        from ..config import MillConfig

        classifier = ScreenClassifier.from_config(MillConfig())

        n = 5
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:, 0] = 0.15
        positions[:, 1] = -0.21
        velocities = np.random.rand(n, 3).astype(np.float32)
        sizes = np.array([0.001, 0.001, 0.002, 0.002, 0.002], dtype=np.float32)
        masses = np.ones(n, dtype=np.float32) * 0.0001

        ret_pos, ret_vel, ret_sizes, ret_masses, passage_flags = classifier.step(
            positions, velocities, sizes, masses
        )

        # Some may pass, some retained
        assert len(ret_pos) <= n
        assert classifier.stats is not None


class TestSimulator:
    """Tests for the high-level simulator."""

    def test_create_simulator(self):
        """Test creating a simulator."""
        from ..simulator import HammerMillSimulator

        sim = HammerMillSimulator.create()
        assert sim is not None
        assert sim.assembly is not None
        assert sim.engine is not None

    def test_initialize_and_step(self):
        """Test initialization and stepping."""
        from ..simulator import HammerMillSimulator

        sim = HammerMillSimulator.create()
        sim.initialize(initial_holdup_kg=0.01)

        state = sim.step(dt=0.001)

        assert state.time_s > 0
        assert state.rotor_theta_rad >= 0

    def test_run_simulation(self):
        """Test running a simulation."""
        from ..simulator import run_milling_simulation
        from ..config import MillConfig, MillRecipe

        config = MillConfig()
        recipe = MillRecipe(
            feed_rate_kg_per_hr=100,
            rotor_rpm=3000,
        )

        result = run_milling_simulation(
            config=config,
            recipe=recipe,
            duration_s=0.1,  # Short run for testing
            dt=0.001,
            initial_holdup_kg=0.001,
        )

        assert result is not None
        assert len(result.history) > 0

    def test_outlet_state(self):
        """Test getting outlet conditions."""
        from ..simulator import HammerMillSimulator

        sim = HammerMillSimulator.create()
        sim.initialize(initial_holdup_kg=0.001)
        sim.run(duration_s=0.05, dt=0.001)

        outlet = sim.get_outlet_conditions()

        assert outlet is not None
        # Outlet state should have PSD info
        assert hasattr(outlet, "d50_um")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
