"""Quadratic quadrotor MLP: run directly for seeds 0--4; --plot-only redraws."""
from supervised_seed_test_common import main

if __name__ == "__main__":
    main("p4_mlp_quadrotor3d_TR_parallel", "quadratic", __file__)
