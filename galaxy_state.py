import numpy as np

from celestial_body import CelestialBody


class GalaxyState:
    """Evolves a disk of test-particle stars around a central black hole.

    This is a test-particle model, not an N-body model: stars feel the black
    hole and the dark matter halo, but not each other. True N-body would be
    O(N^2) — 100 million pairs per frame at N=10,000 — and stars pull on each
    other far too weakly to matter here anyway.

    Units: G = 1, and the halo's flat rotation speed is 1. Mass and time follow
    from those two choices; nothing here is in SI.
    """

    G = 1.0

    def __init__(self, black_hole: CelestialBody,
                 halo_speed: float = 1.0,
                 halo_core: float = 0.5,
                 softening: float = 0.05,
                 substeps: int = 2):
        self.black_hole = black_hole

        # Halo: a logarithmic potential, which gives a flat rotation curve at
        # large radius. Without it the black hole alone gives Keplerian falloff
        # and the arms wind up almost immediately.
        self.halo_speed = halo_speed
        self.halo_core = halo_core

        # Plummer softening. Keeps 1/r^2 finite for stars that get close to the
        # centre; without it the first near-miss returns a near-infinite kick
        # and NaNs the whole array.
        self.softening = softening
        self.substeps = substeps

        self.star_pos = np.zeros((0, 3), dtype=np.float32)
        self.star_vel = np.zeros((0, 3), dtype=np.float32)
        self.alive = np.zeros(0, dtype=bool)
        self.star_mass = 0.0

        self.time_elapsed = 0.0
        self.consumed_count = 0

    def add_stars(self, pos, vel, mass_each: float):
        self.star_pos = np.ascontiguousarray(pos, dtype=np.float32)
        self.star_vel = np.ascontiguousarray(vel, dtype=np.float32)
        self.alive = np.ones(len(self.star_pos), dtype=bool)
        self.star_mass = mass_each

    def circular_speed(self, r):
        """Speed of a circular orbit at radius r, in the same softened potential
        the integrator uses.

        Deriving this from the same constants as _acceleration is the point: a
        star seeded with sqrt(GM/r) while the force law is softened starts on an
        orbit that doesn't match the forces acting on it, and the inner disk
        breathes in visible rings.
        """
        r2 = np.asarray(r, dtype=np.float64) ** 2
        v2 = self.G * self.black_hole.mass * r2 / (r2 + self.softening ** 2) ** 1.5
        v2 += self.halo_speed ** 2 * r2 / (r2 + self.halo_core ** 2)
        return np.sqrt(v2)

    def _acceleration(self, pos):
        """Acceleration on every star at once — one (N, 3) expression, no loop."""
        # Toward the black hole, softened: a = GM * d / (|d|^2 + eps^2)^1.5
        d = self.black_hole.position - pos
        r2 = np.einsum('ij,ij->i', d, d)
        acc = d * (self.G * self.black_hole.mass
                   / (r2 + self.softening ** 2) ** 1.5)[:, None]

        # Toward the galactic centre, from the halo.
        r2_gal = np.einsum('ij,ij->i', pos, pos)
        acc -= pos * (self.halo_speed ** 2
                      / (r2_gal + self.halo_core ** 2))[:, None]
        return acc

    def _accrete(self):
        """Swallow any live star inside the event horizon."""
        bh = self.black_hole
        if bh.event_horizon_radius <= 0.0:
            return 0

        d = self.star_pos - bh.position
        r2 = np.einsum('ij,ij->i', d, d)
        swallowed = self.alive & (r2 < bh.event_horizon_radius ** 2)

        n = int(np.count_nonzero(swallowed))
        if n:
            self.alive[swallowed] = False
            bh.mass += n * self.star_mass
            self.consumed_count += n
            # Park them on the black hole with zero velocity rather than
            # deleting rows: N stays fixed (so the colour array still lines up),
            # and softened gravity at d = 0 is exactly zero, so they sit there
            # costing a few flops. The render mask hides them.
            self.star_pos[swallowed] = bh.position
            self.star_vel[swallowed] = 0.0
        return n

    def step(self, dt: float):
        """Advance the galaxy by dt using kick-drift-kick leapfrog."""
        h = dt / self.substeps
        for _ in range(self.substeps):
            self.star_vel += 0.5 * h * self._acceleration(self.star_pos)
            self.star_pos += h * self.star_vel
            self.star_vel += 0.5 * h * self._acceleration(self.star_pos)
            self._accrete()
            self.time_elapsed += h
