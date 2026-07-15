import numpy as np

# Emission-line colours. These are the real reasons nebulae have colour:
#   H-alpha  656nm  ionised hydrogen recombining -> crimson (the dominant one)
#   S-II     672nm                               -> orange-red
#   O-III    501nm  a *forbidden* transition     -> teal
#   dust                                          -> blue, by scattering
# Ordered as an ionisation ramp: the harder the radiation, the further right.
ANCHOR_T = np.array([0.0, 0.34, 0.68, 1.0])
ANCHOR_C = np.array([
    (1.00, 0.14, 0.28),   # H-alpha
    (0.98, 0.40, 0.12),   # S-II
    (0.16, 0.96, 0.82),   # O-III
    (0.36, 0.52, 1.00),   # reflection dust
], dtype=np.float32)


def fractal_field(size, beta, rng):
    """Fractal noise built by shaping white noise in Fourier space (numpy only).

    beta controls the spectral slope: bigger = smoother, larger features.
    """
    w = rng.normal(size=(size, size))
    kx = np.fft.fftfreq(size)[:, None]
    ky = np.fft.fftfreq(size)[None, :]
    k = np.hypot(kx, ky)
    k[0, 0] = 1.0
    spec = np.fft.fft2(w) * k ** (-beta)
    spec[0, 0] = 0.0
    field = np.real(np.fft.ifft2(spec))
    return field / field.std()


class Nebula:
    """A gas disk that shares the galaxy's potential but adds turbulent swirl.

    Renders additively: overlapping translucent particles *accumulate*, which is
    what turns discrete points into something that reads as glowing gas. That
    also means brightness scales with particle count -- drop n_gas and you must
    raise alpha to compensate, or the nebula just gets dimmer.
    """

    def __init__(self, state, n_gas=60000, num_arms=2, pitch_deg=26.0,
                 arm_spread=0.16, n_clumps=600, r_in=0.5, r_out=8.5,
                 grid=512, extent=12.0, turbulence=0.35, seed=11):
        self.state = state
        self.n_gas = n_gas
        self.num_arms = num_arms
        self.pitch = np.radians(pitch_deg)
        self.extent = extent
        self.grid = grid
        self.turbulence = turbulence
        rng = np.random.default_rng(seed)

        # --- the swirl field ---
        # Curl of a scalar potential is divergence-free by construction, so the
        # flow circulates instead of piling gas up in sinks. That is what makes
        # it look like fluid rather than like particles blown around.
        psi = fractal_field(grid, 2.4, rng)
        dpsi_dy, dpsi_dx = np.gradient(psi, 2 * extent / grid)
        self.fx, self.fy = dpsi_dy, -dpsi_dx
        spec_field = fractal_field(grid, 2.0, rng)

        # --- clumps strung along the arms ---
        # gamma(2) peaks away from r=0, unlike the stars' exponential: it avoids
        # a central pile-up that additive blending would blow out to white.
        cr = np.clip(rng.gamma(2.0, 2.0, n_clumps), r_in, r_out)
        carm = rng.integers(0, num_arms, n_clumps)
        cth = self._theta(cr, carm) + rng.normal(0, arm_spread, n_clumps)
        ccen = np.column_stack([cr * np.cos(cth), cr * np.sin(cth),
                                rng.normal(0, 0.06, n_clumps)])

        n_cl = int(n_gas * 0.74)
        owner = rng.integers(0, n_clumps, n_cl)
        spread = 0.18 + 0.45 * rng.random(n_clumps)
        clumped = ccen[owner] + rng.normal(0, 1, (n_cl, 3)) * spread[owner][:, None] * [1, 1, 0.22]

        n_df = n_gas - n_cl
        dr = np.clip(rng.gamma(2.4, 2.2, n_df), r_in, r_out * 1.1)
        dth = self._theta(dr, rng.integers(0, num_arms, n_df)) + rng.normal(0, 0.6, n_df)
        diffuse = np.column_stack([dr * np.cos(dth), dr * np.sin(dth),
                                   rng.normal(0, 0.14, n_df)])

        self.pos = np.vstack([clumped, diffuse]).astype(np.float32)
        is_clump = np.zeros(n_gas, dtype=bool)
        is_clump[:n_cl] = True

        # Warp along the curl field at three scales: turns round blobs into
        # stretched filaments before the sim even starts.
        for amp in (0.75, 0.35, 0.15):
            self.pos[:, 0] += amp * self._sample(self.fx, self.pos)
            self.pos[:, 1] += amp * self._sample(self.fy, self.pos)

        # --- colour ---
        # Blend continuously along the ramp rather than snapping to a species:
        # hard thresholds paint the galaxy into flat colour blocks. The jitter
        # dithers the boundaries so regions fade into each other.
        t = self._sample(spec_field, self.pos, freq=1.7) * 0.55 + rng.normal(0, 0.30, n_gas)
        t = 0.5 + 0.5 * np.tanh(t)
        self.color = np.empty((n_gas, 4), dtype=np.float32)
        for c in range(3):
            self.color[:, c] = np.interp(t, ANCHOR_T, ANCHOR_C[:, c])
        self.color[:, :3] *= (0.75 + 0.5 * rng.random((n_gas, 1)))

        boost = (220000.0 / n_gas) ** 0.72        # keep total glow ~constant vs n_gas
        rad = np.linalg.norm(self.pos[:, :2], axis=1)
        self.color[:, 3] = (np.where(is_clump, 0.060, 0.022) * boost
                            * np.clip(rad / 2.0, 0.3, 1.0))
        self.color = np.clip(self.color, 0.0, 1.0)
        self.size = (np.where(is_clump, 3 + 11 * rng.random(n_gas) ** 2,
                              2 + 4 * rng.random(n_gas)).astype(np.float32)
                     * boost ** 0.34 * 0.009)

        # Circular orbits in the galaxy's own potential, same as the stars.
        r_xy = np.maximum(np.hypot(self.pos[:, 0], self.pos[:, 1]), 1e-6)
        tangent = np.column_stack([-self.pos[:, 1] / r_xy, self.pos[:, 0] / r_xy,
                                   np.zeros(n_gas)])
        speed = state.circular_speed(np.linalg.norm(self.pos, axis=1))
        self.vel = (tangent * speed[:, None]).astype(np.float32)

    def _theta(self, r, arm):
        """Logarithmic spiral: theta goes as ln(r), which is what gives real
        galaxies a constant pitch angle. theta proportional to r would be an
        Archimedean spiral and winds into bullseye rings."""
        return arm * 2 * np.pi / self.num_arms + np.log(r) / np.tan(self.pitch)

    def _sample(self, field, pos, freq=1.0):
        g, e = self.grid, self.extent
        i = np.clip(((pos[:, 0] * freq + e) / (2 * e) * g).astype(int) % g, 0, g - 1)
        j = np.clip(((pos[:, 1] * freq + e) / (2 * e) * g).astype(int) % g, 0, g - 1)
        return field[j, i]

    def step(self, dt):
        """Orbit under gravity, then advect along the frozen curl field.

        Two motions compose here: differential rotation (inner gas laps outer
        gas, shearing clumps into trailing arcs) and the curl swirl (local
        eddies). The winding that ruins material spiral arms is exactly what
        makes gas look right.
        """
        acc = self.state._acceleration(self.pos)
        self.vel += acc * dt
        swirl = np.empty_like(self.pos)
        swirl[:, 0] = self._sample(self.fx, self.pos)
        swirl[:, 1] = self._sample(self.fy, self.pos)
        swirl[:, 2] = 0.0
        self.pos += (self.vel + self.turbulence * swirl) * dt
