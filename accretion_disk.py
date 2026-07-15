import numpy as np

# Blackbody locus, roughly: deep red -> orange -> white -> blue-white.
# This is why the disk's colour gradient is not arbitrary -- it is temperature.
BB_T = np.array([1000, 2000, 3000, 4000, 5000, 6500, 8000, 12000, 20000], float)
BB_C = np.array([
    (1.00, 0.22, 0.00), (1.00, 0.38, 0.08), (1.00, 0.55, 0.26),
    (1.00, 0.68, 0.44), (1.00, 0.78, 0.62), (1.00, 0.90, 0.85),
    (0.92, 0.92, 1.00), (0.72, 0.80, 1.00), (0.62, 0.72, 1.00),
], dtype=np.float32)


class AccretionDisk:
    """The visible part of a black hole: hot gas spiralling in.

    Geometry follows real Schwarzschild relationships, all keyed off r_s:
      r_shadow = 2.6 * r_s   the silhouette. Bigger than the horizon because the
                             hole bends light from behind it into your eye, so
                             you see a shadow larger than the object casting it.
      r_isco   = 3.0 * r_s   innermost stable circular orbit. Inside this there
                             are no stable orbits at all, so the disk simply
                             ends -- that's the gap, not an artistic choice.

    The disk is rotated analytically rather than integrated: it is scenery, and
    letting the leapfrog touch it would just let the hole slowly eat it.
    """

    def __init__(self, state, r_s=0.05, n=20000, r_out=1.0, t_isco=12000.0,
                 c_sim=2.8, alpha=0.020, seed=5):
        self.state = state
        self.r_s = r_s
        self.r_shadow = 2.6 * r_s
        self.r_isco = 3.0 * r_s
        self.c_sim = c_sim          # "speed of light" in sim units -> v/c ~ 0.5 at ISCO
        self.alpha = alpha
        rng = np.random.default_rng(seed)

        # surface density rises inward: sample r with a 1/r power law
        u = rng.random(n)
        self.r = (self.r_isco**-1.0 + u * (r_out**-1.0 - self.r_isco**-1.0))**-1.0
        self.theta = rng.uniform(0, 2*np.pi, n)
        self.z = (rng.normal(0, 1, n) * 0.015 * self.r).astype(np.float32)

        speed = state.circular_speed(self.r)
        self.omega = speed / self.r          # angular rate for analytic rotation
        self.speed = speed

        # Shakura-Sunyaev thin disk: T ~ r^(-3/4). The real profile.
        T = t_isco * (self.r / self.r_isco)**-0.75
        self.rgb = np.empty((n, 3), dtype=np.float32)
        for c in range(3):
            self.rgb[:, c] = np.interp(T, BB_T, BB_C[:, c])
        self.t_range = (T.max(), T.min())

        self.size = (0.004 + 0.010*rng.random(n)).astype(np.float32)
        self.pos = np.empty((n, 3), dtype=np.float32)
        self._place()

    def _place(self):
        self.pos[:, 0] = self.r * np.cos(self.theta)
        self.pos[:, 1] = self.r * np.sin(self.theta)
        self.pos[:, 2] = self.z

    def step(self, dt):
        self.theta += self.omega * dt
        self._place()

    def view_axis(self, visual):
        """Unit vector pointing away from the camera, recovered from the depth
        gradient of the render transform. Avoids hard-coding vispy's
        azimuth/elevation convention, and follows the camera as you drag."""
        tr = visual.get_transform('visual', 'render')
        o = tr.map(np.zeros(3, np.float32)); oz = o[2]/o[3]
        g = []
        for k in range(3):
            e = np.zeros(3, np.float32); e[k] = 1.0
            p = tr.map(e)
            g.append(p[2]/p[3] - oz)
        g = np.array(g)
        return g / np.linalg.norm(g)

    def colors(self, n_axis):
        """Doppler-beamed colour, with anything behind the shadow culled."""
        # Relativistic beaming: gas coming at you is brighter and bluer. Flux
        # goes as D^3, so a disk orbiting at ~0.5c is ~17x brighter on the
        # approaching side. This asymmetry is why real black hole images
        # (M87*, Sgr A*) have one blazing limb and one dim one.
        tangent = np.column_stack([-np.sin(self.theta), np.cos(self.theta),
                                   np.zeros(len(self.r))])
        vel = tangent * self.speed[:, None]
        beta_los = -(vel @ n_axis) / self.c_sim
        doppler = 1.0 / np.clip(1.0 - beta_los, 0.05, None)
        boost = np.clip(doppler**3, 0.05, 8.0)

        col = np.empty((len(self.r), 4), dtype=np.float32)
        col[:, :3] = np.clip(self.rgb * boost[:, None]**0.35, 0, 1)
        col[:, 3] = np.clip(self.alpha * boost, 0, 1)

        # Occlusion by hand: the offscreen/main framebuffer here has no usable
        # depth buffer (depth_test=True renders nothing), so cull the particles
        # the shadow would hide -- behind the centre and within r_shadow of the
        # line of sight.
        depth = self.pos @ n_axis
        perp = np.linalg.norm(self.pos - depth[:, None]*n_axis, axis=1)
        col[(depth > 0) & (perp < self.r_shadow), 3] = 0.0
        return col
