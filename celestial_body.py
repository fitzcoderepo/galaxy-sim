import numpy as np


class CelestialBody:
    """A single body with identity — the black hole, or anything else massive
    enough to be worth tracking individually.

    Ordinary stars are NOT CelestialBody instances. There are 10,000 of them and
    they are interchangeable, so GalaxyState keeps them as (N, 3) arrays instead.
    """

    def __init__(self, position, velocity, mass: float,
                 is_black_hole: bool = False,
                 event_horizon_radius: float = 0.0):
        # np.array (not asarray) so we copy: asarray would alias a float32 array
        # passed in by the caller, and update_position would mutate it in place.
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.mass = float(mass)

        self.is_black_hole = is_black_hole
        # Radius at which this body swallows stars, in sim units. Not derived
        # from 2GM/c^2: with G=1 there is no metres-per-second to put in c, so
        # the horizon is a chosen parameter sized to look right on screen.
        self.event_horizon_radius = event_horizon_radius
        self.consumed = False

    def update_position(self, dt: float):
        """Advance this body along its current velocity."""
        if not self.consumed:
            self.position += self.velocity * dt
