import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.app import Timer

from celestial_body import CelestialBody
from galaxy_state import GalaxyState
from nebula import Nebula
from accretion_disk import AccretionDisk


# particles
N = 10000

# Gas particle count. This is the frame-rate dial -- see the note at the bottom.
N_GAS = 60000
N_ACCRETION = 12000    # accretion disk particles


# generate points on logarithmic spiral arms in a thin disk
NUM_ARMS = 2
PITCH_DEG = 26.0       # log-spiral pitch angle; gas and stars share this
ARM_SPREAD = 0.55
DISK_SCALE = 2.0
THICKNESS = 0.06

# --- black hole / halo (sim units: G = 1, halo flat rotation speed = 1) ---
M_BH = 0.35            # sphere of influence ~ G*M/V0^2 = 0.35, so the hole
                       # rules the inner disk and the halo rules the rest
EVENT_HORIZON = 0.05
SOFTENING = 0.05       # matched to the horizon: nothing survives closer anyway
HALO_SPEED = 1.0
HALO_CORE = 0.5
STAR_MASS = 1e-5       # only used for accretion bookkeeping
SIM_DT = 0.02          # sim time per frame; ~11 s of wall clock per orbit at r=2

# radius: exponention distribution => dense center, sparse edge
r = np.random.exponential(scale=DISK_SCALE, size=N)
r = np.clip(r, 0.05, None)                     # log(r) below needs r > 0

# Assign each star to an arm, then compute its spiral angle for that radius
arm = np.random.randint(0, NUM_ARMS, size=N)
theta = (arm * 2 * np.pi / NUM_ARMS)          # which arm this star belongs to
# Logarithmic spiral: theta ~ ln(r). The old `theta += r / WIND` was an
# Archimedean spiral, and wound ~6 full turns across the disk -> bullseye rings.
theta += np.log(r) / np.tan(np.radians(PITCH_DEG))
theta += np.random.normal(0, ARM_SPREAD, N)    # scatter so arms look natural

# Convert polar -> cartesian for the disk plane (x, y)
x = r * np.cos(theta)
y = r * np.sin(theta)


# z stays small => thin disk (thinner scatter than x/y)
z = np.random.normal(0, THICKNESS, N) * (1 + r)  # flares slightly at edges


pos = np.column_stack([x, y, z]).astype(np.float32)

# give each point a subtle color and make them brighter toward center
dist = np.linalg.norm(pos, axis=1)
brightness = np.clip(1.0 - dist / dist.max(), 0.15, 1.0)
colors = np.empty((N, 4), dtype=np.float32)
colors[:, 0] = brightness              # R
colors[:, 1] = brightness * 0.9        # G
colors[:, 2] = np.clip(brightness + 0.7, 0, 1)  # B (slightly bluish)
colors[:, 3] = 0.8                     # alpha


# --- build the simulation ---
black_hole = CelestialBody(
    # position=[1.0, 1.0, 1.0],
    position=[0.0, 0.0, 0.0],
    velocity=[0.0, 0.0, 0.0],
    mass=M_BH,
    is_black_hole=True,
    event_horizon_radius=EVENT_HORIZON,
)

state = GalaxyState(
    black_hole,
    halo_speed=HALO_SPEED,
    halo_core=HALO_CORE,
    softening=SOFTENING,
)

# Put every star on a circular orbit: speed from the potential itself, direction
# tangential in the disk plane. Seed them at rest instead and the whole disk
# free-falls into the centre in about one crossing time.
r_xy = np.maximum(np.hypot(x, y), 1e-6)
tangent = np.column_stack([-y / r_xy, x / r_xy, np.zeros(N)])
vel = tangent * state.circular_speed(dist)[:, None]

state.add_stars(pos, vel, STAR_MASS)

# gas disk: same potential, same arms, plus curl-noise swirl
nebula = Nebula(state, n_gas=N_GAS, num_arms=NUM_ARMS, pitch_deg=PITCH_DEG)

# Accretion disk around the black hole (scenery: rotated analytically).
# r_out sits inside the nebula's inner edge (0.5) so the gas doesn't drown it.
disk = AccretionDisk(state, r_s=EVENT_HORIZON, n=N_ACCRETION,
                     r_out=0.45, t_isco=9000.0)


# set up canvas
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', show=True, title='Galaxy Starter')

view = canvas.central_widget.add_view()


# gas first, so the stars draw over it.
# 'additive' is what makes this read as gas rather than as dots: overlapping
# particles sum instead of occluding, so density becomes brightness and
# crimson over teal becomes white-hot, exactly like a long-exposure photo.
gas = visuals.Markers(scaling='scene')
gas.set_data(nebula.pos, edge_width=0, face_color=nebula.color, size=nebula.size)
gas.set_gl_state('additive', blend=True, depth_test=False)
view.add(gas)


# draw the point cloud
scatter = visuals.Markers()
scatter.set_data(pos, edge_width=0, face_color=colors, size=3)
scatter.set_gl_state('translucent', blend=True, depth_test=False)
view.add(scatter)

# --- the black hole ---
# You can't see the hole; you see the shadow it casts and the gas it heats.
# The shadow is a Sphere so its silhouette stays circular from every camera
# angle -- a flat disc would foreshorten to an oval, which a real shadow doesn't.
shadow = visuals.Sphere(radius=disk.r_shadow, method='latitude', rows=32, cols=32,
                        color=(0, 0, 0, 1), parent=view.scene)
shadow.set_gl_state('opaque', depth_test=False)

# drawn after the shadow so the near side of the disk correctly crosses in front
accretion = visuals.Markers(scaling='scene')
# seed with un-beamed colour: the camera doesn't exist yet, and set_data must
# happen before the camera is assigned (assigning it walks every visual's bounds)
accretion.set_data(disk.pos, edge_width=0,
                   face_color=np.column_stack(
                       [disk.rgb, np.full(len(disk.rgb), disk.alpha)]).astype(np.float32),
                   size=disk.size)
accretion.set_gl_state('additive', blend=True, depth_test=False)
view.add(accretion)


# 3D camera for drag to orbit and scroll to zoom
view.camera = scene.TurntableCamera(fov=45, distance=8)


# step the physics and push the new positions to the GPU each frame
def on_timer(event):
    state.step(SIM_DT)
    nebula.step(SIM_DT)
    disk.step(SIM_DT)
    live = state.alive
    if live.any():
        scatter.set_data(state.star_pos[live], edge_width=0,
                         face_color=colors[live], size=3)
    gas.set_data(nebula.pos, edge_width=0, face_color=nebula.color,
                 size=nebula.size)
    # recomputed every frame: beaming depends on where you're looking from
    accretion.set_data(disk.pos, edge_width=0,
                       face_color=disk.colors(disk.view_axis(accretion)),
                       size=disk.size)




rotate = Timer(interval=0.016, connect=on_timer, start=True)



if __name__ == '__main__':
    canvas.app.run()
