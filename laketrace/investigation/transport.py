"""Conservative cell-centred finite-volume transport in a closed rectangular box.

First-order upwind advection, centred diffusion, explicit time integration.
Face velocity is a discrete streamfunction curl: divergence-free with zero
boundary-normal flow. This controlled solver is not a calibrated lake model.
"""
import numpy as np


def flow_faces(shape, spacing, speed):
    nx, ny, nz = shape
    dx, dy, dz = spacing
    lx, ly = nx * dx, ny * dy
    x, y = np.arange(nx + 1) * dx, np.arange(ny + 1) * dy
    psi = speed * min(lx, ly) / np.pi * np.sin(np.pi*x[:, None]/lx) * np.sin(np.pi*y[None, :]/ly)
    u = np.repeat(((psi[:, 1:] - psi[:, :-1]) / dy)[..., None], nz, axis=2)
    v = np.repeat((-(psi[1:, :] - psi[:-1, :]) / dx)[..., None], nz, axis=2)
    w = np.zeros((nx, ny, nz + 1))
    u[[0, -1]] = 0.; v[:, [0, -1]] = 0.
    return u, v, w


def step(field, faces, spacing, diffusivity, dt):
    result = field.copy()
    for axis, (velocity, width, diffusion) in enumerate(zip(faces, spacing, diffusivity)):
        left = [slice(None)] * 3; right = left.copy(); inner = left.copy()
        left[axis] = slice(None, -1); right[axis] = slice(1, None); inner[axis] = slice(1, -1)
        left, right, inner = tuple(left), tuple(right), tuple(inner)
        v = velocity[inner]
        flux = v * np.where(v >= 0, field[left], field[right]) - diffusion*(field[right]-field[left])/width
        result[left] -= dt/width * flux
        result[right] += dt/width * flux
    return result


def simulate(initial, *, duration_s, dt=1., spacing=(10., 10., 4.), diffusivity=(.2, .15, .01), speed=.03):
    field = np.asarray(initial, dtype=float).copy()
    if field.ndim != 3 or not np.isfinite(field).all() or np.min(field) < 0:
        raise ValueError("initial concentrations must be finite and nonnegative")
    if duration_s < 0 or dt <= 0 or min(spacing) <= 0 or min(diffusivity) < 0:
        raise ValueError("invalid transport parameters")
    faces = flow_faces(field.shape, spacing, speed)
    bound = sum(np.max(np.abs(v))/h + 2*d/h**2 for v,h,d in zip(faces,spacing,diffusivity))
    if dt * bound > .9:
        raise ValueError("time step violates the declared positivity/CFL margin")
    remaining = float(duration_s)
    while remaining > 1e-9:
        advance = min(dt, remaining)
        field = step(field, faces, spacing, diffusivity, advance)
        remaining -= advance
    return field, faces
