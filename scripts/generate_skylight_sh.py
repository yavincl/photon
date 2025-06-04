import math
import sys

import numpy as np
from PIL import Image


def sh_coeff_order_2(direction):
    x, y, z = direction
    return np.array([
        0.2820947918,
        0.4886025119 * x,
        0.4886025119 * z,
        0.4886025119 * y,
        1.0925484310 * x * y,
        1.0925484310 * y * z,
        0.3153915653 * (3.0 * z * z - 1.0),
        0.7725484040 * x * z,
        0.3862742020 * (x * x - y * y),
    ], dtype=np.float32)


def project_sky(direction):
    projected_dir = direction[:2] / np.linalg.norm(direction[:2])
    azimuth_angle = math.pi + math.atan2(projected_dir[0], -projected_dir[1])
    altitude_angle = math.pi / 2.0 - math.acos(direction[1])

    coord_x = azimuth_angle * (1.0 / (2.0 * math.pi))
    coord_y = 0.5 + 0.5 * math.copysign(1.0, altitude_angle) * math.sqrt(
        2.0 / math.pi * abs(altitude_angle)
    )

    pad_amount = 2.0
    mul = 1.0 - 2.0 * pad_amount / 191.0
    add = pad_amount / 191.0
    coord_x = coord_x * mul + add
    coord_x *= 191.0 / 192.0

    return np.array([coord_x, coord_y])


def load_sky(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def sample_sky(tex, coord):
    h, w, _ = tex.shape
    x = int(coord[0] * w) % w
    y = max(min(int(coord[1] * h), h - 1), 0)
    return tex[y, x]


def compute_sh(tex, step_count=256):
    sh = np.zeros((9, 3), dtype=np.float64)
    for i in range(step_count):
        u = (i + 0.5) / step_count
        v = (i * 0.5 + 0.5) / step_count
        phi = 2.0 * math.pi * u
        cos_theta = 1.0 - 2.0 * v
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        direction = np.array([
            math.cos(phi) * sin_theta,
            cos_theta,
            math.sin(phi) * sin_theta,
        ])

        radiance = sample_sky(tex, project_sky(direction))
        coeff = sh_coeff_order_2(direction)
        for band in range(9):
            sh[band] += radiance * coeff[band]

    step_solid_angle = 2.0 * math.pi / step_count
    return (sh * step_solid_angle).astype(np.float32)


def main(path):
    tex = load_sky(path)
    sh = compute_sh(tex)
    for band in range(9):
        print("vec3({:.6f}, {:.6f}, {:.6f})".format(*sh[band]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_skylight_sh.py <sky_image>")
        sys.exit(1)
    main(sys.argv[1])
