#!/usr/bin/env python3
"""Interactive viewer for a WindNinja momentum-solver OpenFOAM case, with a
slider for height ABOVE GROUND LEVEL (not a flat horizontal slice).

For each height h, the wind is sampled on a surface that follows the terrain:
    z(x, y) = ground(x, y) + h
where ground(x, y) comes from the case's terrain patch (minZ).

Usage:
    cd NINJAFOAM_xxx && touch case.foam && cd ..
    python view_foam_agl.py NINJAFOAM_xxx/case.foam                # interactive window
    python view_foam_agl.py NINJAFOAM_xxx/case.foam --gif agl.gif  # no display needed
    python view_foam_agl.py NINJAFOAM_xxx/case.foam --decomposed   # if there are processor* folders

Needs:  pip install pyvista numpy scipy
(The interactive window needs a display with OpenGL. Over SSH use X forwarding,
 or use --gif and copy the file back.)
"""
import argparse

import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def load_case(path, time=None, decomposed=False):
    reader = pv.OpenFOAMReader(path)
    if decomposed:
        reader.reader.SetCaseType(0)  # VTK: 0 = decomposed, 1 = reconstructed
        reader.reader.UpdateInformation()
    reader.skip_zero_time = False
    reader.enable_all_cell_arrays()
    reader.enable_all_patch_arrays()  # includes the terrain patch (minZ)
    times = list(reader.time_values)
    t = time if time is not None else times[-1]
    reader.set_active_time_value(t)
    return reader.read(), t, times


def prepare(mesh, terrain_patch="minZ"):
    internal = mesh["internalMesh"]
    if internal.n_cells == 0:
        raise SystemExit("0 cells read. Try --decomposed (if there are processor* folders), "
                         "or check the time directories and that case.foam is in the case folder.")
    pt = internal.cell_data_to_point_data()
    pt.point_data["speed"] = np.linalg.norm(pt.point_data["U"], axis=1)
    try:
        terrain = mesh["boundary"][terrain_patch]
    except Exception:
        keys = list(mesh["boundary"].keys()) if "boundary" in mesh.keys() else "no boundary block"
        raise SystemExit(f"Terrain patch '{terrain_patch}' not found. Available: {keys}")
    return pt, terrain


class AGLSampler:
    """Samples the CFD solution on a terrain-following surface at height h above ground."""

    def __init__(self, pt, terrain, n):
        tp = np.asarray(terrain.points)
        self._lin = LinearNDInterpolator(tp[:, :2], tp[:, 2])
        self._near = NearestNDInterpolator(tp[:, :2], tp[:, 2])
        xmin, xmax, ymin, ymax = pt.bounds[:4]
        padx, pady = 0.002 * (xmax - xmin), 0.002 * (ymax - ymin)
        nx = int(n)
        ny = max(2, int(round(n * (ymax - ymin) / (xmax - xmin))))
        xs = np.linspace(xmin + padx, xmax - padx, nx)
        ys = np.linspace(ymin + pady, ymax - pady, ny)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.G = self.ground(self.X, self.Y)
        self.dx = xs[1] - xs[0]
        self.pt = pt

    def ground(self, x, y):
        """Terrain height at (x, y); nearest-point fallback outside the patch outline."""
        g = np.asarray(self._lin(x, y), dtype=float)
        bad = np.isnan(g)
        if bad.any():
            g[bad] = self._near(np.asarray(x)[bad], np.asarray(y)[bad])
        return g

    def at(self, h):
        grid = pv.StructuredGrid(self.X, self.Y, self.G + h)
        s = grid.sample(self.pt)
        valid = s.point_data["vtkValidPointMask"].astype(bool)
        sp = np.asarray(s.point_data["speed"], dtype=float).copy()
        sp[~valid] = np.nan  # points outside the mesh
        s.point_data["speed"] = sp
        return s, valid


def arrows_at(sampler, h, length):
    """Arrows lying along the terrain-following surface, lifted slightly above it.

    The surface is z = ground + h, so its slope equals the ground slope. Each arrow
    is tilted to run along that slope (instead of staying flat, which pushes one end
    into the terrain), and then raised by a small fraction of its length so the
    shaft and head clear the surface.
    """
    s, valid = sampler.at(h)
    if not valid.any():
        return None
    sub = s.extract_points(np.where(valid)[0])
    x, y = sub.points[:, 0], sub.points[:, 1]
    e = 0.5 * sampler.dx
    gx = (sampler.ground(x + e, y) - sampler.ground(x - e, y)) / (2 * e)
    gy = (sampler.ground(x, y + e) - sampler.ground(x, y - e)) / (2 * e)
    U = np.asarray(sub.point_data["U"])
    # horizontal flow direction, with the vertical component that keeps it in the surface plane
    tangent = np.c_[U[:, 0], U[:, 1], U[:, 0] * gx + U[:, 1] * gy]
    sub.point_data["Ut"] = tangent
    arrows = sub.glyph(orient="Ut", scale=False, factor=length, tolerance=0.0)
    pts = np.asarray(arrows.points).copy()
    pts[:, 2] += 0.2 * length  # clearance for the arrow's width on a tilted surface
    arrows.points = pts
    return arrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_file", help="path to case.foam inside the case folder")
    ap.add_argument("--time", type=float, help="time directory to read (default: latest)")
    ap.add_argument("--decomposed", action="store_true")
    ap.add_argument("--max-agl", type=float, default=500.0, help="top of the slider range (m)")
    ap.add_argument("--start", type=float, default=10.0, help="initial height above ground (m)")
    ap.add_argument("--n", type=int, default=120, help="sample points across x for the colour field")
    ap.add_argument("--arrows", type=int, default=25, help="arrows across x (0 for none)")
    ap.add_argument("--zscale", type=float, default=1.0, help="vertical exaggeration")
    ap.add_argument("--gif", help="write an animation stepping through heights instead of opening a window")
    ap.add_argument("--frames", type=int, default=40, help="number of frames for --gif")
    args = ap.parse_args()

    mesh, t, times = load_case(args.case_file, args.time, args.decomposed)
    print("time steps available:", times, "| reading:", t)
    pt, terrain = prepare(mesh)
    print(f"{pt.n_cells} cells | speed max {np.nanmax(pt.point_data['speed']):.2f} m/s")

    field = AGLSampler(pt, terrain, args.n)
    arrow_sampler = AGLSampler(pt, terrain, args.arrows) if args.arrows > 1 else None
    arrow_len = 0.8 * arrow_sampler.dx if arrow_sampler else 0.0
    clim = [0.0, float(np.nanmax(pt.point_data["speed"]))]  # fixed, so colours compare between heights

    pl = pv.Plotter(off_screen=bool(args.gif), window_size=(1400, 1000))
    surface, _ = field.at(args.start)
    pl.add_mesh(surface, scalars="speed", cmap="viridis", clim=clim, nan_color="lightgrey",
                scalar_bar_args={"title": "Wind speed (m/s)", "vertical": False,
                                 "position_x": 0.25, "position_y": 0.90,
                                 "width": 0.5, "height": 0.07})
    pl.set_scale(zscale=args.zscale)
    pl.view_xy()

    def draw(h):
        s, valid = field.at(h)
        surface.copy_from(s)
        surface.set_active_scalars("speed")
        if arrow_sampler is not None:
            a = arrows_at(arrow_sampler, h, arrow_len)
            if a is not None and a.n_points:
                pl.add_mesh(a, color="white", name="arrows", reset_camera=False)
        mean = np.nanmean(s.point_data["speed"]) if valid.any() else float("nan")
        pl.add_text(f"{h:.0f} m above ground\nmean {mean:.2f} m/s", name="info", font_size=11)
        pl.render()

    if args.gif:
        pl.open_gif(args.gif)
        for h in np.linspace(0.0, args.max_agl, args.frames):
            draw(h)
            pl.write_frame()
        pl.close()
        print("saved", args.gif)
    else:
        draw(args.start)
        pl.add_slider_widget(draw, rng=[0.0, args.max_agl], value=args.start,
                             title="Height above ground (m)",
                             pointa=(0.2, 0.06), pointb=(0.8, 0.06), style="modern")
        pl.show()


if __name__ == "__main__":
    main()
