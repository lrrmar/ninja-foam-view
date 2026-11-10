#!/usr/bin/env python3
"""Interactive viewer for a WindNinja momentum-solver OpenFOAM case, with a
slider for height ABOVE MEAN SEA LEVEL (a flat horizontal slice).

Same look as view_foam_agl.py, but the sampling surface is a flat plane. Where the
plane is below the terrain there is no CFD data, so those areas are grey (the plane
"intersects" the terrain). Where it is above the top of the mesh it is also grey.

Needs view_foam_agl.py in the same folder (it reuses its loading and sampling code).

Usage:
    python view_foam_amsl.py NINJAFOAM_xxx/case.foam
    python view_foam_amsl.py NINJAFOAM_xxx/case.foam --gif amsl.gif     # no display needed
    python view_foam_amsl.py NINJAFOAM_xxx/case.foam --z-offset -996    # if mesh z is not metres above sea level

Needs:  pip install pyvista numpy scipy
"""
import argparse

import numpy as np
import pyvista as pv

from view_foam_agl import AGLSampler, load_case, prepare


class AMSLSampler(AGLSampler):
    """Samples the CFD solution on a flat horizontal plane at a height above sea level."""

    def __init__(self, pt, terrain, n, z_offset=0.0):
        super().__init__(pt, terrain, n)
        self.z_offset = z_offset  # height above sea level = mesh z + z_offset

    def at(self, h):
        z_mesh = h - self.z_offset
        grid = pv.StructuredGrid(self.X, self.Y, np.full_like(self.X, z_mesh))
        s = grid.sample(self.pt)
        valid = s.point_data["vtkValidPointMask"].astype(bool)
        sp = np.asarray(s.point_data["speed"], dtype=float).copy()
        sp[~valid] = np.nan  # inside the terrain, or above the mesh
        s.point_data["speed"] = sp
        return s, valid


def arrows_flat(sampler, h, length):
    """Horizontal arrows on the plane, raised slightly so they sit above it."""
    s, valid = sampler.at(h)
    if not valid.any():
        return None
    sub = s.extract_points(np.where(valid)[0])
    uh = np.asarray(sub.point_data["U"]).copy()
    uh[:, 2] = 0.0
    sub.point_data["Uh"] = uh
    arrows = sub.glyph(orient="Uh", scale=False, factor=length, tolerance=0.0)
    pts = np.asarray(arrows.points).copy()
    pts[:, 2] += 0.12 * length  # just over the arrow's half-width, so it sits on the plane
    arrows.points = pts
    return arrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_file", help="path to case.foam inside the case folder")
    ap.add_argument("--time", type=float, help="time directory to read (default: latest)")
    ap.add_argument("--decomposed", action="store_true")
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="added to the mesh z to give height above sea level (default 0)")
    ap.add_argument("--min", type=float, help="bottom of the slider range, m above sea level")
    ap.add_argument("--max", type=float, help="top of the slider range, m above sea level")
    ap.add_argument("--start", type=float, help="initial height, m above sea level")
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

    tz = np.asarray(terrain.points)[:, 2]
    zlo, zhi = float(tz.min()) + args.z_offset, float(tz.max()) + args.z_offset
    ztop = float(pt.bounds[5]) + args.z_offset
    print(f"terrain patch heights: {zlo:.0f} to {zhi:.0f} m (mesh top {ztop:.0f} m) with z-offset {args.z_offset:g}")
    print("Check these against your DEM's elevation range; if they differ by a constant, pass --z-offset.")

    lo = args.min if args.min is not None else zlo
    hi = args.max if args.max is not None else min(zhi + 500.0, ztop)
    start = args.start if args.start is not None else lo + 0.25 * (hi - lo)

    field = AMSLSampler(pt, terrain, args.n, args.z_offset)
    arrow_sampler = AMSLSampler(pt, terrain, args.arrows, args.z_offset) if args.arrows > 1 else None
    arrow_len = 0.8 * arrow_sampler.dx if arrow_sampler else 0.0
    clim = [0.0, float(np.nanmax(pt.point_data["speed"]))]  # fixed, so colours compare between heights

    pl = pv.Plotter(off_screen=bool(args.gif), window_size=(1400, 1000))
    surface, _ = field.at(start)
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
            a = arrows_flat(arrow_sampler, h, arrow_len)
            if a is not None and a.n_points:
                pl.add_mesh(a, color="white", name="arrows", reset_camera=False)
        mean = np.nanmean(s.point_data["speed"]) if valid.any() else float("nan")
        pl.add_text(f"{h:.0f} m above sea level\nmean {mean:.2f} m/s", name="info", font_size=11)
        pl.render()

    if args.gif:
        pl.open_gif(args.gif)
        for h in np.linspace(lo, hi, args.frames):
            draw(h)
            pl.write_frame()
        pl.close()
        print("saved", args.gif)
    else:
        draw(start)
        pl.add_slider_widget(draw, rng=[lo, hi], value=start,
                             title="Height above sea level (m)",
                             pointa=(0.2, 0.06), pointb=(0.8, 0.06), style="modern")
        pl.show()


if __name__ == "__main__":
    main()
