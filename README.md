# marimo × ShaderToy

A proof of concept for **GPU scientific visualization in [marimo](https://marimo.io)**: a
[ShaderToy](https://www.shadertoy.com)-compatible GLSL renderer, packaged as an
[anywidget](https://anywidget.dev), whose inputs and buffers can be **NumPy arrays**.

**Live demo:** https://dnicotra.github.io/marimo-shadertoy/ (app) ·
[/notebook/](https://dnicotra.github.io/marimo-shadertoy/notebook/) (editable notebook).
Both run fully in the browser: Python via Pyodide, shaders via WebGL2.

## What it does

* **ShaderToy semantics.** `mainImage(out vec4, in vec2)`, `iTime`, `iTimeDelta`, `iFrame`,
  `iMouse`, `iResolution`, `iDate`, `iChannel0..N`, `iChannelResolution`. There is an Image
  pass plus any number of buffer passes, rendered in order with ping-pong feedback (a buffer
  reading itself sees its previous frame). Most ShaderToy shaders paste in unchanged.
* **NumPy → textures.** Any array becomes an `iChannel`:
  `(H, W)` / `(H, W, C≤4)` → `sampler2D`, `(D, H, W)` / `(D, H, W, C≤4)` → `sampler3D`.
  `float*` is uploaded as 32-bit float, `uint8` as normalised bytes. `arr[i, j]` is texel
  `(x=j, y=i)`, so `texelFetch` is index-exact.
* **NumPy → buffers.** A `Buffer(code, init=array)` starts from your array, then evolves on
  the GPU. Use it for simulations, iterative solvers and cellular automata. It can have a
  fixed `size` and run several `iterations` (sub-steps) per frame.
* **GPU → NumPy.** `widget.request_snapshot("A")` (or the camera button) reads a buffer
  back with `readPixels`. It lands in Python as `widget.snapshot`, an `(H, W, 4)` float32
  array, and re-runs dependent cells.
* **Reactive uniforms.** `widget.uniforms = {"uLevel": slider.value}` updates values without
  recompiling. Python types map to GLSL: `float`, `int`, `bool`, `vecN`, `matN`, `float[N]`.
* **Batteries.** `#include <colormaps>` (viridis, magma, inferno, plasma, cividis, turbo,
  coolwarm). Shader errors appear with correct line numbers, both on the canvas and in
  Python as `widget.errors`. Play, pause, step, reset and fullscreen controls. Rendering
  pauses while the widget is off-screen.

The demo notebook walks through four examples:

1. **Playground.** Classic ShaderToy presets, including a feedback buffer and a NumPy noise
   texture, with live GLSL editing.
2. **2-D field inspection.** A NumPy spectral-synthesis elevation map with a colormap,
   hill-shading and anti-aliased contours. Click to highlight an iso-line. Data controls
   re-run NumPy; view controls only change uniforms.
3. **NumPy → GPU simulation → NumPy.** Gray–Scott reaction–diffusion seeded from a NumPy
   array. Snapshots go back to NumPy for an FFT and the pattern wavelength, and the
   spectrum is drawn by a shader again.
4. **Volume rendering.** A hydrogen orbital on a 96³ grid as a `sampler3D`, ray-marched as
   an emission–absorption volume or as an isosurface.

## Usage

`shadertoy.py` is both the marimo app and the library: the widget classes are top-level
marimo definitions, so they can be imported.

```python
import marimo as mo
import numpy as np
from shadertoy import ShaderToy, Buffer, Channel

field = np.random.default_rng(0).random((256, 256)).astype(np.float32)

w = mo.ui.anywidget(ShaderToy(
    """
    #include <colormaps>
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        float v = texture(iChannel0, fragCoord / iResolution.xy).r;
        fragColor = vec4(viridis(v * uGain), 1.0);
    }
    """,
    channels=[Channel(field, filter="nearest")],
    uniforms={"uGain": 1.0},
))
w
```

```python
w.uniforms = {"uGain": gain_slider.value}   # cheap, no recompile
w.channels = [new_field]                    # re-uploads the texture
w.buffers = {"A": Buffer(sim_code, channels=["A"], init=state0, iterations=8)}
w.request_snapshot("A"); w.snapshot         # (H, W, 4) float32, after the browser replies
```

Run it locally:

```bash
uvx marimo edit --sandbox shadertoy.py      # or: uv run marimo edit shadertoy.py
uv run --group dev pytest                   # tests
```

## How it works

```
Python (marimo kernel / Pyodide)            Browser (anywidget view, WebGL2)
-------------------------------             --------------------------------
ShaderToy traits ── _passes, _common ─────▶ compile per pass (generated header, #line mapping)
                 ── _textures (bytes) ────▶ texImage2D / texImage3D (R..RGBA, 32F / 8)
                 ── _uniforms ────────────▶ uniform*() via program reflection
                 ── _snapshot_request ────▶ readPixels(buffer) ──┐
   .snapshot ◀── _snapshot_data (bytes) ◀────────────────────────┘
   .errors   ◀── errors ◀──────────────────── compile log, mapped to user lines
```

Arrays travel as binary buffers through the widget comm, not as JSON. Buffers are
RGBA32F render targets (`EXT_color_buffer_float`), with an RGBA16F or RGBA8 fallback.
Everything else is plain WebGL2.

## Deployment

`.github/workflows/pages.yml` tests the notebook, then exports it twice with
`marimo export html-wasm`: once as an app (`/`) and once as an editable notebook
(`/notebook/`). On the default branch it publishes the result to GitHub Pages. One-time
setup: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Limitations and next steps

This is a proof of concept. Natural next steps toward a scientific visualization tool:

* Send texture updates as deltas. Today, changing one array re-sends every array of that
  widget, which is fine for MBs but not for streaming.
* Add colorbars, axes and value probes (readback of the value under the cursor). Add
  keyboard input (ShaderToy's keyboard texture), cubemaps and audio channels.
* Support integer and half-float textures, `usampler`, and buffers with custom formats or
  multiple render targets.
* Factor the widget out of the notebook into a pip-installable package once the API settles.
