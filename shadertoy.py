# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.25", "anywidget>=0.9", "numpy>=1.24"]
# ///
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="marimo × ShaderToy")

with app.setup:
    import hashlib
    import re
    from dataclasses import dataclass
    from typing import Any

    import anywidget
    import marimo as mo
    import numpy as np
    import traitlets


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # marimo × ShaderToy

    A proof of concept for **GPU scientific visualization in marimo**: a
    [ShaderToy](https://www.shadertoy.com)-compatible GLSL renderer that runs
    in your browser, wired to Python so that **NumPy arrays become GPU
    textures**, **simulation buffers are seeded from NumPy**, and
    **GPU state can be read back into NumPy** for analysis.

    | ShaderToy | here |
    |:---|:---|
    | `mainImage(out vec4, in vec2)`, `iTime`, `iFrame`, `iMouse`, `iResolution`, … | same |
    | Image + Buffer A–D passes, ping-pong feedback | same (any number of named buffers, optional fixed size and sub-steps per frame) |
    | `iChannel0..3` = images / buffers | NumPy arrays (`float32` or `uint8`, 1–4 components, **2-D or 3-D**), or buffers |
    | — | buffers initialised from NumPy arrays |
    | — | `uniform`s driven by marimo sliders, no recompiles |
    | — | snapshot any buffer back to NumPy (`widget.snapshot`) |

    Everything on this page runs client-side: Python via Pyodide, shaders via WebGL2.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Playground: it's ShaderToy

    Pick a preset and edit the GLSL; it recompiles as you type. Drag on the canvas to set `iMouse`.
    The **Buffer A** tab is optional: when it holds code, Buffer A renders every frame,
    reads its own previous frame through `iChannel0`, and the image pass sees it as `iChannel0`.
    `iChannel1` is a 256×256 RGBA noise texture made with NumPy.
    """)
    return


@app.cell(hide_code=True)
def _():
    PRESETS = {
        "Hello, ShaderToy": {
            "image": r"""void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord / iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0, 2, 4));

    // Output to screen
    fragColor = vec4(col, 1.0);
}
""",
            "buffer_a": "",
        },
        "Mandelbrot zoom (colormaps)": {
            "image": r"""#include <colormaps>   // viridis(), magma(), turbo(), cmap(t, id), ...

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec2 center = vec2(-0.743643887, 0.131825904);
    float zoom = 1.6 * pow(0.6, 10.0 * (0.5 - 0.5 * cos(0.12 * iTime)));
    vec2 c = center + p * zoom;

    vec2 z = vec2(0.0);
    float n = 0.0;
    for (int i = 0; i < 400; i++) {
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (dot(z, z) > 256.0) break;
        n += 1.0;
    }
    vec3 col = vec3(0.0);
    if (n < 400.0) {
        float sn = n - log2(log2(dot(z, z))) + 4.0;   // smooth iteration count
        col = magma(0.5 - 0.5 * cos(0.15 * sn));
    }
    fragColor = vec4(col, 1.0);
}
""",
            "buffer_a": "",
        },
        "Feedback buffer: ink": {
            "image": r"""// iChannel0 = Buffer A
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec3 ink = texture(iChannel0, fragCoord / iResolution.xy).rgb;
    fragColor = vec4(1.0 - exp(-1.5 * ink), 1.0);   // tone map
}
""",
            "buffer_a": r"""// Buffer A reads its own previous frame through iChannel0 (ping-pong).
// Drag on the canvas to paint; the ink is advected by a swirling flow.
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec2 p = uv - 0.5;
    vec2 vel = 0.004 * vec2(-p.y, p.x)
             + 0.0015 * vec2(sin(6.0 * p.y + iTime), cos(6.0 * p.x - iTime));
    vec3 ink = 0.995 * texture(iChannel0, uv - vel).rgb;

    vec2 pen = iMouse.z > 0.0
        ? iMouse.xy
        : iResolution.xy * (0.5 + 0.3 * vec2(sin(1.1 * iTime), sin(1.7 * iTime)));
    float r = length(fragCoord - pen) / iResolution.y;
    vec3 hue = 0.5 + 0.5 * cos(0.7 * iTime + vec3(0.0, 2.0, 4.0));
    ink += 0.6 * hue * exp(-2500.0 * r * r);

    fragColor = vec4(min(ink, vec3(4.0)), 1.0);
}
""",
        },
        "Domain warping (NumPy noise texture)": {
            "image": r"""// iChannel1: 256x256 RGBA uint8 noise from
//   np.random.default_rng(0).integers(0, 256, (256, 256, 4), dtype=np.uint8)
float noise(vec2 p) { return texture(iChannel1, p / 256.0).x; }

float fbm(vec2 p)
{
    const mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    float f = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) { f += a * noise(p); p = m * p; a *= 0.5; }
    return f;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 p = 12.0 * fragCoord / iResolution.y;
    vec2 q = vec2(fbm(p + 0.2 * iTime), fbm(p + vec2(5.2, 1.3)));
    vec2 r = vec2(fbm(p + 4.0 * q + vec2(1.7, 9.2) + 0.1 * iTime),
                  fbm(p + 4.0 * q + vec2(8.3, 2.8)));
    float f = fbm(p + 4.0 * r);

    vec3 col = mix(vec3(0.10, 0.25, 0.40), vec3(0.90, 0.65, 0.35), clamp(2.0 * f * f, 0.0, 1.0));
    col = mix(col, vec3(0.00, 0.08, 0.20), 0.6 * clamp(length(q), 0.0, 1.0));
    col = mix(col, vec3(0.95, 0.90, 0.85), 0.5 * clamp(r.x * r.x, 0.0, 1.0));
    fragColor = vec4(1.4 * f * col, 1.0);
}
""",
            "buffer_a": "",
        },
    }
    return (PRESETS,)


@app.cell(hide_code=True)
def _(PRESETS):
    preset = mo.ui.dropdown(options=list(PRESETS), value="Hello, ShaderToy", label="Preset")
    preset
    return (preset,)


@app.cell
def _():
    noise_rgba = np.random.default_rng(0).integers(0, 256, (256, 256, 4), dtype=np.uint8)

    def playground_scene(image: str, buffer_a: str) -> dict:
        noise = Channel(noise_rgba, wrap="repeat", filter="mipmap")
        buffers = {}
        if buffer_a.strip():
            buffers["A"] = Buffer(buffer_a, channels=["A", noise])
        return dict(image=image, buffers=buffers, channels=["A" if buffers else None, noise])
    return (playground_scene,)


@app.cell
def _(PRESETS, playground_scene, preset):
    # Re-created when the preset changes, so time and buffers start fresh.
    _p = PRESETS[preset.value]
    playground = mo.ui.anywidget(ShaderToy(**playground_scene(_p["image"], _p["buffer_a"]), height=380))
    playground
    return (playground,)


@app.cell(hide_code=True)
def _(PRESETS, preset):
    _p = PRESETS[preset.value]
    image_code = mo.ui.code_editor(_p["image"], language="cpp", min_height=260, debounce=400, theme="dark")
    buffer_a_code = mo.ui.code_editor(
        _p["buffer_a"],
        language="cpp",
        min_height=260,
        debounce=400,
        theme="dark",
        placeholder="// optional: void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }",
    )
    mo.ui.tabs({"Image": image_code, "Buffer A": buffer_a_code})
    return buffer_a_code, image_code


@app.cell
def _(buffer_a_code, image_code, playground, playground_scene):
    # Editing only updates the widget's sources; it is not re-created.
    playground.update(**playground_scene(image_code.value, buffer_a_code.value))
    return


@app.cell(hide_code=True)
def _(playground):
    error_callout(playground)
    return


@app.function(hide_code=True)
def error_callout(widget) -> Any:
    """Render the widget's shader compile errors (synced back from the browser)."""
    errors = widget.errors
    if not errors:
        return mo.md("")
    lines = []
    for _pass, items in errors.items():
        for e in items:
            where = f"{e['source']}:{e['line']}" if e.get("line") else (e.get("source") or _pass)
            lines.append(f"{where}  {e['message']}")
    return mo.callout(mo.md("**Shader errors**\n\n```text\n" + "\n".join(lines) + "\n```"), kind="danger")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. NumPy array → texture: inspecting a 2-D field

    A 512×512 `float32` elevation field is synthesised in NumPy (spectral synthesis with power
    spectrum $P(k) \propto k^{-\beta}$) and bound to `iChannel0`. The shader does the
    visualization: colormap, hill-shading from the gradient, anti-aliased contour lines, and
    **click anywhere to highlight the iso-line through that elevation**.

    There are two kinds of controls. The *data* controls re-run NumPy and re-upload the
    texture. The *view* controls only set GLSL `uniform`s: no recompile, no data transfer.
    """)
    return


@app.function
def fractal_terrain(n: int = 512, beta: float = 3.0, seed: int = 0) -> np.ndarray:
    """Periodic random field with power spectrum ~ k^-beta, scaled to [0, 1]."""
    rng = np.random.default_rng(seed)
    kx, ky = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n))
    k = np.hypot(kx, ky)
    k[0, 0] = np.inf  # drop the mean
    spectrum = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * k ** (-beta / 2)
    z = np.fft.ifft2(spectrum).real
    return ((z - z.min()) / np.ptp(z)).astype(np.float32)


@app.cell(hide_code=True)
def _():
    beta = mo.ui.slider(2.0, 5.0, step=0.1, value=4.0, label="spectral slope β", show_value=True)
    terrain_seed = mo.ui.number(0, 9999, value=7, label="seed")
    sea = mo.ui.slider(0.0, 0.9, step=0.01, value=0.35, label="sea level", show_value=True)
    spacing = mo.ui.slider(0.01, 0.2, step=0.005, value=0.04, label="contour spacing", show_value=True)
    relief = mo.ui.slider(0.0, 600.0, step=10.0, value=120.0, label="relief", show_value=True)
    sun = mo.ui.slider(0, 360, step=5, value=135, label="sun azimuth (°)", show_value=True)
    terrain_cmap = mo.ui.dropdown(
        options={"viridis": 0, "magma": 1, "inferno": 2, "plasma": 3, "cividis": 4, "turbo": 5, "coolwarm": 6},
        value="viridis",
        label="colormap",
    )
    zoom = mo.ui.slider(0.5, 4.0, step=0.1, value=1.0, label="zoom", show_value=True)
    mo.vstack(
        [
            mo.hstack([mo.md("**data (NumPy)**"), beta, terrain_seed], justify="start", gap=1.5, wrap=True),
            mo.hstack([mo.md("**view (uniforms)**"), sea, spacing, relief, sun, terrain_cmap, zoom], justify="start", gap=1.5, wrap=True),
        ]
    )
    return beta, relief, sea, spacing, sun, terrain_cmap, terrain_seed, zoom


@app.cell
def _(beta, terrain_seed):
    elevation = fractal_terrain(512, beta.value, int(terrain_seed.value))
    return (elevation,)


@app.cell
def _():
    TERRAIN = r"""#include <colormaps>
// iChannel0: elevation from NumPy (512x512 float32 in [0, 1], wrap = repeat)
uniform float uSea, uSpacing, uRelief, uSun, uZoom;
uniform int uCmap;

vec2 toUV(vec2 frag) { return (frag - 0.5 * iResolution.xy) / iResolution.y / uZoom + 0.5; }
float elev(vec2 uv) { return texture(iChannel0, uv).r; }

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = toUV(fragCoord);
    vec2 e = 1.0 / iChannelResolution[0].xy;
    float h = elev(uv);
    vec2 g = vec2(elev(uv + vec2(e.x, 0)) - elev(uv - vec2(e.x, 0)),
                  elev(uv + vec2(0, e.y)) - elev(uv - vec2(0, e.y)));

    // hill-shading
    vec3 n = normalize(vec3(-0.5 * uRelief * g, 1.0));
    vec3 sun = normalize(vec3(cos(uSun), sin(uSun), 0.9));
    float shade = clamp(dot(n, sun), 0.0, 1.0);

    bool land = h >= uSea;
    vec3 col = land
        ? cmap((h - uSea) / max(1.0 - uSea, 1e-3), uCmap) * (0.3 + 0.9 * shade)
        : mix(vec3(0.02, 0.08, 0.20), vec3(0.10, 0.32, 0.50), h / max(uSea, 1e-3));

    // anti-aliased contour lines
    float v = h / uSpacing;
    float line = abs(fract(v - 0.5) - 0.5) / max(fwidth(v), 1e-4);
    col = mix(col, 0.2 * col, 0.8 * (1.0 - smoothstep(0.0, 1.2, line)) * float(land));

    // iso-line through the clicked elevation
    if (iMouse.x > 0.0) {
        float h0 = elev(toUV(iMouse.xy));
        float d = abs(h - h0) / max(fwidth(h), 1e-6);
        col = mix(col, vec3(1.0, 0.92, 0.25), 1.0 - smoothstep(1.0, 2.5, d));
    }
    fragColor = vec4(col, 1.0);
}
"""
    terrain = mo.ui.anywidget(ShaderToy(TERRAIN, height=440))
    terrain
    return (terrain,)


@app.cell
def _(elevation, terrain):
    terrain.channels = [Channel(elevation, wrap="repeat")]
    return


@app.cell
def _(relief, sea, spacing, sun, terrain, terrain_cmap, zoom):
    terrain.uniforms = dict(
        uSea=sea.value,
        uSpacing=spacing.value,
        uRelief=relief.value,
        uSun=np.radians(sun.value),
        uZoom=zoom.value,
        uCmap=terrain_cmap.value,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. NumPy → GPU simulation → NumPy

    A Gray–Scott reaction–diffusion simulation. **Buffer A** holds the state
    $(U, V)$ in its red and green channels. It is **initialised from a NumPy array**
    (640×256×2), steps itself forward several times per frame (`iterations`), and is shown
    with a colormap by the image pass. Drag on the canvas to inject $V$. `uFeed` and `uKill`
    come from the sliders.

    $$\partial_t U = D_U \nabla^2 U - UV^2 + F(1-U), \qquad
      \partial_t V = D_V \nabla^2 V + UV^2 - (F+k)V$$
    """)
    return


@app.function
def text_mask(text: str, scale: int) -> np.ndarray:
    """Rasterize capital letters with a tiny 5x7 bitmap font (row 0 = top)."""
    font = {
        "M": "10001 11011 11111 10101 10001 10001 10001",
        "A": "01110 10001 10001 11111 10001 10001 10001",
        "R": "11110 10001 10001 11110 10100 10010 10001",
        "I": "01110 00100 00100 00100 00100 00100 01110",
        "O": "01110 10001 10001 10001 10001 10001 01110",
        " ": "00000 00000 00000 00000 00000 00000 00000",
    }
    glyphs = [np.array([[c == "1" for c in row] for row in font[ch].split()]) for ch in text]
    gap = np.zeros((7, 1), bool)
    bitmap = np.hstack([part for g in glyphs for part in (g, gap)][:-1])
    return np.kron(bitmap, np.ones((scale, scale))).astype(bool)


@app.function
def gray_scott_seed(pattern: str, h: int = 256, w: int = 640, seed: int = 0) -> np.ndarray:
    """Initial (U, V) state as an (h, w, 2) float32 array; row 0 is the bottom of the canvas."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    v = np.zeros((h, w), bool)
    if pattern == "random spots":
        for cx, cy, r in zip(rng.uniform(0, w, 40), rng.uniform(0, h, 40), rng.uniform(2, 7, 40)):
            v |= (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
    elif pattern == "ring":
        v = np.abs(np.hypot(xx - w / 2, yy - h / 2) - 0.32 * h) < 2.5
    elif pattern == "text":
        mask = np.flipud(text_mask("MARIMO", 8))  # flip: array row 0 is the *bottom* row
        y0, x0 = (h - mask.shape[0]) // 2, (w - mask.shape[1]) // 2
        v[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
    else:  # "noise"
        v = rng.random((h, w)) < 0.004
    state = np.empty((h, w, 2), np.float32)
    state[..., 0] = np.where(v, 0.5, 1.0)
    state[..., 1] = np.where(v, 0.25, 0.0) + 0.02 * rng.random((h, w))
    return state


@app.cell(hide_code=True)
def _():
    GS_REGIMES = {
        "coral": (0.0545, 0.062),
        "mitosis": (0.0367, 0.0649),
        "maze": (0.029, 0.057),
        "solitons": (0.030, 0.062),
        "worms": (0.078, 0.061),
        "holes": (0.039, 0.058),
        "chaos": (0.026, 0.051),
    }
    gs_pattern = mo.ui.dropdown(["text", "random spots", "ring", "noise"], value="text", label="NumPy seed")
    gs_seed = mo.ui.number(0, 9999, value=0, label="rng seed")
    gs_regime = mo.ui.dropdown(list(GS_REGIMES), value="coral", label="regime")
    gs_speed = mo.ui.slider(1, 40, value=16, label="steps / frame", show_value=True)
    mo.hstack([gs_pattern, gs_seed, gs_regime, gs_speed], justify="start", gap=1.5, wrap=True)
    return GS_REGIMES, gs_pattern, gs_regime, gs_seed, gs_speed


@app.cell(hide_code=True)
def _(GS_REGIMES, gs_regime):
    _F, _k = GS_REGIMES[gs_regime.value]
    feed = mo.ui.slider(0.01, 0.1, step=0.0005, value=_F, label="feed F", show_value=True)
    kill = mo.ui.slider(0.04, 0.075, step=0.0005, value=_k, label="kill k", show_value=True)
    mo.hstack([feed, kill], justify="start", gap=1.5)
    return feed, kill


@app.cell
def _(gs_pattern, gs_seed):
    seed_state = gray_scott_seed(gs_pattern.value, seed=int(gs_seed.value))
    return (seed_state,)


@app.cell
def _():
    GRAY_SCOTT = r"""// Buffer A: one explicit Euler step of Gray-Scott per iteration (Karl Sims' scheme).
// iChannel0 = this buffer (previous state): r = U, g = V. Periodic boundaries.
uniform float uFeed, uKill;
const float Du = 1.0, Dv = 0.5;

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    ivec2 size = ivec2(iResolution.xy);
    ivec2 p = ivec2(fragCoord);
    #define S(dx, dy) texelFetch(iChannel0, (p + ivec2(dx, dy) + size) % size, 0).rg
    vec2 c = S(0, 0);
    vec2 lap = 0.2 * (S(1, 0) + S(-1, 0) + S(0, 1) + S(0, -1))
             + 0.05 * (S(1, 1) + S(-1, 1) + S(1, -1) + S(-1, -1)) - c;
    float u = c.r, v = c.g, uvv = u * v * v;
    u += Du * lap.r - uvv + uFeed * (1.0 - u);
    v += Dv * lap.g + uvv - (uKill + uFeed) * v;

    // iMouse is scaled to this pass's resolution, so painting lines up with the image.
    if (iMouse.z > 0.0 && length(fragCoord - iMouse.xy) < 5.0) v = 0.5;
    fragColor = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), 0.0, 1.0);
}
"""
    GS_IMAGE = r"""#include <colormaps>
// iChannel0 = Buffer A (640x256, stretched to the canvas)
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    float v = texture(iChannel0, fragCoord / iResolution.xy).g;
    fragColor = vec4(inferno(smoothstep(0.0, 0.42, v)), 1.0);
}
"""
    gs = mo.ui.anywidget(ShaderToy(GS_IMAGE, channels=["A"], buffers={"A": Buffer(GRAY_SCOTT)}, height=300, snapshot_target="A"))
    gs
    return GRAY_SCOTT, gs


@app.cell
def _(GRAY_SCOTT, gs, gs_speed, seed_state):
    # A new seed array re-initialises Buffer A; changing the speed keeps the running state.
    gs.buffers = {"A": Buffer(GRAY_SCOTT, channels=["A"], init=seed_state, iterations=gs_speed.value)}
    return


@app.cell
def _(feed, gs, kill):
    gs.uniforms = dict(uFeed=feed.value, uKill=kill.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Read the GPU state back into NumPy

    Press the button (or the camera icon in the widget toolbar). The browser reads Buffer A
    with `readPixels` and sends the `float32` array to Python: `gs.snapshot` has shape
    `(256, 640, 4)`. Below, NumPy computes the radially averaged power spectrum of $V$. Its
    peak gives the characteristic wavelength of the Turing pattern. The 2-D spectrum goes back
    to the GPU as a texture.
    """)
    return


@app.cell(hide_code=True)
def _():
    snap_button = mo.ui.run_button(label="Snapshot Buffer A → NumPy")
    snap_button
    return (snap_button,)


@app.cell
def _(gs, snap_button):
    if snap_button.value:
        gs.request_snapshot("A")
    return


@app.function
def radial_spectrum(field: np.ndarray, nbins: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D power spectrum (fftshifted) and its radial average vs. spatial frequency (cycles/px)."""
    h, w = field.shape
    power = np.abs(np.fft.fft2(field - field.mean())) ** 2
    kr = np.hypot(*np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h)))
    edges = np.linspace(0, 0.5, nbins + 1)
    idx = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, nbins)
    sums = np.bincount(idx, weights=power.ravel(), minlength=nbins + 1)[:nbins]
    counts = np.bincount(idx, minlength=nbins + 1)[:nbins]
    radial = sums / np.maximum(counts, 1)
    return np.fft.fftshift(power), 0.5 * (edges[1:] + edges[:-1]), radial


@app.cell
def _(gs):
    snapshot = gs.snapshot  # None until the browser has sent one
    spectrum2d = None
    if snapshot is None:
        _out = mo.callout(mo.md("No snapshot yet. Press the button above or the camera icon."), kind="neutral")
    else:
        _V = snapshot[..., 1]
        spectrum2d, _freqs, _radial = radial_spectrum(_V)
        _band = _freqs > 1 / 64  # look for the pattern scale, not the domain scale
        _peak = _freqs[_band][np.argmax(_radial[_band])]
        _info = gs.snapshot_info
        _stats = mo.md(
            f"""
    **snapshot** `{snapshot.dtype}{list(snapshot.shape)}` at frame {_info["frame"]} (t = {_info["time"]:.1f} s)

    | quantity | value |
    |:---|---:|
    | mean V | {_V.mean():.4f} |
    | coverage (V > 0.2) | {(_V > 0.2).mean():.1%} |
    | pattern wavelength (peak, λ < 64 px) | {1 / _peak:.1f} px |
    """
        )
        _chart = line_chart(_freqs[1:], _radial[1:], "spatial frequency (cycles / px)", "power", peak=_peak)
        _out = mo.hstack([_stats, _chart], widths=[1, 1.4], align="center")
    _out
    return (spectrum2d,)


@app.cell
def _(spectrum2d):
    SPECTRUM = r"""#include <colormaps>
// iChannel0: log10 power spectrum computed in NumPy (fftshifted, normalised to [0, 1])
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 tex = iChannelResolution[0].xy;                       // 640 x 256
    float scale = min(iResolution.x / tex.x, iResolution.y / tex.y);
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / (scale * tex) + 0.5;
    bool inside = all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));
    fragColor = vec4(inside ? viridis(texture(iChannel0, uv).r) : vec3(0.05), 1.0);
}
"""
    if spectrum2d is not None:
        _logp = np.log10(spectrum2d + 1e-12)
        _lo, _hi = np.percentile(_logp, [5, 99.9])
        _spec_tex = np.clip((_logp - _lo) / max(_hi - _lo, 1e-9), 0, 1).astype(np.float32)
        _w = ShaderToy(SPECTRUM, channels=[Channel(_spec_tex, filter="nearest")], height=280, paused=True)
        _view = mo.vstack([mo.md("**2-D power spectrum of V (log scale), computed in NumPy, drawn by a shader**"), _w])
    else:
        _view = None
    _view
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. 3-D NumPy array → `sampler3D`: volume rendering

    A hydrogen-atom orbital $\psi_{n\ell m}$ is evaluated on a 96³ grid in NumPy and uploaded
    as a 3-D texture (`(D, H, W, 2)` float32: $|\psi|^2$ and the sign of $\psi$). The shader
    ray-marches it, either as emission–absorption volume rendering or as an isosurface.
    Drag to rotate.
    """)
    return


@app.function
def orbital_volume(name: str, n: int = 96) -> np.ndarray:
    """Real hydrogen orbitals (unnormalised) on an n^3 grid -> (n, n, n, 2): |psi|^2 / max, sign(psi)."""
    orbitals = {  # psi(x, y, z, r) with the polar axis along z, and the half-width of the box (in Bohr radii)
        "2p (z)": (lambda x, y, z, r: z * np.exp(-r / 2), 16.0),
        "3d (z²)": (lambda x, y, z, r: (3 * z**2 - r**2) * np.exp(-r / 3), 30.0),
        "3d (xy)": (lambda x, y, z, r: x * y * np.exp(-r / 3), 30.0),
        "4f (z³)": (lambda x, y, z, r: z * (5 * z**2 - 3 * r**2) * np.exp(-r / 4), 48.0),
        "4f (xyz)": (lambda x, y, z, r: x * y * z * np.exp(-r / 4), 48.0),
    }
    psi_fn, half_width = orbitals[name]
    lin = np.linspace(-half_width, half_width, n, dtype=np.float32)
    # Texture axes: (w, v, u) = (array axis 0, 1, 2) = shader (z, y, x). Put the polar axis on screen-up y.
    sz, sy, sx = np.meshgrid(lin, lin, lin, indexing="ij")
    r = np.sqrt(sx**2 + sy**2 + sz**2)
    psi = psi_fn(sx, sz, sy, r)
    density = psi**2
    return np.stack([density / density.max(), (np.sign(psi) + 1) / 2], axis=-1).astype(np.float32)


@app.cell(hide_code=True)
def _():
    orbital = mo.ui.dropdown(["2p (z)", "3d (z²)", "3d (xy)", "4f (z³)", "4f (xyz)"], value="3d (z²)", label="orbital")
    vol_mode = mo.ui.radio({"volume": 0, "isosurface": 1}, value="volume", label="mode", inline=True)
    density = mo.ui.slider(1.0, 600.0, step=1.0, value=300.0, label="density", show_value=True)
    gamma = mo.ui.slider(0.1, 2.5, step=0.05, value=1.5, label="gamma", show_value=True)
    iso = mo.ui.slider(0.001, 0.5, step=0.001, value=0.05, label="iso level", show_value=True)
    steps = mo.ui.slider(32, 384, step=16, value=192, label="ray steps", show_value=True)
    mo.hstack([orbital, vol_mode, density, gamma, iso, steps], justify="start", gap=1.5, wrap=True)
    return density, gamma, iso, orbital, steps, vol_mode


@app.cell
def _(orbital):
    volume = orbital_volume(orbital.value)
    return (volume,)


@app.cell
def _():
    VOLUME = r"""// iChannel0: 3-D texture from NumPy, (D, H, W, 2): r = |psi|^2 (normalised), g = sign(psi) in {0, 1}
uniform float uDensity, uGamma, uIso;
uniform int uMode, uSteps;

vec2 boxHit(vec3 ro, vec3 rd)
{
    vec3 inv = 1.0 / rd;
    vec3 t0 = (-1.0 - ro) * inv, t1 = (1.0 - ro) * inv;
    vec3 lo = min(t0, t1), hi = max(t0, t1);
    return vec2(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}
vec4 vol(vec3 p) { return texture(iChannel0, 0.5 * p + 0.5); }
vec3 phase(float s) { return mix(vec3(0.20, 0.45, 1.00), vec3(1.00, 0.40, 0.15), s); }

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float yaw = 0.35 * iTime, pitch = 0.35;
    if (iMouse.x > 0.0) {
        yaw = 6.2832 * (iMouse.x / iResolution.x - 0.5);
        pitch = 3.0 * (iMouse.y / iResolution.y - 0.5);
    }
    vec3 ro = 3.2 * vec3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw));
    vec3 fw = normalize(-ro), rt = normalize(cross(fw, vec3(0, 1, 0))), up = cross(rt, fw);
    vec3 rd = normalize(p.x * rt + p.y * up + 2.2 * fw);

    vec3 bg = mix(vec3(0.015, 0.02, 0.035), vec3(0.07, 0.08, 0.12), 0.5 + 0.5 * p.y);
    vec3 col = bg;
    vec2 t = boxHit(ro, rd);
    if (t.x < t.y) {
        t.x = max(t.x, 0.0);
        float dt = 3.4641 / float(uSteps);
        float jitter = fract(sin(dot(fragCoord, vec2(12.9898, 78.233))) * 43758.5453);
        vec3 acc = vec3(0.0);
        float trans = 1.0;
        for (int i = 0; i < 512; i++) {
            float s = t.x + (float(i) + jitter) * dt;
            if (i >= uSteps || s > t.y) break;
            vec4 v = vol(ro + rd * s);
            if (uMode == 0) {
                float d = pow(v.r, uGamma);
                float a = 1.0 - exp(-uDensity * d * dt);
                acc += trans * a * phase(v.g) * (0.5 + 1.5 * d);
                trans *= 1.0 - a;
                if (trans < 0.01) break;
            } else if (v.r > uIso) {
                float a = s - dt, b = s;   // refine the crossing by bisection
                for (int k = 0; k < 6; k++) {
                    float mid = 0.5 * (a + b);
                    if (vol(ro + rd * mid).r > uIso) b = mid; else a = mid;
                }
                vec3 q = ro + rd * b;
                vec2 e = vec2(2.0 / iChannelResolution[0].x, 0.0);
                vec3 n = -normalize(vec3(vol(q + e.xyy).r - vol(q - e.xyy).r,
                                         vol(q + e.yxy).r - vol(q - e.yxy).r,
                                         vol(q + e.yyx).r - vol(q - e.yyx).r));
                vec3 l = normalize(vec3(0.6, 0.8, 0.4));
                float diff = max(dot(n, l), 0.0);
                float spec = pow(max(dot(reflect(-l, n), -rd), 0.0), 32.0);
                acc = phase(vol(q).g) * (0.2 + 0.8 * diff) + 0.35 * spec;
                trans = 0.0;
                break;
            }
        }
        col = acc + trans * bg;
    }
    fragColor = vec4(pow(col, vec3(0.4545)), 1.0);
}
"""
    vol = mo.ui.anywidget(ShaderToy(VOLUME, height=460))
    vol
    return (vol,)


@app.cell
def _(vol, volume):
    vol.channels = [Channel(volume)]  # 4-D array -> sampler3D
    return


@app.cell
def _(density, gamma, iso, steps, vol, vol_mode):
    vol.uniforms = dict(uDensity=density.value, uGamma=gamma.value, uIso=iso.value, uMode=vol_mode.value, uSteps=steps.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. API

    ```python
    from shadertoy import ShaderToy, Buffer, Channel   # this notebook is importable

    w = mo.ui.anywidget(ShaderToy(
        image_code,                       # GLSL with mainImage(out vec4, in vec2)
        channels=[field, "A"],            # iChannel0..7: arrays, buffer names, Channel(...)
        buffers={"A": Buffer(code, channels=["A"], init=state0, iterations=8)},
        common="...",                     # GLSL shared by all passes
        uniforms={"uScale": 1.0},         # float / int / bool / vecN / matN / float[N]
        height=400, resolution_scale=1.0, paused=False,
    ))
    w.uniforms = {"uScale": slider.value}            # cheap: no recompile
    w.channels = [Channel(new_field, filter="nearest", wrap="repeat")]
    w.update(image=..., buffers=..., channels=...)     # several changes, one message
    w.reset(); w.request_snapshot("A")                 # snapshot -> w.snapshot (H, W, 4)
    w.errors                                           # compile errors, per pass
    ```

    **Arrays.** `float*`/`int*`/`bool` are uploaded as `float32`. `uint8` stays normalised to
    [0, 1]. Shapes: `(H, W)`, `(H, W, C≤4)` → `sampler2D`. `(D, H, W)`, `(D, H, W, C≤4)` →
    `sampler3D` (`Channel(volume=True/False)` overrides the guess). `arr[i, j]` is texel
    `(x=j, y=i)`, so row 0 is the **bottom**: `texelFetch` is index-exact. Use
    `Channel(img, origin="upper")` for pictures.

    **Uniforms.** Values are auto-declared unless your code declares them. Python `int` →
    `int`, `float` → `float`. Declared types win, so `uniform float uN;` accepts an `int`.

    **Extras over ShaderToy.** `#include <colormaps>` (viridis, magma, inferno, plasma,
    cividis, turbo, coolwarm, `cmap(t, id)`). `iHover` (pointer position without clicking).
    Fixed-size buffers, where `iMouse`/`iHover` are scaled to the pass's resolution. Sub-steps
    with `iterations`. `#include <yours>` via `ShaderToy(includes={"yours": "..."})`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ### Appendix: implementation

    The widget is a self-contained [anywidget](https://anywidget.dev): a ~600-line
    WebGL2 renderer in JavaScript plus the Python classes below. They are top-level marimo
    definitions, so `from shadertoy import ShaderToy, Buffer, Channel` works from other code.
    """)
    return


@app.function(hide_code=True)
def line_chart(x, y, xlabel: str, ylabel: str, peak: float | None = None, width: int = 460, height: int = 230) -> mo.Html:
    """Minimal dependency-free SVG line chart (log y-axis)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (y > 0) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return mo.md("_(no signal to plot)_")
    left, right, top, bottom = 52, 16, 12, 40
    pw, ph = width - left - right, height - top - bottom
    x0, x1 = 0.0, float(x.max())
    ly = np.log10(y)
    y0, y1 = np.floor(ly.min()), np.ceil(ly.max())
    y1 = max(y1, y0 + 1)

    def sx(v):
        return left + (v - x0) / (x1 - x0) * pw

    def sy(v):
        return top + (1 - (v - y0) / (y1 - y0)) * ph

    muted = 'fill="currentColor" fill-opacity="0.65"'
    out = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{ylabel} versus {xlabel}" '
        f'style="max-width:{width}px;font:11px system-ui,sans-serif;font-variant-numeric:tabular-nums">'
    ]
    for d in range(int(y0), int(y1) + 1, max(1, int(np.ceil((y1 - y0) / 5)))):
        yy = sy(d)
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="currentColor" stroke-opacity="0.12"/>')
        out.append(f'<text x="{left - 6}" y="{yy:.1f}" text-anchor="end" dominant-baseline="middle" {muted}>10<tspan dy="-5" font-size="8">{d}</tspan></text>')
    for t in np.arange(0.0, x1 + 1e-9, 0.1):
        out.append(f'<text x="{sx(t):.1f}" y="{top + ph + 16}" text-anchor="middle" {muted}>{t:.1f}</text>')
    out.append(f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" stroke="currentColor" stroke-opacity="0.35"/>')
    out.append(f'<text x="{left + pw / 2}" y="{height - 4}" text-anchor="middle" {muted}>{xlabel}</text>')
    out.append(f'<text transform="translate(12 {top + ph / 2}) rotate(-90)" text-anchor="middle" {muted}>{ylabel}</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(a):.1f},{sy(b):.1f}" for i, (a, b) in enumerate(zip(x, ly)))
    out.append(f'<path d="{path}" fill="none" stroke="#3987e5" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    if peak is not None:
        py = sy(np.interp(peak, x, ly))
        anchor = "end" if sx(peak) > left + 0.7 * pw else "start"
        dx = -8 if anchor == "end" else 8
        out.append(f'<circle cx="{sx(peak):.1f}" cy="{py:.1f}" r="4" fill="#3987e5"/>')
        out.append(f'<text x="{sx(peak) + dx:.1f}" y="{py - 6:.1f}" text-anchor="{anchor}" fill="currentColor">peak: λ = {1 / peak:.1f} px</text>')
    # hover layer: one invisible column per sample, with a native tooltip
    edges = np.concatenate([[x[0]], (x[1:] + x[:-1]) / 2, [x[-1]]])
    for a, b, xv, yv in zip(edges[:-1], edges[1:], x, y):
        out.append(
            f'<rect x="{sx(a):.1f}" y="{top}" width="{max(sx(b) - sx(a), 1):.1f}" height="{ph}" fill="transparent">'
            f"<title>{xlabel}: {xv:.3f}\nwavelength: {1 / xv:.1f} px\n{ylabel}: {yv:.3g}</title></rect>"
        )
    out.append("</svg>")
    return mo.Html("".join(out))


@app.class_definition
@dataclass
class Channel:
    """Input for one `iChannelN` slot.

    source: a NumPy array (uploaded as a texture) or the name of a buffer pass, e.g. "A".
    filter: "linear" | "nearest" | "mipmap" (mipmaps are generated for uint8 arrays only).
    wrap:   "clamp" | "repeat" | "mirror".
    volume: treat a 3-D array as (depth, height, width) -> sampler3D. Default: True for 4-D
            arrays and for 3-D arrays whose last axis is longer than 4.
    origin: "lower" (default) maps arr[0] to the bottom row, so texelFetch(ivec2(j, i)) == arr[i, j];
            "upper" flips vertically, so pictures appear upright.
    """

    source: Any
    filter: str = "linear"
    wrap: str = "clamp"
    volume: bool | None = None
    origin: str = "lower"


@app.class_definition
@dataclass
class Buffer:
    """An offscreen pass, like ShaderToy's "Buffer A": an RGBA float texture redrawn every frame.

    code:       GLSL defining `void mainImage(out vec4 fragColor, in vec2 fragCoord)`.
    channels:   inputs, like the image pass. Reading the buffer's own name gives its previous frame.
    init:       optional NumPy array, (H, W) or (H, W, C<=4), loaded into the buffer on start and
                reset. Missing components are 0.
    size:       (width, height). Defaults to init's shape, else the canvas resolution.
    iterations: how many times the pass runs per frame (simulation sub-steps).
    """

    code: str
    channels: Any = ()
    init: Any = None
    size: tuple[int, int] | None = None
    iterations: int = 1


@app.class_definition(hide_code=True)
class ShaderToy(anywidget.AnyWidget):
    """ShaderToy-style multipass GLSL renderer whose channels and buffers can be NumPy arrays.

    ShaderToy(image, *, channels=(), buffers=None, common="", uniforms=None, includes=None,
              height=400, resolution_scale=1.0, paused=False, snapshot_target="")
    """

    _esm = r"""
// marimo-shadertoy: a ShaderToy-style multipass WebGL2 renderer (anywidget front-end).
const ICONS = {
  play: '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>',
  reset: '<svg viewBox="0 0 16 16"><path d="M2.5 2.5h2v11h-2zM13.5 2.5v11L5.5 8z"/></svg>',
  step: '<svg viewBox="0 0 16 16"><path d="M2.5 2.5v11l8-5.5zM11.5 2.5h2v11h-2z"/></svg>',
  camera: '<svg viewBox="0 0 16 16"><path fill-rule="evenodd" d="M5.5 2.5 4.5 4h-2a1 1 0 0 0-1 1v7.5a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-2l-1-1.5zM8 6a2.75 2.75 0 1 1 0 5.5A2.75 2.75 0 0 1 8 6z"/></svg>',
  expand: '<svg viewBox="0 0 16 16"><path d="M2 2h5v2H4v3H2zM14 2v5h-2V4H9V2zM2 14V9h2v3h3v2zM14 14H9v-2h3V9h2z"/></svg>',
};

const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Shader source-string numbers used with `#line`, so compile errors point at user code.
const SRC_CODE = 0;
const SRC_COMMON = 1;
const SRC_HEADER = 2;

// Only one view per model answers snapshot requests and reports errors.
const VIEWS = new WeakMap();

function bytesOf(x) {
  if (x == null) return null;
  if (x instanceof DataView) return new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
  if (x instanceof ArrayBuffer) return new Uint8Array(x);
  if (ArrayBuffer.isView(x)) return new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
  if (typeof x === "string") {
    const bin = atob(x);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  throw new Error("unsupported binary payload");
}

function typedArray(bytes, dtype) {
  if (dtype === "uint8") return bytes;
  if (bytes.byteOffset % 4 !== 0) bytes = bytes.slice();
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
}

function passLabel(name) {
  return name === "image" ? "Image" : `Buffer ${name}`;
}

function expandIncludes(src, srcId, ctx) {
  const lines = src.split("\n");
  const out = [];
  lines.forEach((line, idx) => {
    const m = /^\s*#\s*include\s*[<"]([^>"]+)[>"]\s*(\/\/.*)?$/.exec(line);
    if (!m) {
      out.push(line);
      return;
    }
    const name = m[1].trim();
    if (!(name in ctx.includes)) {
      out.push(`#error unknown include "${name}"`);
      return;
    }
    if (ctx.seen.has(name)) {
      out.push("");
      return;
    }
    ctx.seen.add(name);
    const id = ctx.nextId++;
    ctx.srcNames[id] = `<${name}>`;
    out.push(`#line 1 ${id}`);
    out.push(expandIncludes(ctx.includes[name], id, ctx));
    out.push(`#line ${idx + 2} ${srcId}`);
  });
  return out.join("\n");
}

function formatLog(log, srcNames) {
  const out = [];
  for (const raw of (log || "").split("\n")) {
    const line = raw.trim().replace(/\0/g, "");
    if (!line) continue;
    const m = /^(ERROR|WARNING):\s*(\d+):(\d+):\s*(.*)$/.exec(line);
    if (m) {
      if (m[1] === "WARNING") continue;
      out.push({ source: srcNames[m[2]] ?? `source ${m[2]}`, line: Number(m[3]), message: m[4] });
    } else {
      out.push({ source: "", line: 0, message: line });
    }
  }
  return out;
}

function render({ model, el }) {
  // ---------------------------------------------------------------- DOM
  const root = document.createElement("div");
  root.className = "mst-root";
  const wrap = document.createElement("div");
  wrap.className = "mst-wrap";
  const canvas = document.createElement("canvas");
  canvas.className = "mst-canvas";
  const overlay = document.createElement("pre");
  overlay.className = "mst-errors";
  const bar = document.createElement("div");
  bar.className = "mst-bar";
  wrap.append(canvas, overlay);
  root.append(wrap, bar);
  el.appendChild(root);

  const button = (icon, title, onClick) => {
    const b = document.createElement("button");
    b.className = "mst-btn";
    b.type = "button";
    b.title = title;
    b.innerHTML = ICONS[icon];
    b.addEventListener("click", onClick);
    bar.appendChild(b);
    return b;
  };
  button("reset", "Restart: time, frame and buffers", () => doReset());
  const playBtn = button("pause", "Play / pause", () => setPlaying(!S.playing, true));
  button("step", "Advance one frame", () => {
    if (S.playing) setPlaying(false, true);
    S.stepRequested = true;
  });
  const stats = document.createElement("span");
  stats.className = "mst-stats";
  bar.appendChild(stats);
  const spacer = document.createElement("span");
  spacer.className = "mst-spacer";
  bar.appendChild(spacer);
  const snapBtn = button("camera", "Snapshot to Python (widget.snapshot)", () => {
    S.pendingSnapshots.push({ id: `ui-${Date.now()}`, target: model.get("snapshot_target") || "" });
    S.needsDraw = true;
  });
  button("expand", "Fullscreen", () => {
    if (document.fullscreenElement === root) document.exitFullscreen();
    else root.requestFullscreen?.();
  });

  // Track the views of this model so only one of them talks back to Python.
  const viewToken = {};
  if (!VIEWS.has(model)) VIEWS.set(model, []);
  VIEWS.get(model).push(viewToken);
  const isPrimary = () => VIEWS.get(model)?.[0] === viewToken;

  // ---------------------------------------------------------------- state
  const S = {
    playing: !model.get("paused"),
    time: 0,
    frame: 0,
    started: false,
    dt: 1 / 60,
    fps: 60,
    lastNow: null,
    mouse: [0, 0, 0, 0],
    mouseDown: false,
    hover: [0, 0, 0, 0],
    visible: true,
    needsDraw: true,
    stepRequested: false,
    pendingSnapshots: [],
    dirty: { textures: true, program: true, buffers: true, reset: false },
    textures: {}, // name -> {tex, target, w, h, d, dtype, hash, mipmaps}
    inits: {}, // name -> {shape, components, data, hash}
    buffers: {}, // name -> {w, h, key, tex: [2], fbo: [2], read}
    pipeline: null,
    uniforms: model.get("_uniforms") || {},
    uniformSig: "",
    lost: false,
    lastStats: 0,
  };

  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
  });
  if (!gl) {
    overlay.textContent = "WebGL2 is not available in this browser.";
    overlay.style.display = "block";
    return () => root.remove();
  }

  const ext = {
    cbf: gl.getExtension("EXT_color_buffer_float"),
    fl: gl.getExtension("OES_texture_float_linear"),
  };
  // Buffers (render targets): RGBA32F when we can render to and filter it,
  // RGBA16F when we can only render floats, RGBA8 as a last resort.
  const BUF = ext.cbf && ext.fl
    ? { internal: gl.RGBA32F, type: gl.FLOAT, float: true, filterable: true }
    : ext.cbf
      ? { internal: gl.RGBA16F, type: gl.FLOAT, float: true, filterable: true }
      : { internal: gl.RGBA8, type: gl.UNSIGNED_BYTE, float: false, filterable: true };

  const FORMATS = {
    float32: {
      1: [gl.R32F, gl.RED], 2: [gl.RG32F, gl.RG], 3: [gl.RGB32F, gl.RGB], 4: [gl.RGBA32F, gl.RGBA],
    },
    uint8: {
      1: [gl.R8, gl.RED], 2: [gl.RG8, gl.RG], 3: [gl.RGB8, gl.RGB], 4: [gl.RGBA8, gl.RGBA],
    },
  };

  const f = (fn) => (loc, a) => fn.call(gl, loc, a);
  const m = (fn) => (loc, a) => fn.call(gl, loc, false, a);
  const SETTERS = new Map([
    [gl.FLOAT, { n: 1, k: "f", fn: f(gl.uniform1fv) }],
    [gl.FLOAT_VEC2, { n: 2, k: "f", fn: f(gl.uniform2fv) }],
    [gl.FLOAT_VEC3, { n: 3, k: "f", fn: f(gl.uniform3fv) }],
    [gl.FLOAT_VEC4, { n: 4, k: "f", fn: f(gl.uniform4fv) }],
    [gl.INT, { n: 1, k: "i", fn: f(gl.uniform1iv) }],
    [gl.INT_VEC2, { n: 2, k: "i", fn: f(gl.uniform2iv) }],
    [gl.INT_VEC3, { n: 3, k: "i", fn: f(gl.uniform3iv) }],
    [gl.INT_VEC4, { n: 4, k: "i", fn: f(gl.uniform4iv) }],
    [gl.BOOL, { n: 1, k: "i", fn: f(gl.uniform1iv) }],
    [gl.BOOL_VEC2, { n: 2, k: "i", fn: f(gl.uniform2iv) }],
    [gl.BOOL_VEC3, { n: 3, k: "i", fn: f(gl.uniform3iv) }],
    [gl.BOOL_VEC4, { n: 4, k: "i", fn: f(gl.uniform4iv) }],
    [gl.UNSIGNED_INT, { n: 1, k: "u", fn: f(gl.uniform1uiv) }],
    [gl.UNSIGNED_INT_VEC2, { n: 2, k: "u", fn: f(gl.uniform2uiv) }],
    [gl.UNSIGNED_INT_VEC3, { n: 3, k: "u", fn: f(gl.uniform3uiv) }],
    [gl.UNSIGNED_INT_VEC4, { n: 4, k: "u", fn: f(gl.uniform4uiv) }],
    [gl.FLOAT_MAT2, { n: 4, k: "f", fn: m(gl.uniformMatrix2fv) }],
    [gl.FLOAT_MAT3, { n: 9, k: "f", fn: m(gl.uniformMatrix3fv) }],
    [gl.FLOAT_MAT4, { n: 16, k: "f", fn: m(gl.uniformMatrix4fv) }],
  ]);
  const ARRAYS = { f: Float32Array, i: Int32Array, u: Uint32Array };

  let vs, vao, dummy2D, dummy3D, samplers;

  function initGL() {
    vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, VERT);
    gl.compileShader(vs);
    vao = gl.createVertexArray();
    dummy2D = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, dummy2D);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    dummy3D = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, dummy3D);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    samplers = new Map();
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
  }
  initGL();

  function sampler(filter, wrapMode) {
    const key = `${filter}|${wrapMode}`;
    let s = samplers.get(key);
    if (s) return s;
    s = gl.createSampler();
    const w = { repeat: gl.REPEAT, mirror: gl.MIRRORED_REPEAT }[wrapMode] ?? gl.CLAMP_TO_EDGE;
    gl.samplerParameteri(s, gl.TEXTURE_WRAP_S, w);
    gl.samplerParameteri(s, gl.TEXTURE_WRAP_T, w);
    gl.samplerParameteri(s, gl.TEXTURE_WRAP_R, w);
    const min = { nearest: gl.NEAREST, mipmap: gl.LINEAR_MIPMAP_LINEAR }[filter] ?? gl.LINEAR;
    const mag = filter === "nearest" ? gl.NEAREST : gl.LINEAR;
    gl.samplerParameteri(s, gl.TEXTURE_MIN_FILTER, min);
    gl.samplerParameteri(s, gl.TEXTURE_MAG_FILTER, mag);
    samplers.set(key, s);
    return s;
  }

  // ---------------------------------------------------------------- textures
  function syncTextures() {
    const specs = model.get("_textures") || {};
    const errors = [];
    for (const name of Object.keys(S.textures)) {
      if (!(name in specs) || specs[name].role !== "texture") {
        gl.deleteTexture(S.textures[name].tex);
        delete S.textures[name];
      }
    }
    for (const name of Object.keys(S.inits)) {
      if (!(name in specs) || specs[name].role !== "init") delete S.inits[name];
    }
    for (const [name, t] of Object.entries(specs)) {
      if (t.role === "init") {
        if (S.inits[name]?.hash !== t.hash) {
          S.inits[name] = { shape: t.shape, components: t.components, hash: t.hash, data: typedArray(bytesOf(t.data), "float32") };
        }
        continue;
      }
      if (S.textures[name]?.hash === t.hash) continue;
      try {
        uploadTexture(name, t);
      } catch (e) {
        errors.push({ source: `texture ${name}`, line: 0, message: String(e.message || e) });
      }
    }
    return errors;
  }

  function uploadTexture(name, t) {
    const data = typedArray(bytesOf(t.data), t.dtype);
    const [internal, format] = FORMATS[t.dtype][t.components];
    const type = t.dtype === "uint8" ? gl.UNSIGNED_BYTE : gl.FLOAT;
    const old = S.textures[name];
    if (old) gl.deleteTexture(old.tex);
    const tex = gl.createTexture();
    const mipmaps = t.dtype === "uint8";
    if (t.dims === 3) {
      const [d, h, w] = t.shape;
      const max = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
      if (Math.max(w, h, d) > max) throw new Error(`3D texture ${w}x${h}x${d} exceeds MAX_3D_TEXTURE_SIZE=${max}`);
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texImage3D(gl.TEXTURE_3D, 0, internal, w, h, d, 0, format, type, data);
      if (mipmaps) gl.generateMipmap(gl.TEXTURE_3D);
      S.textures[name] = { tex, target: gl.TEXTURE_3D, w, h, d, dtype: t.dtype, hash: t.hash, mipmaps };
    } else {
      const [h, w] = t.shape;
      const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      if (Math.max(w, h) > max) throw new Error(`texture ${w}x${h} exceeds MAX_TEXTURE_SIZE=${max}`);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
      if (mipmaps) gl.generateMipmap(gl.TEXTURE_2D);
      S.textures[name] = { tex, target: gl.TEXTURE_2D, w, h, d: 1, dtype: t.dtype, hash: t.hash, mipmaps };
    }
  }

  // ---------------------------------------------------------------- programs
  function channelTypes(channels) {
    const n = Math.max(4, channels?.length || 0);
    const types = [];
    for (let i = 0; i < n; i++) {
      const ch = channels?.[i];
      const is3D = ch?.kind === "texture" && S.textures[ch.name]?.target === gl.TEXTURE_3D;
      types.push(is3D ? "sampler3D" : "sampler2D");
    }
    return types;
  }

  function buildFragment(name, pass, chanTypes) {
    const ctx = {
      includes: model.get("_includes") || {},
      seen: new Set(),
      nextId: 3,
      srcNames: { [SRC_CODE]: passLabel(name), [SRC_COMMON]: "Common", [SRC_HEADER]: "header" },
    };
    const common = expandIncludes(model.get("_common_code") || "", SRC_COMMON, ctx);
    const code = expandIncludes(pass.code || "", SRC_CODE, ctx);
    const userText = `${common}\n${code}`;
    const decls = [];
    for (const [uname, u] of Object.entries(S.uniforms)) {
      if (!/^[A-Za-z_]\w*$/.test(uname)) continue;
      if (new RegExp(`\\buniform\\b[^;]*\\b${uname}\\b`).test(userText)) continue;
      decls.push(`uniform ${u.decl};`);
    }
    const n = chanTypes.length;
    const header = [
      "#version 300 es",
      `#line 1 ${SRC_HEADER}`,
      "precision highp float;",
      "precision highp int;",
      "precision highp sampler2D;",
      "precision highp sampler3D;",
      "uniform vec3 iResolution;",
      "uniform float iTime;",
      "uniform float iTimeDelta;",
      "uniform float iFrameRate;",
      "uniform int iFrame;",
      "uniform vec4 iMouse;",
      "uniform vec4 iHover;",
      "uniform vec4 iDate;",
      "uniform float iSampleRate;",
      `uniform vec3 iChannelResolution[${n}];`,
      `uniform float iChannelTime[${n}];`,
      ...chanTypes.map((t, i) => `uniform ${t} iChannel${i};`),
      ...decls,
      "#define HW_PERFORMANCE 1",
      "out vec4 st_FragColor;",
    ].join("\n");
    const main = name === "image"
      ? "void main() { vec4 c = vec4(0.0, 0.0, 0.0, 1.0); mainImage(c, gl_FragCoord.xy); st_FragColor = vec4(c.rgb, 1.0); }"
      : "void main() { vec4 c = vec4(0.0); mainImage(c, gl_FragCoord.xy); st_FragColor = c; }";
    const src = `${header}\n#line 1 ${SRC_COMMON}\n${common}\n#line 1 ${SRC_CODE}\n${code}\n${main}\n`;
    return { src, srcNames: ctx.srcNames };
  }

  function compile(src) {
    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, src);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(fs);
      return { error: log || "compilation failed" };
    }
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      return { error: log || "link failed" };
    }
    const uniforms = new Map();
    const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(prog, i);
      const uname = info.name.replace(/\[0\]$/, "");
      uniforms.set(uname, { loc: gl.getUniformLocation(prog, uname), type: info.type, size: info.size });
    }
    gl.useProgram(prog);
    for (let i = 0; i < 16; i++) {
      const u = uniforms.get(`iChannel${i}`);
      if (u) gl.uniform1i(u.loc, i);
    }
    return { prog, uniforms };
  }

  function typesSignature(passes) {
    return Object.entries(passes).map(([k, p]) => `${k}:${channelTypes(p.channels).join(",")}`).join(";");
  }

  function buildPipeline() {
    const passes = model.get("_passes") || {};
    const bufferNames = Object.keys(passes).filter((k) => k !== "image");
    const errors = {};
    const programs = {};
    for (const name of [...bufferNames, "image"]) {
      const pass = passes[name] || { code: "", channels: [] };
      const chanTypes = channelTypes(pass.channels);
      const { src, srcNames } = buildFragment(name, pass, chanTypes);
      const r = compile(src);
      if (r.error) errors[name] = formatLog(r.error, srcNames);
      else programs[name] = { ...r, pass, chanTypes };
    }
    if (Object.keys(errors).length) {
      for (const p of Object.values(programs)) gl.deleteProgram(p.prog);
      return { pipeline: null, errors };
    }
    return { pipeline: { programs, bufferNames, passes, sig: typesSignature(passes) }, errors };
  }

  function destroyPipeline(p) {
    if (!p) return;
    for (const prog of Object.values(p.programs)) gl.deleteProgram(prog.prog);
  }

  // ---------------------------------------------------------------- buffers
  function initPixels(init, w, h) {
    const n = w * h;
    const c = init.components;
    const src = init.data;
    let out = new Float32Array(n * 4);
    if (c === 4) out.set(src.subarray(0, n * 4));
    else for (let i = 0; i < n; i++) for (let k = 0; k < c; k++) out[i * 4 + k] = src[i * c + k];
    if (!BUF.float) {
      const u8 = new Uint8Array(n * 4);
      for (let i = 0; i < n * 4; i++) u8[i] = Math.max(0, Math.min(255, Math.round(out[i] * 255)));
      out = u8;
    }
    return out;
  }

  function freeBuffer(b) {
    for (const t of b.tex) gl.deleteTexture(t);
    for (const fb of b.fbo) gl.deleteFramebuffer(fb);
  }

  function bufferGeometry(cfg) {
    const init = cfg.init ? S.inits[cfg.init] : null;
    let w, h;
    if (cfg.size) [w, h] = cfg.size;
    else if (init) [h, w] = init.shape;
    else [w, h] = [canvas.width, canvas.height];
    return { w, h, init, key: `${w}x${h}|${cfg.init || ""}|${init?.hash || ""}` };
  }

  function makeBuffer({ w, h, init, key }) {
    const pixels = init ? initPixels(init, w, h) : null;
    const tex = [];
    const fbo = [];
    for (let i = 0; i < 2; i++) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, BUF.internal, w, h, 0, gl.RGBA, BUF.float ? gl.FLOAT : gl.UNSIGNED_BYTE, pixels);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      tex.push(t);
      fbo.push(fb);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { w, h, key, tex, fbo, read: 0 };
  }

  function syncBuffers(reset) {
    const p = S.pipeline;
    const names = p ? p.bufferNames : [];
    for (const name of Object.keys(S.buffers)) {
      if (!names.includes(name)) {
        freeBuffer(S.buffers[name]);
        delete S.buffers[name];
      }
    }
    for (const name of names) {
      const geo = bufferGeometry(p.passes[name]);
      const cur = S.buffers[name];
      if (cur && cur.key === geo.key && !reset) continue;
      if (cur) freeBuffer(cur);
      S.buffers[name] = makeBuffer(geo);
    }
  }

  // ---------------------------------------------------------------- drawing
  function applyUniform(info, v) {
    const s = SETTERS.get(info.type);
    if (!s || v == null) return;
    const n = s.n * info.size;
    const arr = new ARRAYS[s.k](n);
    const len = Math.min(n, v.length);
    if (s.k === "f") for (let i = 0; i < len; i++) arr[i] = v[i];
    else for (let i = 0; i < len; i++) arr[i] = Math.round(v[i]);
    s.fn(info.loc, arr);
  }

  function bindChannel(unit, spec, type) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    let tex = null;
    let res = [0, 0, 0];
    let unfilterable = false;
    let mips = false;
    if (spec?.kind === "buffer") {
      const b = S.buffers[spec.name];
      if (b) {
        tex = b.tex[b.read];
        res = [b.w, b.h, 1];
        unfilterable = !BUF.filterable;
      }
    } else if (spec?.kind === "texture") {
      const t = S.textures[spec.name];
      if (t) {
        tex = t.tex;
        res = [t.w, t.h, t.d];
        unfilterable = t.dtype === "float32" && !ext.fl;
        mips = t.mipmaps;
      }
    }
    const target = type === "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;
    gl.bindTexture(target, tex ?? (target === gl.TEXTURE_3D ? dummy3D : dummy2D));
    let filter = spec?.filter || "linear";
    if (filter === "mipmap" && !mips) filter = "linear";
    if (unfilterable) filter = "nearest";
    gl.bindSampler(unit, sampler(filter, spec?.wrap || "clamp"));
    return res;
  }

  function drawPass(prog, fbo, w, h) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, w, h);
    gl.useProgram(prog.prog);
    const U = prog.uniforms;
    const n = prog.chanTypes.length;
    const chanRes = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) chanRes.set(bindChannel(i, prog.pass.channels?.[i], prog.chanTypes[i]), i * 3);
    const now = new Date();
    const secs = now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds() + now.getMilliseconds() / 1000;
    const set = (name, v) => {
      const info = U.get(name);
      if (info) applyUniform(info, v);
    };
    set("iResolution", [w, h, 1]);
    set("iTime", [S.time]);
    set("iTimeDelta", [S.dt]);
    set("iFrameRate", [S.fps]);
    set("iFrame", [S.frame]);
    // Mouse coordinates live in each pass's own pixel space (matters for fixed-size buffers).
    const sx = w / canvas.width;
    const sy = h / canvas.height;
    const [mx, my, mz, mw] = S.mouse;
    set("iMouse", [mx * sx, my * sy, mz * sx, mw * sy]);
    set("iHover", [S.hover[0] * sx, S.hover[1] * sy, S.hover[2], S.hover[3]]);
    set("iDate", [now.getFullYear(), now.getMonth(), now.getDate(), secs]);
    set("iSampleRate", [44100]);
    set("iChannelResolution", chanRes);
    set("iChannelTime", new Float32Array(n).fill(S.time));
    for (const [uname, u] of Object.entries(S.uniforms)) set(uname, u.value);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function drawBuffers() {
    const p = S.pipeline;
    for (const name of p.bufferNames) {
      const b = S.buffers[name];
      const prog = p.programs[name];
      if (!b || !prog) continue;
      const iterations = Math.max(1, p.passes[name].iterations || 1);
      for (let k = 0; k < iterations; k++) {
        const write = 1 - b.read;
        drawPass(prog, b.fbo[write], b.w, b.h);
        b.read = write;
      }
    }
  }

  function drawImage() {
    const p = S.pipeline;
    if (!p) return false;
    drawPass(p.programs.image, null, canvas.width, canvas.height);
    return true;
  }

  // ---------------------------------------------------------------- snapshots
  function readSnapshot(req) {
    const p = S.pipeline;
    const target = req.target || p.bufferNames[0] || "image";
    let w, h, data, dtype;
    if (target === "image") {
      w = canvas.width;
      h = canvas.height;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      data = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
      dtype = "uint8";
    } else {
      const b = S.buffers[target];
      if (!b) throw new Error(`snapshot: no buffer named "${target}"`);
      w = b.w;
      h = b.h;
      gl.bindFramebuffer(gl.FRAMEBUFFER, b.fbo[b.read]);
      if (BUF.float) {
        data = new Float32Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, data);
        dtype = "float32";
      } else {
        data = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
        dtype = "uint8";
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    model.set("_snapshot_data", new DataView(data.buffer));
    model.set("_snapshot_meta", { id: req.id, target, width: w, height: h, dtype, time: S.time, frame: S.frame });
    model.save_changes();
    snapBtn.classList.add("mst-flash");
    setTimeout(() => snapBtn.classList.remove("mst-flash"), 300);
  }

  function processSnapshots(drewImage) {
    if (!S.pendingSnapshots.length || !S.pipeline) return;
    const reqs = S.pendingSnapshots.splice(0);
    if (!isPrimary()) return;
    for (const req of reqs) {
      try {
        const target = req.target || S.pipeline.bufferNames[0] || "image";
        if (target === "image" && !drewImage) drewImage = drawImage();
        readSnapshot(req);
      } catch (e) {
        console.error("[shadertoy]", e);
        showErrors({ snapshot: [{ source: "snapshot", line: 0, message: String(e.message || e) }] });
      }
    }
  }

  // ---------------------------------------------------------------- errors
  function showErrors(errors) {
    const lines = [];
    for (const [pass, list] of Object.entries(errors)) {
      for (const e of list) {
        const where = e.line ? `${e.source}:${e.line}` : e.source || passLabel(pass);
        lines.push(`${where}  ${e.message}`);
      }
    }
    overlay.textContent = lines.join("\n");
    overlay.style.display = lines.length ? "block" : "none";
  }

  function publishErrors(errors) {
    showErrors(errors);
    if (!isPrimary()) return;
    const current = JSON.stringify(model.get("errors") || {});
    if (JSON.stringify(errors) !== current) {
      model.set("errors", errors);
      model.save_changes();
    }
  }

  // ---------------------------------------------------------------- update loop
  function uniformsSignature(u) {
    return Object.entries(u).map(([k, v]) => `${k}:${v.decl}`).sort().join(";");
  }

  function processDirty() {
    const d = S.dirty;
    let texErrors = [];
    if (d.textures) {
      d.textures = false;
      texErrors = syncTextures();
      if (!S.pipeline || S.pipeline.sig !== typesSignature(model.get("_passes") || {})) d.program = true;
      d.buffers = true;
    }
    if (d.program) {
      d.program = false;
      const r = buildPipeline();
      if (r.pipeline) {
        destroyPipeline(S.pipeline);
        S.pipeline = r.pipeline;
        d.buffers = true;
      }
      const errors = { ...r.errors };
      if (texErrors.length) errors.textures = texErrors;
      publishErrors(errors);
      S.needsDraw = true;
    } else if (texErrors.length) {
      publishErrors({ textures: texErrors });
    }
    if (d.buffers || d.reset) {
      syncBuffers(d.reset);
      d.buffers = false;
      d.reset = false;
      S.needsDraw = true;
    }
  }

  function updateStats(now) {
    if (now - S.lastStats < 250) return;
    S.lastStats = now;
    stats.textContent = `${S.time.toFixed(2)}s  ${S.fps.toFixed(0).padStart(3)} fps  ${canvas.width}×${canvas.height}  #${S.frame}`;
  }

  let raf = 0;
  function loop(now) {
    raf = requestAnimationFrame(loop);
    if (S.lost) return;
    processDirty();
    const step = S.stepRequested;
    S.stepRequested = false;
    const animate = S.pipeline && ((S.playing && S.visible) || step);
    let drewImage = false;
    if (animate) {
      const t = now / 1000;
      const dt = step || S.lastNow == null ? 1 / 60 : Math.min(Math.max(t - S.lastNow, 1e-4), 0.25);
      S.lastNow = step ? null : t;
      if (S.started) S.time += dt;
      S.started = true;
      S.dt = dt;
      S.fps = S.fps * 0.9 + (1 / dt) * 0.1;
      drawBuffers();
      drewImage = drawImage();
      S.frame++;
      if (S.mouse[3] > 0) S.mouse[3] = -S.mouse[3];
    } else {
      S.lastNow = null;
      if (S.needsDraw && S.visible) drewImage = drawImage();
    }
    if (drewImage) S.needsDraw = false;
    processSnapshots(drewImage);
    updateStats(now);
  }

  function setPlaying(playing, sync) {
    S.playing = playing;
    playBtn.innerHTML = ICONS[playing ? "pause" : "play"];
    if (sync && model.get("paused") !== !playing) {
      model.set("paused", !playing);
      model.save_changes();
    }
  }

  function doReset() {
    S.time = 0;
    S.frame = 0;
    S.started = false;
    S.lastNow = null;
    S.dirty.reset = true;
    S.needsDraw = true;
  }

  // ---------------------------------------------------------------- events
  const toPx = (e) => {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (r.bottom - e.clientY) * (canvas.height / r.height)];
  };
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const [x, y] = toPx(e);
    S.mouse = [x, y, x, y];
    S.mouseDown = true;
    S.needsDraw = true;
  });
  canvas.addEventListener("pointermove", (e) => {
    const [x, y] = toPx(e);
    S.hover = [x, y, 1, S.mouseDown ? 1 : 0];
    if (S.mouseDown) {
      S.mouse[0] = x;
      S.mouse[1] = y;
    }
    S.needsDraw = true;
  });
  const release = () => {
    if (!S.mouseDown) return;
    S.mouseDown = false;
    S.mouse[2] = -Math.abs(S.mouse[2]);
    S.mouse[3] = -Math.abs(S.mouse[3]);
    S.hover[3] = 0;
    S.needsDraw = true;
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointerleave", () => {
    S.hover[2] = 0;
    S.needsDraw = true;
  });
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    S.lost = true;
  });
  canvas.addEventListener("webglcontextrestored", () => {
    S.lost = false;
    S.textures = {};
    S.buffers = {};
    S.pipeline = null;
    initGL();
    S.dirty = { textures: true, program: true, buffers: true, reset: true };
  });

  function resize() {
    wrap.style.height = document.fullscreenElement === root ? "" : `${model.get("height")}px`;
    const scale = Math.min(window.devicePixelRatio || 1, 2) * (model.get("resolution_scale") || 1);
    const w = Math.max(1, Math.round(wrap.clientWidth * scale));
    const h = Math.max(1, Math.round(wrap.clientHeight * scale));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      S.dirty.buffers = true;
      S.needsDraw = true;
    }
  }
  const ro = new ResizeObserver(resize);
  ro.observe(wrap);
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) S.visible = e.isIntersecting;
    S.needsDraw = true;
  });
  io.observe(root);
  document.addEventListener("fullscreenchange", resize);

  const handlers = {
    "change:_textures": () => { S.dirty.textures = true; },
    "change:_passes": () => { S.dirty.program = true; },
    "change:_common_code": () => { S.dirty.program = true; },
    "change:_includes": () => { S.dirty.program = true; },
    "change:_uniforms": () => {
      S.uniforms = model.get("_uniforms") || {};
      const sig = uniformsSignature(S.uniforms);
      if (sig !== S.uniformSig) {
        S.uniformSig = sig;
        S.dirty.program = true;
      }
      S.needsDraw = true;
    },
    "change:_reset": () => doReset(),
    "change:paused": () => setPlaying(!model.get("paused"), false),
    "change:height": () => resize(),
    "change:resolution_scale": () => resize(),
    "change:_snapshot_request": () => {
      const req = model.get("_snapshot_request");
      if (req && req.id != null) S.pendingSnapshots.push(req);
      S.needsDraw = true;
    },
  };
  for (const [evt, fn] of Object.entries(handlers)) model.on(evt, fn);
  S.uniformSig = uniformsSignature(S.uniforms);
  setPlaying(S.playing, false);
  resize();
  raf = requestAnimationFrame(loop);

  return () => {
    cancelAnimationFrame(raf);
    ro.disconnect();
    io.disconnect();
    document.removeEventListener("fullscreenchange", resize);
    for (const [evt, fn] of Object.entries(handlers)) model.off(evt, fn);
    const views = VIEWS.get(model);
    if (views) views.splice(views.indexOf(viewToken), 1);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    root.remove();
  };
}

export default { render };
"""

    _css = r"""
.mst-root {
  --mst-bg: #0d1017;
  --mst-bar: #141923;
  --mst-line: #262d3d;
  --mst-fg: #c9d1e0;
  position: relative;
  box-sizing: border-box;
  width: 100%;
  overflow: hidden;
  border: 1px solid var(--mst-line);
  border-radius: 8px;
  background: var(--mst-bg);
  color: var(--mst-fg);
  font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.mst-wrap { position: relative; width: 100%; }
.mst-canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
.mst-errors {
  display: none;
  position: absolute;
  inset: 0 0 auto 0;
  max-height: 100%;
  overflow: auto;
  margin: 0;
  padding: 8px 12px;
  background: rgba(48, 6, 10, 0.88);
  color: #ffc9c9;
  font: inherit;
  white-space: pre-wrap;
}
.mst-bar {
  display: flex;
  align-items: center;
  gap: 2px;
  min-width: 0;
  padding: 3px 6px;
  border-top: 1px solid var(--mst-line);
  background: var(--mst-bar);
}
.mst-btn {
  display: inline-flex;
  flex: none;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 24px;
  padding: 0;
  border: 1px solid transparent;
  border-radius: 4px;
  background: transparent;
  color: inherit;
  cursor: pointer;
}
.mst-btn:hover { border-color: var(--mst-line); background: #1f2635; }
.mst-btn svg { width: 13px; height: 13px; fill: currentColor; }
.mst-btn.mst-flash { background: #2f6f4f; }
.mst-stats {
  min-width: 0;
  margin-left: 6px;
  overflow: hidden;
  opacity: 0.75;
  font-variant-numeric: tabular-nums;
  text-overflow: ellipsis;
  white-space: pre;
}
.mst-spacer { flex: 1; }
.mst-root:fullscreen { display: flex; flex-direction: column; border: 0; border-radius: 0; }
.mst-root:fullscreen .mst-wrap { flex: 1; }
"""

    INCLUDES = {"colormaps": r"""
// Colormaps sampled from matplotlib (33 knots, piecewise linear).
// Usage: viridis(t), magma(t), ..., or cmap(t, id) with id in 0..6:
//   0 viridis, 1 magma, 2 inferno, 3 plasma, 4 cividis, 5 turbo, 6 coolwarm
const vec3 CM_VIRIDIS[33] = vec3[33](vec3(0.267,0.005,0.329),vec3(0.277,0.050,0.376),vec3(0.282,0.095,0.417),vec3(0.283,0.136,0.453),vec3(0.279,0.175,0.483),vec3(0.271,0.214,0.507),vec3(0.259,0.252,0.525),vec3(0.245,0.288,0.537),vec3(0.230,0.322,0.546),vec3(0.214,0.356,0.551),vec3(0.199,0.388,0.555),vec3(0.186,0.419,0.557),vec3(0.173,0.449,0.558),vec3(0.161,0.479,0.558),vec3(0.149,0.508,0.557),vec3(0.138,0.537,0.555),vec3(0.128,0.567,0.551),vec3(0.121,0.596,0.544),vec3(0.121,0.626,0.533),vec3(0.132,0.655,0.520),vec3(0.158,0.684,0.502),vec3(0.197,0.712,0.479),vec3(0.246,0.739,0.452),vec3(0.304,0.765,0.420),vec3(0.369,0.789,0.383),vec3(0.440,0.811,0.341),vec3(0.516,0.831,0.294),vec3(0.596,0.849,0.243),vec3(0.678,0.864,0.190),vec3(0.762,0.876,0.137),vec3(0.846,0.887,0.100),vec3(0.926,0.897,0.104),vec3(0.993,0.906,0.144));
vec3 viridis(float t) { t = clamp(t, 0.0, 1.0) * 32.0; int i = min(int(t), 31); return mix(CM_VIRIDIS[i], CM_VIRIDIS[i + 1], t - float(i)); }
const vec3 CM_MAGMA[33] = vec3[33](vec3(0.001,0.000,0.014),vec3(0.014,0.012,0.069),vec3(0.040,0.031,0.134),vec3(0.074,0.052,0.203),vec3(0.113,0.065,0.277),vec3(0.159,0.068,0.353),vec3(0.212,0.062,0.419),vec3(0.265,0.060,0.462),vec3(0.317,0.072,0.485),vec3(0.366,0.090,0.498),vec3(0.415,0.110,0.505),vec3(0.464,0.130,0.508),vec3(0.513,0.148,0.508),vec3(0.563,0.165,0.505),vec3(0.614,0.182,0.499),vec3(0.665,0.198,0.489),vec3(0.716,0.215,0.475),vec3(0.767,0.234,0.458),vec3(0.817,0.256,0.436),vec3(0.863,0.284,0.412),vec3(0.904,0.320,0.388),vec3(0.937,0.365,0.369),vec3(0.961,0.418,0.360),vec3(0.977,0.476,0.364),vec3(0.987,0.536,0.382),vec3(0.993,0.595,0.410),vec3(0.996,0.654,0.446),vec3(0.997,0.712,0.488),vec3(0.997,0.770,0.535),vec3(0.995,0.827,0.586),vec3(0.992,0.884,0.640),vec3(0.989,0.941,0.698),vec3(0.987,0.991,0.750));
vec3 magma(float t) { t = clamp(t, 0.0, 1.0) * 32.0; int i = min(int(t), 31); return mix(CM_MAGMA[i], CM_MAGMA[i + 1], t - float(i)); }
const vec3 CM_INFERNO[33] = vec3[33](vec3(0.001,0.000,0.014),vec3(0.014,0.011,0.072),vec3(0.042,0.028,0.141),vec3(0.082,0.043,0.215),vec3(0.129,0.047,0.291),vec3(0.183,0.040,0.355),vec3(0.238,0.037,0.396),vec3(0.291,0.046,0.419),vec3(0.342,0.062,0.429),vec3(0.391,0.081,0.433),vec3(0.441,0.099,0.432),vec3(0.491,0.117,0.426),vec3(0.541,0.135,0.415),vec3(0.591,0.153,0.400),vec3(0.640,0.171,0.381),vec3(0.689,0.192,0.358),vec3(0.736,0.216,0.330),vec3(0.781,0.243,0.300),vec3(0.822,0.275,0.266),vec3(0.861,0.312,0.231),vec3(0.894,0.353,0.194),vec3(0.923,0.399,0.155),vec3(0.947,0.449,0.115),vec3(0.965,0.502,0.074),vec3(0.978,0.558,0.035),vec3(0.986,0.616,0.026),vec3(0.988,0.675,0.065),vec3(0.984,0.736,0.130),vec3(0.975,0.798,0.206),vec3(0.961,0.859,0.298),vec3(0.948,0.917,0.411),vec3(0.955,0.966,0.540),vec3(0.988,0.998,0.645));
vec3 inferno(float t) { t = clamp(t, 0.0, 1.0) * 32.0; int i = min(int(t), 31); return mix(CM_INFERNO[i], CM_INFERNO[i + 1], t - float(i)); }
const vec3 CM_PLASMA[33] = vec3[33](vec3(0.050,0.030,0.528),vec3(0.132,0.022,0.563),vec3(0.193,0.018,0.590),vec3(0.248,0.014,0.613),vec3(0.300,0.010,0.632),vec3(0.350,0.004,0.646),vec3(0.399,0.001,0.656),vec3(0.448,0.002,0.660),vec3(0.495,0.012,0.658),vec3(0.541,0.035,0.649),vec3(0.584,0.069,0.633),vec3(0.626,0.103,0.611),vec3(0.665,0.139,0.586),vec3(0.702,0.174,0.557),vec3(0.736,0.209,0.528),vec3(0.768,0.245,0.498),vec3(0.798,0.280,0.470),vec3(0.827,0.316,0.441),vec3(0.853,0.352,0.414),vec3(0.878,0.388,0.387),vec3(0.902,0.425,0.360),vec3(0.923,0.463,0.333),vec3(0.943,0.503,0.306),vec3(0.959,0.543,0.279),vec3(0.973,0.586,0.252),vec3(0.984,0.630,0.225),vec3(0.991,0.675,0.198),vec3(0.994,0.723,0.174),vec3(0.993,0.772,0.155),vec3(0.987,0.822,0.144),vec3(0.974,0.875,0.144),vec3(0.957,0.928,0.152),vec3(0.940,0.975,0.131));
vec3 plasma(float t) { t = clamp(t, 0.0, 1.0) * 32.0; int i = min(int(t), 31); return mix(CM_PLASMA[i], CM_PLASMA[i + 1], t - float(i)); }
const vec3 CM_CIVIDIS[33] = vec3[33](vec3(0.000,0.135,0.305),vec3(0.000,0.158,0.358),vec3(0.000,0.179,0.415),vec3(0.018,0.199,0.441),vec3(0.103,0.220,0.436),vec3(0.154,0.242,0.430),vec3(0.195,0.264,0.426),vec3(0.231,0.286,0.423),vec3(0.264,0.308,0.423),vec3(0.295,0.330,0.424),vec3(0.324,0.351,0.426),vec3(0.353,0.373,0.430),vec3(0.381,0.395,0.436),vec3(0.408,0.417,0.443),vec3(0.435,0.440,0.451),vec3(0.462,0.462,0.462),vec3(0.489,0.485,0.471),vec3(0.518,0.508,0.473),vec3(0.548,0.532,0.472),vec3(0.578,0.556,0.469),vec3(0.609,0.580,0.464),vec3(0.640,0.604,0.457),vec3(0.672,0.629,0.448),vec3(0.704,0.655,0.437),vec3(0.736,0.681,0.424),vec3(0.769,0.707,0.409),vec3(0.803,0.734,0.390),vec3(0.836,0.761,0.369),vec3(0.871,0.790,0.343),vec3(0.906,0.818,0.313),vec3(0.941,0.848,0.276),vec3(0.978,0.877,0.228),vec3(0.996,0.909,0.218));
vec3 cividis(float t) { t = clamp(t, 0.0, 1.0) * 32.0; int i = min(int(t), 31); return mix(CM_CIVIDIS[i], CM_CIVIDIS[i + 1], t - float(i)); }
const vec3 CM_TURBO[33] = vec3[33](vec3(0.190,0.072,0.232),vec3(0.225,0.164,0.451),vec3(0.251,0.252,0.634),vec3(0.268,0.338,0.780),vec3(0.276,0.421,0.891),vec3(0.275,0.501,0.966),vec3(0.259,0.580,0.999),vec3(0.214,0.659,0.980),vec3(0.158,0.736,0.923),vec3(0.112,0.806,0.845),vec3(0.093,0.866,0.762),vec3(0.120,0.912,0.687),vec3(0.197,0.949,0.595),vec3(0.305,0.977,0.490),vec3(0.428,0.994,0.386),vec3(0.547,0.999,0.296),vec3(0.644,0.990,0.234),vec3(0.726,0.965,0.206),vec3(0.805,0.925,0.205),vec3(0.875,0.873,0.216),vec3(0.933,0.812,0.227),vec3(0.973,0.747,0.225),vec3(0.993,0.674,0.203),vec3(0.996,0.587,0.169),vec3(0.984,0.493,0.128),vec3(0.958,0.400,0.088),vec3(0.921,0.315,0.055),vec3(0.874,0.245,0.033),vec3(0.816,0.185,0.018),vec3(0.746,0.131,0.009),vec3(0.664,0.084,0.004),vec3(0.571,0.045,0.005),vec3(0.480,0.016,0.011));
vec3 turbo(float t) { t = clamp(t, 0.0, 1.0) * 32.0; int i = min(int(t), 31); return mix(CM_TURBO[i], CM_TURBO[i + 1], t - float(i)); }
const vec3 CM_COOLWARM[33] = vec3[33](vec3(0.230,0.299,0.754),vec3(0.266,0.353,0.802),vec3(0.304,0.407,0.845),vec3(0.343,0.459,0.884),vec3(0.384,0.510,0.918),vec3(0.425,0.559,0.946),vec3(0.468,0.606,0.969),vec3(0.511,0.649,0.985),vec3(0.554,0.690,0.996),vec3(0.598,0.727,1.000),vec3(0.641,0.761,0.998),vec3(0.683,0.790,0.990),vec3(0.724,0.815,0.976),vec3(0.763,0.835,0.956),vec3(0.801,0.850,0.930),vec3(0.835,0.861,0.899),vec3(0.867,0.864,0.863),vec3(0.900,0.848,0.818),vec3(0.926,0.826,0.771),vec3(0.946,0.799,0.723),vec3(0.960,0.767,0.674),vec3(0.968,0.731,0.625),vec3(0.970,0.690,0.575),vec3(0.966,0.646,0.526),vec3(0.957,0.598,0.477),vec3(0.942,0.546,0.430),vec3(0.921,0.491,0.383),vec3(0.896,0.433,0.339),vec3(0.865,0.371,0.296),vec3(0.830,0.305,0.255),vec3(0.791,0.231,0.216),vec3(0.747,0.140,0.180),vec3(0.706,0.016,0.150));
vec3 coolwarm(float t) { t = clamp(t, 0.0, 1.0) * 32.0; int i = min(int(t), 31); return mix(CM_COOLWARM[i], CM_COOLWARM[i + 1], t - float(i)); }
vec3 cmap(float t, int id) {
    if (id == 0) return viridis(t);
    else if (id == 1) return magma(t);
    else if (id == 2) return inferno(t);
    else if (id == 3) return plasma(t);
    else if (id == 4) return cividis(t);
    else if (id == 5) return turbo(t);
    else if (id == 6) return coolwarm(t);
    return vec3(clamp(t, 0.0, 1.0));
}
"""}

    DEFAULT_IMAGE = r"""void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0, 2, 4));
    fragColor = vec4(col, 1.0);
}
"""

    BUILTINS = frozenset(
        "iResolution iTime iTimeDelta iFrameRate iFrame iMouse iHover iDate iSampleRate "
        "iChannelResolution iChannelTime".split()
    ) | {f"iChannel{i}" for i in range(16)}

    # Synced with the browser. Underscored traits are managed through the Python API below.
    _passes = traitlets.Dict().tag(sync=True)
    _common_code = traitlets.Unicode("").tag(sync=True)
    _includes = traitlets.Dict().tag(sync=True)
    _textures = traitlets.Dict().tag(sync=True)
    _uniforms = traitlets.Dict().tag(sync=True)
    _reset = traitlets.Int(0).tag(sync=True)
    _snapshot_request = traitlets.Dict().tag(sync=True)
    _snapshot_meta = traitlets.Dict().tag(sync=True)
    _snapshot_data = traitlets.Bytes(b"").tag(sync=True, from_json=lambda v, _w: b"" if v is None else bytes(v))

    height = traitlets.Int(400).tag(sync=True)
    resolution_scale = traitlets.Float(1.0).tag(sync=True)
    paused = traitlets.Bool(False).tag(sync=True)
    snapshot_target = traitlets.Unicode("").tag(sync=True)
    errors = traitlets.Dict().tag(sync=True)

    def __init__(self, image: str | None = None, *, channels=(), buffers=None, common: str = "",
                 uniforms=None, includes=None, **kwargs):
        cfg = {
            "image": self.DEFAULT_IMAGE if image is None else image,
            "channels": list(channels),
            "buffers": dict(buffers or {}),
            "common": common,
            "uniforms": dict(uniforms or {}),
        }
        state = {**self._scene_state(cfg), **self._uniform_state(cfg["uniforms"])}
        super().__init__(_includes={**self.INCLUDES, **(includes or {})}, **state, **kwargs)
        self._cfg = cfg

    # -- scene ---------------------------------------------------------------------------
    @property
    def image(self) -> str:
        return self._cfg["image"]

    @image.setter
    def image(self, code: str) -> None:
        self.update(image=code)

    @property
    def channels(self) -> list:
        return list(self._cfg["channels"])

    @channels.setter
    def channels(self, channels) -> None:
        self.update(channels=channels)

    @property
    def buffers(self) -> dict:
        return dict(self._cfg["buffers"])

    @buffers.setter
    def buffers(self, buffers) -> None:
        self.update(buffers=buffers)

    @property
    def common(self) -> str:
        return self._cfg["common"]

    @common.setter
    def common(self, code: str) -> None:
        self.update(common=code)

    @property
    def uniforms(self) -> dict:
        return dict(self._cfg["uniforms"])

    @uniforms.setter
    def uniforms(self, values) -> None:
        self.update(uniforms=values)

    def set_uniforms(self, **values) -> None:
        """Merge uniform values into the current ones."""
        self.update(uniforms={**self._cfg["uniforms"], **values})

    def update(self, *, image=None, channels=None, buffers=None, common=None, uniforms=None) -> None:
        """Change several parts of the scene at once (sent to the browser as one message)."""
        cfg = dict(self._cfg)
        if image is not None:
            cfg["image"] = image
        if channels is not None:
            cfg["channels"] = list(channels)
        if buffers is not None:
            cfg["buffers"] = dict(buffers)
        if common is not None:
            cfg["common"] = common
        if uniforms is not None:
            cfg["uniforms"] = dict(uniforms)
        state = {}
        if any(v is not None for v in (image, channels, buffers, common)):
            state.update(self._scene_state(cfg))
        if uniforms is not None:
            state.update(self._uniform_state(cfg["uniforms"]))
        self._cfg = cfg
        with self.hold_sync():
            for name, value in state.items():
                setattr(self, name, value)

    def reset(self) -> None:
        """Restart time and frame count, and re-initialise all buffers."""
        self._reset += 1

    # -- read-back -----------------------------------------------------------------------
    def request_snapshot(self, target: str = "") -> None:
        """Ask the browser to read a buffer (or "image") back; it arrives in `.snapshot`."""
        self._snapshot_request = {"id": int(self._snapshot_request.get("id", 0)) + 1, "target": target}

    @property
    def snapshot(self) -> np.ndarray | None:
        """Latest snapshot as an (H, W, 4) array (float32 for buffers, uint8 for "image").

        Row 0 is the bottom row, consistent with arrays passed in.
        """
        meta = self._snapshot_meta
        if not meta or not self._snapshot_data:
            return None
        dtype = np.uint8 if meta["dtype"] == "uint8" else np.float32
        return np.frombuffer(self._snapshot_data, dtype=dtype).reshape(meta["height"], meta["width"], 4)

    @property
    def snapshot_info(self) -> dict:
        """Metadata of the latest snapshot: target, width, height, dtype, time, frame."""
        return dict(self._snapshot_meta)

    # -- serialization -------------------------------------------------------------------
    @staticmethod
    def _check_choice(value: str, choices: tuple, what: str) -> None:
        if value not in choices:
            raise ValueError(f"{what} must be one of {choices}, got {value!r}")

    @staticmethod
    def _texture_payload(ch: "Channel") -> dict:
        a = np.asarray(ch.source)
        if a.dtype == np.bool_:
            a = a.astype(np.float32)
        volume = ch.volume if ch.volume is not None else (a.ndim == 4 or (a.ndim == 3 and a.shape[-1] > 4))
        a = a[..., None] if a.ndim == (3 if volume else 2) else a
        if a.ndim != (4 if volume else 3) or not 1 <= a.shape[-1] <= 4:
            raise ValueError(
                f"array of shape {np.shape(ch.source)} is not a texture: expected (H, W[, C<=4]) "
                "or, for volumes, (D, H, W[, C<=4])"
            )
        if ch.origin == "upper":
            a = np.flip(a, axis=-3)
        dtype = "uint8" if a.dtype == np.uint8 else "float32"
        data = np.ascontiguousarray(a, dtype=np.uint8 if dtype == "uint8" else np.float32).tobytes()
        return {
            "role": "texture",
            "dims": 3 if volume else 2,
            "shape": [int(s) for s in a.shape[:-1]],
            "components": int(a.shape[-1]),
            "dtype": dtype,
            "hash": hashlib.blake2b(data, digest_size=12).hexdigest(),
            "data": data,
        }

    @staticmethod
    def _init_payload(buf: "Buffer", name: str) -> dict:
        a = np.asarray(buf.init, dtype=np.float32)
        a = a[..., None] if a.ndim == 2 else a
        if a.ndim != 3 or not 1 <= a.shape[-1] <= 4:
            raise ValueError(f"Buffer {name}: init must be (H, W) or (H, W, C<=4), got {np.shape(buf.init)}")
        if buf.size is not None and tuple(buf.size) != (a.shape[1], a.shape[0]):
            raise ValueError(f"Buffer {name}: size {tuple(buf.size)} != init (width, height) {(a.shape[1], a.shape[0])}")
        data = np.ascontiguousarray(a).tobytes()
        return {
            "role": "init",
            "dims": 2,
            "shape": [int(a.shape[0]), int(a.shape[1])],
            "components": int(a.shape[-1]),
            "dtype": "float32",
            "hash": hashlib.blake2b(data, digest_size=12).hexdigest(),
            "data": data,
        }

    @classmethod
    def _scene_state(cls, cfg: dict) -> dict:
        buffers = {name: (b if isinstance(b, Buffer) else Buffer(b)) for name, b in cfg["buffers"].items()}
        for name in buffers:
            if name == "image" or not re.fullmatch(r"[A-Za-z_]\w*", name):
                raise ValueError(f"invalid buffer name {name!r}")
        textures: dict[str, dict] = {}
        seen: dict[tuple, str] = {}

        def spec(src, where: str) -> dict:
            if src is None:
                return {"kind": "none"}
            ch = src if isinstance(src, Channel) else Channel(src)
            cls._check_choice(ch.filter, ("linear", "nearest", "mipmap"), "filter")
            cls._check_choice(ch.wrap, ("clamp", "repeat", "mirror"), "wrap")
            cls._check_choice(ch.origin, ("lower", "upper"), "origin")
            if isinstance(ch.source, str):
                if ch.source not in buffers:
                    raise ValueError(f"{where}: unknown buffer {ch.source!r} (buffers: {list(buffers)})")
                return {"kind": "buffer", "name": ch.source, "filter": ch.filter, "wrap": ch.wrap}
            key = (id(ch.source), ch.volume, ch.origin)
            if key not in seen:
                seen[key] = f"t{len(seen)}"
                textures[seen[key]] = cls._texture_payload(ch)
            return {"kind": "texture", "name": seen[key], "filter": ch.filter, "wrap": ch.wrap}

        def channel_specs(channels, where: str) -> list:
            channels = list(channels)
            if len(channels) > 8:
                raise ValueError(f"{where}: at most 8 channels, got {len(channels)}")
            return [spec(c, where) for c in channels]

        passes = {}
        for name, buf in buffers.items():
            init = None
            if buf.init is not None:
                init = f"init:{name}"
                textures[init] = cls._init_payload(buf, name)
            passes[name] = {
                "code": buf.code,
                "channels": channel_specs(buf.channels, f"Buffer {name}"),
                "size": [int(s) for s in buf.size] if buf.size is not None else None,
                "init": init,
                "iterations": max(1, int(buf.iterations)),
            }
        passes["image"] = {"code": cfg["image"], "channels": channel_specs(cfg["channels"], "Image")}
        return {"_passes": passes, "_textures": textures, "_common_code": cfg["common"]}

    @classmethod
    def _uniform_state(cls, uniforms: dict) -> dict:
        out = {}
        for name, value in uniforms.items():
            if not re.fullmatch(r"[A-Za-z_]\w*", name) or name in cls.BUILTINS:
                raise ValueError(f"invalid uniform name {name!r}")
            out[name] = cls._uniform_payload(name, value)
        return {"_uniforms": out}

    @staticmethod
    def _uniform_payload(name: str, v) -> dict:
        if isinstance(v, (bool, np.bool_)):
            return {"decl": f"bool {name}", "value": [float(v)]}
        if isinstance(v, (int, np.integer)):
            return {"decl": f"int {name}", "value": [float(v)]}
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0:
            return {"decl": f"float {name}", "value": [float(a)]}
        if a.ndim == 1 and a.size in (2, 3, 4):
            return {"decl": f"vec{a.size} {name}", "value": a.tolist()}
        if a.ndim == 1:
            return {"decl": f"float {name}[{a.size}]", "value": a.tolist()}
        if a.ndim == 2 and a.shape[0] == a.shape[1] and a.shape[0] in (2, 3, 4):
            # GLSL matrices are column-major; send M so that `M * v` matches NumPy's `M @ v`.
            return {"decl": f"mat{a.shape[0]} {name}", "value": a.T.ravel().tolist()}
        if a.ndim == 2 and a.shape[1] in (2, 3, 4):
            return {"decl": f"vec{a.shape[1]} {name}[{a.shape[0]}]", "value": a.ravel().tolist()}
        raise ValueError(f"uniform {name!r}: unsupported value of shape {a.shape}")


if __name__ == "__main__":
    app.run()
