"""Tests for the Python side of the ShaderToy widget (serialization, validation, read-back)."""

import shutil
import subprocess

import numpy as np
import pytest

from shadertoy import Buffer, Channel, ShaderToy, gray_scott_seed, orbital_volume, radial_spectrum


def textures(w, role="texture"):
    return {k: v for k, v in w._textures.items() if v["role"] == role}


def test_2d_float_texture_keeps_row_order_and_casts_to_float32():
    field = np.arange(12, dtype=np.float64).reshape(3, 4)
    w = ShaderToy(channels=[field])
    (tex,) = textures(w).values()
    assert (tex["dims"], tex["shape"], tex["components"], tex["dtype"]) == (2, [3, 4], 1, "float32")
    assert np.array_equal(np.frombuffer(tex["data"], np.float32).reshape(3, 4), field)
    assert w._passes["image"]["channels"][0]["kind"] == "texture"


def test_origin_upper_flips_rows():
    img = np.arange(6, dtype=np.uint8).reshape(3, 2)
    w = ShaderToy(channels=[Channel(img, origin="upper")])
    (tex,) = textures(w).values()
    assert tex["dtype"] == "uint8"
    assert np.array_equal(np.frombuffer(tex["data"], np.uint8).reshape(3, 2), img[::-1])


@pytest.mark.parametrize(
    ("shape", "volume", "dims", "tex_shape", "components"),
    [
        ((8, 8, 3), None, 2, [8, 8], 3),  # RGB image
        ((8, 8, 8), None, 3, [8, 8, 8], 1),  # last axis > 4 -> volume
        ((4, 4, 4), True, 3, [4, 4, 4], 1),  # forced volume
        ((4, 5, 6, 2), None, 3, [4, 5, 6], 2),  # 4-D -> volume with 2 components
    ],
)
def test_texture_shape_inference(shape, volume, dims, tex_shape, components):
    w = ShaderToy(channels=[Channel(np.zeros(shape, np.float32), volume=volume)])
    (tex,) = textures(w).values()
    assert (tex["dims"], tex["shape"], tex["components"]) == (dims, tex_shape, components)


def test_same_array_in_two_slots_is_uploaded_once():
    a = np.zeros((4, 4), np.float32)
    w = ShaderToy(channels=[a, a], buffers={"A": Buffer("", channels=[a])})
    assert len(textures(w)) == 1


def test_buffer_init_and_size():
    init = np.ones((5, 7, 2), np.float32)
    w = ShaderToy(channels=["A"], buffers={"A": Buffer("", channels=["A"], init=init, iterations=3)})
    a = w._passes["A"]
    assert a["init"] == "init:A" and a["iterations"] == 3 and a["size"] is None
    init_tex = w._textures["init:A"]
    assert (init_tex["role"], init_tex["shape"], init_tex["components"]) == ("init", [5, 7], 2)
    with pytest.raises(ValueError, match="size"):
        ShaderToy(buffers={"A": Buffer("", init=init, size=(5, 7))})


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="unknown buffer"):
        ShaderToy(channels=["B"])
    with pytest.raises(ValueError, match="not a texture"):
        ShaderToy(channels=[np.zeros((4, 4, 9, 9, 9))])
    with pytest.raises(ValueError, match="filter"):
        ShaderToy(channels=[Channel(np.zeros((2, 2)), filter="cubic")])
    with pytest.raises(ValueError, match="invalid uniform"):
        ShaderToy(uniforms={"iTime": 1.0})


def test_uniform_declarations():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = ShaderToy(uniforms={"a": 1, "b": 0.5, "c": True, "v": (1, 2, 3), "m": m, "arr": np.arange(6), "pts": np.zeros((5, 2)), "f": np.float32(2)})
    decls = {k: v["decl"] for k, v in w._uniforms.items()}
    assert decls == {
        "a": "int a",
        "b": "float b",
        "c": "bool c",
        "v": "vec3 v",
        "m": "mat2 m",
        "arr": "float arr[6]",
        "pts": "vec2 pts[5]",
        "f": "float f",
    }
    assert w._uniforms["m"]["value"] == [1.0, 3.0, 2.0, 4.0]  # column-major


def test_update_only_resends_what_changed():
    w = ShaderToy(channels=[np.zeros((4, 4))], uniforms={"u": 1.0})
    tex_before = w._textures
    w.set_uniforms(u=2.0, v=3.0)
    assert w._textures is tex_before
    assert w.uniforms == {"u": 2.0, "v": 3.0}
    w.image = "void mainImage(out vec4 c, in vec2 p) { c = vec4(1); }"
    assert w._passes["image"]["code"].startswith("void mainImage")


def test_snapshot_round_trip_from_browser_state():
    w = ShaderToy()
    assert w.snapshot is None
    w.request_snapshot("A")
    w.request_snapshot("A")
    assert w._snapshot_request == {"id": 2, "target": "A"}
    data = np.arange(2 * 3 * 4, dtype=np.float32)
    # This is how the browser's reply arrives (binary buffers come in as memoryviews).
    w.set_state({
        "_snapshot_data": memoryview(data.tobytes()),
        "_snapshot_meta": {"id": 2, "target": "A", "width": 3, "height": 2, "dtype": "float32", "time": 1.5, "frame": 90},
    })
    assert w.snapshot.shape == (2, 3, 4)
    assert np.array_equal(w.snapshot.ravel(), data)
    assert w.snapshot_info["frame"] == 90


def test_example_data_generators():
    seed = gray_scott_seed("text")
    assert seed.shape == (256, 640, 2) and seed.dtype == np.float32
    assert (seed[..., 1] > 0.2).any()
    vol = orbital_volume("2p (z)", n=16)
    assert vol.shape == (16, 16, 16, 2) and np.isclose(vol[..., 0].max(), 1.0)
    power, freqs, radial = radial_spectrum(np.sin(np.arange(64) * 2 * np.pi / 8)[None, :].repeat(32, 0))
    assert power.shape == (32, 64)
    assert abs(1 / freqs[np.argmax(radial)] - 8) < 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_widget_javascript_parses(tmp_path):
    path = tmp_path / "widget.mjs"
    path.write_text(ShaderToy._esm)
    subprocess.run(["node", "--check", str(path)], check=True)
