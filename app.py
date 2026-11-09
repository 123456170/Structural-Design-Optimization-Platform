"""
Structural Design Optimization Platform (SDOP)

A runnable Streamlit application for density-based SIMP topology optimization.

Modules implemented inside this single file:
- Geometry input manager
- Material database
- Load and boundary-condition manager
- Finite element mesh generator
- Structural analysis engine
- Topology optimization engine
- Geometry reconstruction engine
- Result visualization system
- CAD/export engine
- Live demo video generator
- Benchmark validation
- Automated Pytest-compatible tests

Important:
This tool is for research, education, concept generation, and numerical exploration.
It does NOT certify any design for real-world structural use. Any physical implementation
requires independent engineering verification, code verification, validation, mesh convergence,
manufacturing review, safety factors, fatigue/buckling/dynamic assessment, and appropriate
regulatory or professional oversight.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shutil
import struct
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import torch
    from torch.nn.functional import conv2d
    from torch.nn.functional import pad as tpad
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import trimesh
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import gmsh
    HAS_GMSH = True
except Exception:
    HAS_GMSH = False

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
X_MIN = 1e-3
VIDEO_FPS = 8
VIDEO_SIZE = (640, 480)

UI_PRESETS = [
    "Live Demo Cantilever",
    "Cantilever",
    "MBB Bridge",
    "Tower",
    "Custom",
]


# ==============================================================================
# Material database
# ==============================================================================

@dataclass
class Material:
    name: str
    E: float
    nu: float
    rho: float
    yield_strength: Optional[float] = None
    notes: str = ""


MATERIALS: Dict[str, Material] = {
    "Aluminum 6061": Material(
        name="Aluminum 6061",
        E=68900.0,
        nu=0.33,
        rho=2700.0,
        yield_strength=276.0,
        notes="Isotropic approximation for concept studies.",
    ),
    "Steel A36": Material(
        name="Steel A36",
        E=200000.0,
        nu=0.30,
        rho=7850.0,
        yield_strength=250.0,
        notes="Common structural steel approximation.",
    ),
    "Titanium Ti-6Al-4V": Material(
        name="Titanium Ti-6Al-4V",
        E=110000.0,
        nu=0.34,
        rho=4430.0,
        yield_strength=880.0,
        notes="Aerospace-grade titanium approximation.",
    ),
    "ABS Polymer": Material(
        name="ABS Polymer",
        E=2300.0,
        nu=0.35,
        rho=1040.0,
        yield_strength=40.0,
        notes="Generic thermoplastic approximation.",
    ),
    "Nylon 6/6": Material(
        name="Nylon 6/6",
        E=2800.0,
        nu=0.39,
        rho=1140.0,
        yield_strength=70.0,
        notes="Generic engineering polymer approximation.",
    ),
    "Concrete C30/37": Material(
        name="Concrete C30/37",
        E=33000.0,
        nu=0.20,
        rho=2400.0,
        yield_strength=None,
        notes="Linear elastic approximation only; concrete is not ductile.",
    ),
}


# ==============================================================================
# Design specification and benchmark presets
# ==============================================================================

@dataclass
class DesignSpec:
    name: str = "Custom"
    nx: int = 48
    ny: int = 16
    length: float = 120.0
    height: float = 40.0
    thickness: float = 10.0

    volume_fraction: float = 0.35
    penal: float = 3.0
    filter_radius: float = 1.6
    iterations: int = 20
    move_limit: float = 0.2
    oc_eta: float = 0.5

    E0: float = 68900.0
    Emin: float = 1e-6
    nu: float = 0.33
    rho: float = 2700.0
    material_name: str = "Aluminum 6061"

    fixed_edges: List[str] = field(default_factory=lambda: ["left"])
    point_constraints: List[Tuple[float, float, bool, bool]] = field(default_factory=list)
    loads: List[Tuple[float, float, float, float]] = field(
        default_factory=lambda: [(120.0, 20.0, 0.0, -1000.0)]
    )

    max_permitted_displacement: float = 5.0
    threshold: float = 0.5
    seed: int = 0

    tol_compliance: float = 1e-4
    tol_design: float = 1e-3


def auto_demo_spec() -> DesignSpec:
    return DesignSpec(
        name="Live Demo Cantilever",
        nx=48,
        ny=16,
        length=120.0,
        height=40.0,
        thickness=10.0,
        volume_fraction=0.35,
        penal=3.0,
        filter_radius=1.6,
        iterations=20,
        move_limit=0.2,
        oc_eta=0.5,
        E0=MATERIALS["Aluminum 6061"].E,
        Emin=1e-6,
        nu=MATERIALS["Aluminum 6061"].nu,
        rho=MATERIALS["Aluminum 6061"].rho,
        material_name="Aluminum 6061",
        fixed_edges=["left"],
        point_constraints=[],
        loads=[(120.0, 20.0, 0.0, -1000.0)],
        max_permitted_displacement=5.0,
        threshold=0.5,
        seed=0,
    )


def cantilever_spec() -> DesignSpec:
    s = auto_demo_spec()
    return replace(
        s,
        name="Cantilever",
        nx=60,
        ny=20,
        iterations=30,
    )


def mbb_spec() -> DesignSpec:
    return DesignSpec(
        name="MBB Bridge",
        nx=70,
        ny=25,
        length=160.0,
        height=40.0,
        thickness=10.0,
        volume_fraction=0.40,
        penal=3.0,
        filter_radius=1.6,
        iterations=30,
        move_limit=0.2,
        oc_eta=0.5,
        E0=MATERIALS["Aluminum 6061"].E,
        Emin=1e-6,
        nu=MATERIALS["Aluminum 6061"].nu,
        rho=MATERIALS["Aluminum 6061"].rho,
        material_name="Aluminum 6061",
        fixed_edges=[],
        point_constraints=[
            (0.0, 0.0, True, True),
            (160.0, 0.0, False, True),
        ],
        loads=[(80.0, 40.0, 0.0, -1000.0)],
        max_permitted_displacement=3.0,
        threshold=0.5,
        seed=0,
    )


def tower_spec() -> DesignSpec:
    return DesignSpec(
        name="Tower",
        nx=40,
        ny=60,
        length=40.0,
        height=120.0,
        thickness=10.0,
        volume_fraction=0.25,
        penal=3.0,
        filter_radius=1.6,
        iterations=30,
        move_limit=0.2,
        oc_eta=0.5,
        E0=MATERIALS["Steel A36"].E,
        Emin=1e-6,
        nu=MATERIALS["Steel A36"].nu,
        rho=MATERIALS["Steel A36"].rho,
        material_name="Steel A36",
        fixed_edges=["bottom"],
        point_constraints=[],
        loads=[(20.0, 120.0, 0.0, -800.0)],
        max_permitted_displacement=4.0,
        threshold=0.5,
        seed=0,
    )


def custom_spec() -> DesignSpec:
    s = auto_demo_spec()
    return replace(
        s,
        name="Custom",
        nx=50,
        ny=25,
        iterations=25,
    )


PRESETS: Dict[str, Callable[[], DesignSpec]] = {
    "Live Demo Cantilever": auto_demo_spec,
    "Cantilever": cantilever_spec,
    "MBB Bridge": mbb_spec,
    "Tower": tower_spec,
    "Custom": custom_spec,
}


# ==============================================================================
# Utilities
# ==============================================================================

def get_output_dir() -> str:
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        probe = os.path.join(OUTPUT_DIR, ".sdop_write_probe")
        with open(probe, "wb") as fh:
            fh.write(b"ok")
        os.remove(probe)
        return OUTPUT_DIR
    except Exception:
        return tempfile.mkdtemp(prefix="sdop_outputs_")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, np.int32, np.int64, int)):
        return int(obj)
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    return obj


def design_spec_from_dict(d: Dict[str, Any]) -> DesignSpec:
    valid = set(DesignSpec.__dataclass_fields__.keys())
    dd = {k: v for k, v in dict(d).items() if k in valid}
    dd["fixed_edges"] = list(dd.get("fixed_edges", ["left"]))
    dd["point_constraints"] = [tuple(x) for x in dd.get("point_constraints", [])]
    dd["loads"] = [tuple(x) for x in dd.get("loads", [])]
    return DesignSpec(**dd)


# ==============================================================================
# Structured finite element mesh generator
# ==============================================================================

@dataclass
class StructuredMesh:
    nx: int
    ny: int
    hx: float
    hy: float
    node_coords: np.ndarray
    elem_dofs: np.ndarray
    ndof: int

    @classmethod
    def create(cls, nx: int, ny: int, length: float, height: float) -> "StructuredMesh":
        nx = int(max(2, nx))
        ny = int(max(2, ny))
        hx = float(length) / float(nx)
        hy = float(height) / float(ny)

        nnx = nx + 1
        nny = ny + 1
        node_count = nnx * nny
        coords = np.zeros((node_count, 2), dtype=float)

        for i in range(nny):
            for j in range(nnx):
                nid = i * nnx + j
                coords[nid, 0] = j * hx
                coords[nid, 1] = i * hy

        elem_count = nx * ny
        elem_dofs = np.zeros((elem_count, 8), dtype=int)
        e = 0
        for i in range(ny):
            for j in range(nx):
                n0 = i * nnx + j
                n1 = n0 + 1
                n2 = n0 + nnx + 1
                n3 = n0 + nnx
                elem_dofs[e] = [
                    2 * n0, 2 * n0 + 1,
                    2 * n1, 2 * n1 + 1,
                    2 * n2, 2 * n2 + 1,
                    2 * n3, 2 * n3 + 1,
                ]
                e += 1

        return cls(
            nx=nx,
            ny=ny,
            hx=hx,
            hy=hy,
            node_coords=coords,
            elem_dofs=elem_dofs,
            ndof=2 * node_count,
        )


def q4_plane_stress_base_stiffness(hx: float, hy: float, nu: float) -> np.ndarray:
    """
    Base element stiffness matrix for a 4-node bilinear quadrilateral
    with unit Young's modulus under plane stress.
    """
    nu = float(np.clip(nu, 0.0, 0.49))
    D = np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=float,
    ) / (1.0 - nu * nu)

    gp = 1.0 / math.sqrt(3.0)
    gauss_points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    ref_nodes = np.array([[0.0, 0.0], [hx, 0.0], [hx, hy], [0.0, hy]], dtype=float)

    ke = np.zeros((8, 8), dtype=float)

    for xi, eta in gauss_points:
        dNdxi = np.array(
            [
                -(1.0 - eta),
                (1.0 - eta),
                (1.0 + eta),
                -(1.0 + eta),
            ],
            dtype=float,
        ) / 4.0

        dNdeta = np.array(
            [
                -(1.0 - xi),
                -(1.0 + xi),
                (1.0 + xi),
                (1.0 - xi),
            ],
            dtype=float,
        ) / 4.0

        J = np.array(
            [
                [ref_nodes[:, 0] @ dNdxi, ref_nodes[:, 1] @ dNdxi],
                [ref_nodes[:, 0] @ dNdeta, ref_nodes[:, 1] @ dNdeta],
            ],
            dtype=float,
        )

        detJ = float(np.linalg.det(J))
        if detJ <= 0.0:
            detJ = abs(detJ) if abs(detJ) > 0.0 else 1.0

        invJ = np.linalg.inv(J)

        dNdx = invJ[0, 0] * dNdxi + invJ[0, 1] * dNdeta
        dNdy = invJ[1, 0] * dNdxi + invJ[1, 1] * dNdeta

        B = np.zeros((3, 8), dtype=float)
        for a in range(4):
            B[0, 2 * a] = dNdx[a]
            B[1, 2 * a + 1] = dNdy[a]
            B[2, 2 * a] = dNdy[a]
            B[2, 2 * a + 1] = dNdx[a]

        ke += detJ * (B.T @ D @ B)

    ke = 0.5 * (ke + ke.T)
    return ke


# ==============================================================================
# Structural analysis engine
# ==============================================================================

class FEEngine:
    def __init__(self, mesh: StructuredMesh, nu: float):
        self.mesh = mesh
        self.nu = float(nu)
        self.base_ke = q4_plane_stress_base_stiffness(mesh.hx, mesh.hy, self.nu)
        self.ndof = mesh.ndof

    def assemble(self, E_flat: np.ndarray) -> sp.csr_matrix:
        E_flat = np.asarray(E_flat, dtype=float).ravel()
        elem_dofs = self.mesh.elem_dofs
        nel = E_flat.size

        rows = np.repeat(elem_dofs, 8, axis=1).ravel()
        cols = np.tile(elem_dofs, (1, 8)).ravel()
        data = (E_flat[:, None, None] * self.base_ke[None, :, :]).reshape(nel, 64).ravel()

        K = sp.coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
        return K

    def solve(
        self,
        E_flat: np.ndarray,
        fixed_dofs: np.ndarray,
        force: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        K = self.assemble(E_flat)

        free_mask = np.ones(self.ndof, dtype=bool)
        if fixed_dofs.size > 0:
            free_mask[fixed_dofs] = False
        free_dofs = np.where(free_mask)[0]

        u = np.zeros(self.ndof, dtype=float)

        if free_dofs.size == 0:
            return u, 0.0, np.zeros(self.mesh.elem_dofs.shape[0], dtype=float)

        Kff = K[free_dofs][:, free_dofs].tocsc()
        fff = np.asarray(force, dtype=float)[free_dofs]

        try:
            uf = spla.spsolve(Kff, fff)
        except Exception:
            uf = spla.lsqr(Kff, fff, atol=1e-10, btol=1e-10)[0]

        uf = np.nan_to_num(np.asarray(uf, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        u[free_dofs] = uf

        compliance = float(np.dot(force, u))
        if not np.isfinite(compliance):
            compliance = float("inf")

        U = u[self.mesh.elem_dofs]
        ce0 = np.einsum("ni,ij,nj->n", U, self.base_ke, U, optimize=True)
        ce0 = np.nan_to_num(ce0, nan=0.0, posinf=0.0, neginf=0.0)
        ce0 = np.maximum(ce0, 0.0)

        return u, compliance, ce0


# ==============================================================================
# Load and boundary-condition manager
# ==============================================================================

def nearest_node(mesh: StructuredMesh, x: float, y: float) -> int:
    j = int(round(float(x) / mesh.hx))
    i = int(round(float(y) / mesh.hy))
    j = int(np.clip(j, 0, mesh.nx))
    i = int(np.clip(i, 0, mesh.ny))
    return i * (mesh.nx + 1) + j


def build_fixed_dofs(
    mesh: StructuredMesh,
    fixed_edges: List[str],
    point_constraints: List[Tuple[float, float, bool, bool]],
) -> np.ndarray:
    fixed: List[int] = []

    for edge in fixed_edges:
        edge = str(edge).lower()
        if edge == "left":
            nodes = [i * (mesh.nx + 1) for i in range(mesh.ny + 1)]
        elif edge == "right":
            nodes = [i * (mesh.nx + 1) + mesh.nx for i in range(mesh.ny + 1)]
        elif edge == "bottom":
            nodes = list(range(mesh.nx + 1))
        elif edge == "top":
            nodes = [mesh.ny * (mesh.nx + 1) + j for j in range(mesh.nx + 1)]
        else:
            continue

        for n in nodes:
            fixed.extend([2 * n, 2 * n + 1])

    for x, y, fix_ux, fix_uy in point_constraints:
        n = nearest_node(mesh, x, y)
        if fix_ux:
            fixed.append(2 * n)
        if fix_uy:
            fixed.append(2 * n + 1)

    fixed = sorted(set(int(v) for v in fixed))
    return np.array(fixed, dtype=int)


def build_force_vector(mesh: StructuredMesh, loads: List[Tuple[float, float, float, float]]) -> np.ndarray:
    f = np.zeros(mesh.ndof, dtype=float)
    for x, y, fx, fy in loads:
        n = nearest_node(mesh, x, y)
        f[2 * n] += float(fx)
        f[2 * n + 1] += float(fy)
    return f


def nodal_displacement_magnitude(u: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    ux = u[0::2]
    uy = u[1::2]
    mag = np.sqrt(ux * ux + uy * uy)
    return mag.reshape(mesh.ny + 1, mesh.nx + 1)


# ==============================================================================
# Density filtering
# ==============================================================================

class DensityFilter:
    def __init__(self, radius: float):
        self.radius = float(radius)
        self.kernel = self._make_kernel()
        if HAS_TORCH:
            self.tkernel = torch.from_numpy(self.kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        else:
            self.tkernel = None

    def _make_kernel(self) -> np.ndarray:
        if self.radius <= 0.0:
            return np.array([[1.0]], dtype=float)

        r = max(1, int(math.floor(self.radius)))
        size = 2 * r + 1
        coords = np.arange(size, dtype=float) - r
        dx, dy = np.meshgrid(coords, coords, indexing="xy")
        dist = np.sqrt(dx * dx + dy * dy)
        w = np.maximum(0.0, self.radius - dist)

        if w.sum() <= 0.0:
            w[r, r] = 1.0

        w /= w.sum()
        return w.astype(float)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.ndim != 2:
            raise ValueError("Density filter expects a 2D array.")

        if self.radius <= 0.0 or self.kernel.size == 1:
            return a.copy()

        if HAS_TORCH and self.tkernel is not None:
            try:
                t = torch.from_numpy(a.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                p = self.kernel.shape[0] // 2
                tp = tpad(t, (p, p, p, p), mode="replicate")
                out = conv2d(tp, self.tkernel, stride=1, padding=0)
                return out.squeeze(0).squeeze(0).detach().cpu().numpy().astype(float)
            except Exception:
                pass

        return ndi.convolve(a, self.kernel, mode="nearest")


# ==============================================================================
# Optimality Criteria update for SIMP
# ==============================================================================

def oc_update(
    x: np.ndarray,
    sens: np.ndarray,
    spec: DesignSpec,
    filt: DensityFilter,
) -> np.ndarray:
    B = np.maximum(-np.nan_to_num(sens, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    lower = np.maximum(X_MIN, x - spec.move_limit)
    upper = np.minimum(1.0, x + spec.move_limit)

    def volume_of(xx: np.ndarray) -> float:
        return float(np.mean(filt(xx)))

    lo = 1e-12
    hi = 1e12

    for _ in range(35):
        lam = 0.5 * (lo + hi)
        ratio = np.divide(B, lam, out=np.zeros_like(B), where=lam > 0)
        ratio = np.clip(ratio, 0.0, 1e6)
        xt = np.clip(x * np.power(ratio, spec.oc_eta), lower, upper)

        if volume_of(xt) > spec.volume_fraction:
            lo = lam
        else:
            hi = lam

    lam = 0.5 * (lo + hi)
    ratio = np.divide(B, lam, out=np.zeros_like(B), where=lam > 0)
    ratio = np.clip(ratio, 0.0, 1e6)
    xt = np.clip(x * np.power(ratio, spec.oc_eta), lower, upper)
    return np.clip(xt, X_MIN, 1.0)


# ==============================================================================
# Live video generation
# ==============================================================================

def create_video_writer(path: str, fps: int = VIDEO_FPS, size: Tuple[int, int] = VIDEO_SIZE) -> Tuple[Optional[Any], Optional[str]]:
    if not HAS_CV2:
        return None, None

    out_dir = os.path.dirname(path) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = get_output_dir()
        path = os.path.join(out_dir, os.path.basename(path))

    fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc_mp4, float(fps), size)
    if writer.isOpened():
        return writer, path

    alt = os.path.splitext(path)[0] + ".avi"
    fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(alt, fourcc_avi, float(fps), size)
    if writer.isOpened():
        return writer, alt

    return None, None


def write_video_frame(
    writer: Any,
    xf: np.ndarray,
    iteration: int,
    compliance: float,
    volume: float,
    dx: float,
    spec: DesignSpec,
) -> None:
    if writer is None:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    im = ax.imshow(
        xf,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=[0.0, spec.length, 0.0, spec.height],
        aspect="auto",
    )
    ax.set_title(f"SIMP Topology Optimization | Iteration {iteration + 1}")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")

    txt = (
        f"Compliance: {compliance:.3g} N-mm\n"
        f"Volume fraction: {volume:.3f}\n"
        f"Max design change: {dx:.3g}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        fontsize=9,
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)

    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return

    img = cv2.resize(img, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
    writer.write(img)


# ==============================================================================
# Optimization result container
# ==============================================================================

@dataclass
class OptimizationResult:
    spec: Dict[str, Any]
    physical_density: np.ndarray
    design_density: np.ndarray
    history: pd.DataFrame
    compliance: float
    volume_fraction: float
    max_displacement: float
    runtime_s: float
    element_energy: np.ndarray
    nodal_displacement: np.ndarray
    converged: bool
    iterations_run: int
    mesh_info: Dict[str, Any]


# ==============================================================================
# Topology optimization engine
# ==============================================================================

def run_topology_optimization(
    spec: DesignSpec,
    record_video_path: Optional[str] = None,
    progress: Optional[Callable[[int, float, float, float], None]] = None,
) -> Tuple[OptimizationResult, Optional[str]]:
    spec = replace(spec)

    spec.nx = int(max(4, min(160, int(spec.nx))))
    spec.ny = int(max(4, min(80, int(spec.ny))))
    spec.iterations = int(max(1, min(200, int(spec.iterations))))

    spec.length = max(1.0, finite_float(spec.length, 120.0))
    spec.height = max(1.0, finite_float(spec.height, 40.0))
    spec.thickness = max(0.01, finite_float(spec.thickness, 10.0))

    spec.volume_fraction = float(np.clip(finite_float(spec.volume_fraction, 0.35), 0.05, 0.90))
    spec.penal = float(np.clip(finite_float(spec.penal, 3.0), 1.0, 5.0))
    spec.filter_radius = float(np.clip(finite_float(spec.filter_radius, 1.6), 0.0, 5.0))
    spec.move_limit = float(np.clip(finite_float(spec.move_limit, 0.2), 0.01, 0.50))
    spec.oc_eta = float(np.clip(finite_float(spec.oc_eta, 0.5), 0.05, 1.0))

    spec.E0 = max(1.0, finite_float(spec.E0, 68900.0))
    spec.Emin = max(1e-12, min(finite_float(spec.Emin, 1e-6), spec.E0 * 1e-8))
    spec.nu = float(np.clip(finite_float(spec.nu, 0.33), 0.0, 0.49))
    spec.rho = max(1.0, finite_float(spec.rho, 2700.0))
    spec.max_permitted_displacement = max(0.0, finite_float(spec.max_permitted_displacement, 5.0))
    spec.threshold = float(np.clip(finite_float(spec.threshold, 0.5), 0.05, 0.95))
    spec.seed = int(finite_float(spec.seed, 0.0))

    clean_point_constraints: List[Tuple[float, float, bool, bool]] = []
    for pc in spec.point_constraints:
        if len(pc) >= 4:
            x = finite_float(pc[0], 0.0)
            y = finite_float(pc[1], 0.0)
            fix_ux = bool(pc[2])
            fix_uy = bool(pc[3])
            clean_point_constraints.append((x, y, fix_ux, fix_uy))
    spec.point_constraints = clean_point_constraints

    clean_loads: List[Tuple[float, float, float, float]] = []
    for load in spec.loads:
        if len(load) >= 4:
            x = finite_float(load[0], spec.length * 0.5)
            y = finite_float(load[1], spec.height * 0.5)
            fx = finite_float(load[2], 0.0)
            fy = finite_float(load[3], -1000.0)
            clean_loads.append((x, y, fx, fy))
    if not clean_loads:
        clean_loads = [(spec.length, spec.height * 0.5, 0.0, -1000.0)]
    spec.loads = clean_loads

    mesh = StructuredMesh.create(spec.nx, spec.ny, spec.length, spec.height)
    fixed_dofs = build_fixed_dofs(mesh, spec.fixed_edges, spec.point_constraints)

    if fixed_dofs.size == 0:
        fixed_dofs = build_fixed_dofs(mesh, ["left"], [])

    if fixed_dofs.size == 0:
        fixed_dofs = np.array([0, 1], dtype=int)

    force = build_force_vector(mesh, spec.loads)
    fe = FEEngine(mesh, spec.nu)
    filt = DensityFilter(spec.filter_radius)

    rng = np.random.default_rng(spec.seed)
    x = np.full((spec.ny, spec.nx), spec.volume_fraction, dtype=float)
    x += 0.01 * rng.standard_normal(size=x.shape)
    x = np.clip(x, X_MIN, 1.0)

    writer, actual_video_path = (None, None)
    if record_video_path:
        writer, actual_video_path = create_video_writer(record_video_path)

    history_rows: List[Dict[str, Any]] = []
    old_compliance: Optional[float] = None
    converged = False
    start = time.perf_counter()

    for it in range(spec.iterations):
        xf = filt(x)
        xf = np.clip(xf, 0.0, 1.0)

        E = spec.Emin + np.power(xf, spec.penal) * (spec.E0 - spec.Emin)
        u, compliance, ce0 = fe.solve(E.ravel(), fixed_dofs, force)

        vol = float(np.mean(xf))
        disp = nodal_displacement_magnitude(u, mesh)
        max_disp = float(np.max(disp)) if disp.size else 0.0

        x_safe = np.maximum(xf, X_MIN)
        ce0_grid = np.asarray(ce0, dtype=float).reshape(xf.shape)
        sens_phys = -spec.penal * (E - spec.Emin) / x_safe * ce0_grid
        sens_phys = np.nan_to_num(sens_phys, nan=0.0, posinf=0.0, neginf=0.0)
        sens_design = filt(sens_phys)

        x_new = oc_update(x, sens_design, spec, filt)
        dx = float(np.max(np.abs(x_new - x)))
        runtime = time.perf_counter() - start

        history_rows.append(
            {
                "iteration": it,
                "compliance": compliance,
                "volume": vol,
                "max_design_change": dx,
                "max_displacement": max_disp,
                "runtime_s": runtime,
            }
        )

        if writer is not None:
            try:
                write_video_frame(writer, xf, it, compliance, vol, dx, spec)
            except Exception:
                pass

        if progress is not None:
            progress(it + 1, compliance, vol, dx)

        if (
            old_compliance is not None
            and np.isfinite(old_compliance)
            and np.isfinite(compliance)
            and abs(compliance - old_compliance) / max(abs(old_compliance), 1e-12) < spec.tol_compliance
            and dx < spec.tol_design
        ):
            converged = True
            x = x_new
            break

        x = x_new
        old_compliance = compliance

    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass

    xf = filt(x)
    xf = np.clip(xf, 0.0, 1.0)
    E = spec.Emin + np.power(xf, spec.penal) * (spec.E0 - spec.Emin)
    u, compliance, ce0 = fe.solve(E.ravel(), fixed_dofs, force)

    vol = float(np.mean(xf))
    disp = nodal_displacement_magnitude(u, mesh)
    max_disp = float(np.max(disp)) if disp.size else 0.0

    ce0_grid = np.asarray(ce0, dtype=float).reshape(xf.shape)
    element_energy = E * ce0_grid
    element_energy = np.nan_to_num(element_energy, nan=0.0, posinf=0.0, neginf=0.0)

    runtime = time.perf_counter() - start

    if not history_rows:
        history_rows.append(
            {
                "iteration": 0,
                "compliance": compliance,
                "volume": vol,
                "max_design_change": 0.0,
                "max_displacement": max_disp,
                "runtime_s": runtime,
            }
        )

    history = pd.DataFrame(history_rows)

    result = OptimizationResult(
        spec=asdict(spec),
        physical_density=xf,
        design_density=x,
        history=history,
        compliance=float(compliance),
        volume_fraction=float(vol),
        max_displacement=float(max_disp),
        runtime_s=float(runtime),
        element_energy=element_energy,
        nodal_displacement=disp,
        converged=bool(converged),
        iterations_run=int(len(history)),
        mesh_info={
            "nx": int(spec.nx),
            "ny": int(spec.ny),
            "hx": float(mesh.hx),
            "hy": float(mesh.hy),
            "node_count": int(mesh.node_coords.shape[0]),
            "element_count": int(mesh.elem_dofs.shape[0]),
            "dof_count": int(mesh.ndof),
        },
    )

    return result, actual_video_path


# ==============================================================================
# Geometry reconstruction engine
# ==============================================================================

def _surface_raw_mesh(
    density: np.ndarray,
    threshold: float,
    hx: float,
    hy: float,
    thickness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(density, dtype=float) >= float(threshold)
    ny, nx = mask.shape

    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    def add_quad(p1, p2, p3, p4):
        idx = len(verts)
        verts.extend([p1, p2, p3, p4])
        faces.append((idx, idx + 1, idx + 2))
        faces.append((idx, idx + 2, idx + 3))

    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                continue

            x0 = j * hx
            x1 = x0 + hx
            y0 = i * hy
            y1 = y0 + hy
            z0 = 0.0
            z1 = float(thickness)

            if j == nx - 1 or not mask[i, j + 1]:
                add_quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))

            if j == 0 or not mask[i, j - 1]:
                add_quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))

            if i == ny - 1 or not mask[i + 1, j]:
                add_quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))

            if i == 0 or not mask[i - 1, j]:
                add_quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))

            add_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
            add_quad((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))

    v = np.asarray(verts, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    return v, f


def build_surface_mesh(
    density: np.ndarray,
    threshold: float,
    hx: float,
    hy: float,
    thickness: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    verts, faces = _surface_raw_mesh(density, threshold, hx, hy, thickness)

    if faces.size == 0:
        return verts, faces, None

    if HAS_TRIMESH:
        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            mesh.fix_normals()
            if mesh.faces.size > 0:
                return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int), mesh
        except Exception:
            pass

    return verts, faces, None


def build_voxel_hex_mesh(
    density: np.ndarray,
    threshold: float,
    hx: float,
    hy: float,
    thickness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(density, dtype=float) >= float(threshold)
    ny, nx = mask.shape

    node_map: Dict[Tuple[float, float, float], int] = {}
    nodes: List[Tuple[float, float, float]] = []
    elems: List[List[int]] = []

    def get_node(x: float, y: float, z: float) -> int:
        key = (round(float(x), 8), round(float(y), 8), round(float(z), 8))
        idx = node_map.get(key)
        if idx is None:
            idx = len(nodes)
            node_map[key] = idx
            nodes.append(key)
        return idx

    z0 = 0.0
    z1 = float(thickness)

    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                continue

            x0 = j * hx
            x1 = x0 + hx
            y0 = i * hy
            y1 = y0 + hy

            n1 = get_node(x0, y0, z0)
            n2 = get_node(x1, y0, z0)
            n3 = get_node(x1, y1, z0)
            n4 = get_node(x0, y1, z0)
            n5 = get_node(x0, y0, z1)
            n6 = get_node(x1, y0, z1)
            n7 = get_node(x1, y1, z1)
            n8 = get_node(x0, y1, z1)

            elems.append([n1, n2, n3, n4, n5, n6, n7, n8])

    return np.asarray(nodes, dtype=float).reshape(-1, 3), np.asarray(elems, dtype=int).reshape(-1, 8)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)

    if a.size == 0 or b.size == 0:
        return 0.0

    zoom = (b.shape[0] / a.shape[0], b.shape[1] / a.shape[1])
    resized_full = ndi.zoom(a.astype(float), zoom, order=0, mode="nearest")

    resized = np.zeros(b.shape, dtype=bool)
    h = min(resized_full.shape[0], b.shape[0])
    w = min(resized_full.shape[1], b.shape[1])
    resized[:h, :w] = resized_full[:h, :w] > 0.5

    inter = np.logical_and(resized, b).sum()
    union = np.logical_or(resized, b).sum()

    if union == 0:
        return 0.0

    return float(inter) / float(union)


# ==============================================================================
# Export engine
# ==============================================================================

def density_csv_bytes(xf: np.ndarray) -> bytes:
    return pd.DataFrame(xf).to_csv(index=False).encode("utf-8")


def history_csv_bytes(history: pd.DataFrame) -> bytes:
    return history.to_csv(index=False).encode("utf-8")


def summary_json_bytes(result: OptimizationResult) -> bytes:
    spec = to_jsonable(result.spec)
    hist = to_jsonable(result.history.replace([np.inf, -np.inf], np.nan).to_dict(orient="records"))
    payload = {
        "spec": spec,
        "compliance_Nmm": float(result.compliance) if math.isfinite(result.compliance) else None,
        "volume_fraction": float(result.volume_fraction),
        "max_displacement_mm": float(result.max_displacement),
        "runtime_s": float(result.runtime_s),
        "iterations_run": int(result.iterations_run),
        "converged": bool(result.converged),
        "mesh_info": to_jsonable(result.mesh_info),
        "history": hist,
        "disclaimer": "Research/concept output only. Not certified for real-world structural use.",
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def stl_bytes_from_mesh(verts: np.ndarray, faces: np.ndarray, mesh: Optional[Any] = None) -> bytes:
    if mesh is not None and HAS_TRIMESH:
        try:
            data = mesh.export(file_type="stl")
            if isinstance(data, str):
                return data.encode("utf-8")
            return bytes(data)
        except Exception:
            pass

    buf = io.BytesIO()
    header = b"SDOP optimized structure".ljust(80, b"\0")
    buf.write(header)
    buf.write(struct.pack("<I", int(len(faces))))

    for f in faces:
        v0 = verts[int(f[0])]
        v1 = verts[int(f[1])]
        v2 = verts[int(f[2])]

        n = np.cross(v1 - v0, v2 - v0)
        norm = float(np.linalg.norm(n))
        if norm > 0.0:
            n = n / norm
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=float)

        buf.write(struct.pack("<3f", float(n[0]), float(n[1]), float(n[2])))
        for v in (v0, v1, v2):
            buf.write(struct.pack("<3f", float(v[0]), float(v[1]), float(v[2])))
        buf.write(struct.pack("<H", 0))

    return buf.getvalue()


def obj_bytes(verts: np.ndarray, faces: np.ndarray) -> bytes:
    lines = ["# SDOP optimized structure", "o optimized_structure"]

    for v in verts:
        lines.append(f"v {float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}")

    for f in faces:
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")

    return ("\n".join(lines) + "\n").encode("utf-8")


def gmsh_msh_bytes(verts: np.ndarray, faces: np.ndarray) -> bytes:
    s = io.StringIO()
    s.write("$MeshFormat\n")
    s.write("2.2 0 8\n")
    s.write("$EndMeshFormat\n")

    s.write("$Nodes\n")
    s.write(f"{len(verts)}\n")
    for i, v in enumerate(verts, start=1):
        s.write(f"{i} {float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}\n")
    s.write("$EndNodes\n")

    s.write("$Elements\n")
    s.write(f"{len(faces)}\n")
    for i, f in enumerate(faces, start=1):
        n1 = int(f[0]) + 1
        n2 = int(f[1]) + 1
        n3 = int(f[2]) + 1
        s.write(f"{i} 2 2 1 1 {n1} {n2} {n3}\n")
    s.write("$EndElements\n")

    return s.getvalue().encode("utf-8")


def calculix_inp_bytes(
    nodes: np.ndarray,
    elems: np.ndarray,
    spec: Dict[str, Any],
) -> bytes:
    lines: List[str] = []
    lines.append("*HEADING")
    lines.append("SDOP reconstructed voxel volume mesh for independent verification.")
    lines.append("Not a certified design.")
    lines.append("")

    if nodes.size > 0:
        lines.append("*NODE")
        for i, n in enumerate(nodes, start=1):
            lines.append(f"{i}, {float(n[0]):.8f}, {float(n[1]):.8f}, {float(n[2]):.8f}")
        lines.append("")

    if elems.size > 0:
        lines.append("*ELEMENT, TYPE=C3D8, ELSET=SOLID")
        for i, e in enumerate(elems, start=1):
            conn = ", ".join(str(int(v) + 1) for v in e)
            lines.append(f"{i}, {conn}")
        lines.append("")

    lines.append("*MATERIAL, NAME=OPTIMIZED_MATERIAL")
    lines.append("*ELASTIC")
    E = float(spec.get("E0", 68900.0))
    nu = float(spec.get("nu", 0.33))
    lines.append(f"{E:.8f}, {nu:.8f}")
    lines.append("")

    if elems.size > 0:
        lines.append("*SOLID SECTION, ELSET=SOLID, MATERIAL=OPTIMIZED_MATERIAL")
        lines.append("")

    return ("\n".join(lines) + "\n").encode("utf-8")


def freecad_macro_bytes() -> bytes:
    text = """# FreeCAD macro placeholder
# Place optimized_structure.stl beside this macro or adjust the path.
# This macro is informational; FreeCAD is not required for the Streamlit app.

import os
import FreeCAD
import Part

stl_path = "optimized_structure.stl"
if not os.path.exists(stl_path):
    stl_path = os.path.join(os.getcwd(), "optimized_structure.stl")

doc = FreeCAD.newDocument("OptimizedStructure")
shape = Part.read(stl_path)
Part.show(shape)
doc.recompute()
FreeCADGui.ActiveDocument.ActiveView.viewIsometric()
FreeCADGui.SendMsgToActiveView("ViewFit")
"""
    return text.encode("utf-8")


# ==============================================================================
# Visualization helpers
# ==============================================================================

def make_density_fig(xf: np.ndarray, spec_dict: Dict[str, Any], title: str = "Physical density field") -> go.Figure:
    nx = int(spec_dict.get("nx", xf.shape[1]))
    ny = int(spec_dict.get("ny", xf.shape[0]))
    length = float(spec_dict.get("length", nx))
    height = float(spec_dict.get("height", ny))

    hx = length / max(1, nx)
    hy = height / max(1, ny)

    xcenters = (np.arange(nx) + 0.5) * hx
    ycenters = (np.arange(ny) + 0.5) * hy

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=xf,
            x=xcenters,
            y=ycenters,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>density=%{z:.3f}<extra></extra>",
        )
    )

    for edge in spec_dict.get("fixed_edges", []):
        edge = str(edge).lower()
        if edge == "left":
            fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=height, line=dict(color="red", width=4))
        elif edge == "right":
            fig.add_shape(type="line", x0=length, x1=length, y0=0, y1=height, line=dict(color="red", width=4))
        elif edge == "bottom":
            fig.add_shape(type="line", x0=0, x1=length, y0=0, y1=0, line=dict(color="red", width=4))
        elif edge == "top":
            fig.add_shape(type="line", x0=0, x1=length, y0=height, y1=height, line=dict(color="red", width=4))

    pc = spec_dict.get("point_constraints", [])
    if pc:
        px = [float(p[0]) for p in pc]
        py = [float(p[1]) for p in pc]
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="markers",
                marker=dict(symbol="square", size=12, color="green", line=dict(width=1, color="black")),
                name="Point constraints",
            )
        )

    loads = spec_dict.get("loads", [])
    for load in loads:
        lx, ly, fx, fy = [float(v) for v in load]
        fig.add_trace(
            go.Scatter(
                x=[lx],
                y=[ly],
                mode="markers",
                marker=dict(symbol="circle", size=12, color="red", line=dict(width=1, color="black")),
                name="Load",
                showlegend=False,
            )
        )
        ay = ly + (0.18 * height if fy < 0 else -0.18 * height)
        fig.add_annotation(
            x=lx,
            y=ly,
            ax=lx,
            ay=ay,
            text=f"{abs(fy):.0f} N",
            showarrow=True,
            arrowhead=6,
            arrowcolor="red",
            font=dict(color="red"),
        )

    fig.update_layout(
        title=title,
        xaxis_title="X [mm]",
        yaxis_title="Y [mm]",
        xaxis=dict(range=[0, length]),
        yaxis=dict(range=[0, height], scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
    )
    return fig


def make_displacement_fig(disp: np.ndarray, spec_dict: Dict[str, Any]) -> go.Figure:
    nx = int(spec_dict.get("nx", disp.shape[1] - 1))
    ny = int(spec_dict.get("ny", disp.shape[0] - 1))
    length = float(spec_dict.get("length", nx))
    height = float(spec_dict.get("height", ny))

    hx = length / max(1, nx)
    hy = height / max(1, ny)

    x = np.arange(nx + 1) * hx
    y = np.arange(ny + 1) * hy

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=disp,
            x=x,
            y=y,
            colorscale="Plasma",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>|u|=%{z:.6f} mm<extra></extra>",
        )
    )
    fig.update_layout(
        title="Nodal displacement magnitude",
        xaxis_title="X [mm]",
        yaxis_title="Y [mm]",
        xaxis=dict(range=[0, length]),
        yaxis=dict(range=[0, height], scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,
    )
    return fig


def make_energy_fig(energy: np.ndarray, spec_dict: Dict[str, Any]) -> go.Figure:
    nx = int(spec_dict.get("nx", energy.shape[1]))
    ny = int(spec_dict.get("ny", energy.shape[0]))
    length = float(spec_dict.get("length", nx))
    height = float(spec_dict.get("height", ny))

    hx = length / max(1, nx)
    hy = height / max(1, ny)

    xcenters = (np.arange(nx) + 0.5) * hx
    ycenters = (np.arange(ny) + 0.5) * hy

    e = np.asarray(energy, dtype=float)
    emax = float(np.nanmax(e)) if e.size else 0.0
    if emax <= 0.0:
        e_norm = np.zeros_like(e)
    else:
        e_norm = e / emax

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=e_norm,
            x=xcenters,
            y=ycenters,
            colorscale="Magma",
            zmin=0.0,
            zmax=1.0,
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>normalized strain energy=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Normalized element strain energy",
        xaxis_title="X [mm]",
        yaxis_title="Y [mm]",
        xaxis=dict(range=[0, length]),
        yaxis=dict(range=[0, height], scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,
    )
    return fig


def make_convergence_fig(history: pd.DataFrame) -> go.Figure:
    df = history.copy()
    df["iteration"] = df["iteration"] + 1
    df = df.replace([np.inf, -np.inf], np.nan)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["compliance"], name="Compliance", mode="lines+markers"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["iteration"], y=df["volume"], name="Volume fraction", mode="lines+markers", line=dict(dash="dot")),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Compliance [N-mm]", secondary_y=False)
    fig.update_yaxes(title_text="Volume fraction", secondary_y=True, range=[0, 1])
    fig.update_xaxes(title_text="Iteration")
    fig.update_layout(height=340, margin=dict(l=40, r=40, t=40, b=40), legend=dict(orientation="h"))
    return fig


def make_mesh3d_fig(verts: np.ndarray, faces: np.ndarray) -> Optional[go.Figure]:
    if faces.size == 0:
        return None

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=i,
            j=j,
            k=k,
            color="skyblue",
            opacity=0.95,
            flatshading=True,
            name="Reconstructed geometry",
        )
    )
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=10),
        height=520,
    )
    return fig


def render_figure(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df: pd.DataFrame, key: str) -> None:
    st.dataframe(df, use_container_width=True, key=key)


# ==============================================================================
# Mesh sensitivity check
# ==============================================================================

def run_mesh_sensitivity(spec: DesignSpec) -> Dict[str, Any]:
    base_nx = int(spec.nx)
    base_ny = int(spec.ny)

    coarse_nx = int(max(8, min(160, round(base_nx * 0.75))))
    coarse_ny = int(max(4, min(80, round(base_ny * 0.75))))

    fine_nx = int(max(coarse_nx + 2, min(160, round(base_nx * 1.25))))
    fine_ny = int(max(coarse_ny + 2, min(80, round(base_ny * 1.25))))

    coarse_radius = float(max(1.0, spec.filter_radius * coarse_nx / max(1, base_nx)))
    fine_radius = float(max(1.0, spec.filter_radius * fine_nx / max(1, base_nx)))

    coarse = replace(
        spec,
        name=spec.name + " coarse",
        nx=coarse_nx,
        ny=coarse_ny,
        iterations=int(max(5, min(spec.iterations, 15))),
        filter_radius=coarse_radius,
    )

    fine = replace(
        spec,
        name=spec.name + " fine",
        nx=fine_nx,
        ny=fine_ny,
        iterations=int(max(5, min(spec.iterations, 15))),
        filter_radius=fine_radius,
    )

    res_c, _ = run_topology_optimization(coarse)
    res_f, _ = run_topology_optimization(fine)

    iou = mask_iou(
        res_c.physical_density >= spec.threshold,
        res_f.physical_density >= spec.threshold,
    )

    table = pd.DataFrame(
        [
            {
                "case": "coarse",
                "nx": coarse.nx,
                "ny": coarse.ny,
                "compliance_Nmm": res_c.compliance,
                "volume_fraction": res_c.volume_fraction,
                "max_displacement_mm": res_c.max_displacement,
                "runtime_s": res_c.runtime_s,
                "iterations": res_c.iterations_run,
            },
            {
                "case": "fine",
                "nx": fine.nx,
                "ny": fine.ny,
                "compliance_Nmm": res_f.compliance,
                "volume_fraction": res_f.volume_fraction,
                "max_displacement_mm": res_f.max_displacement,
                "runtime_s": res_f.runtime_s,
                "iterations": res_f.iterations_run,
            },
        ]
    )

    return {
        "table": table,
        "iou": iou,
        "coarse_density": res_c.physical_density,
        "fine_density": res_f.physical_density,
        "coarse_spec": res_c.spec,
        "fine_spec": res_f.spec,
    }


# ==============================================================================
# Benchmark validation
# ==============================================================================

def run_benchmark_validation() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    mesh = StructuredMesh.create(12, 6, 24.0, 12.0)
    fixed = build_fixed_dofs(mesh, ["left"], [])
    force = build_force_vector(mesh, [(24.0, 6.0, 0.0, -100.0)])
    fe = FEEngine(mesh, 0.33)
    nel = mesh.elem_dofs.shape[0]

    E1 = np.ones(nel, dtype=float)
    E2 = 2.0 * np.ones(nel, dtype=float)

    _, c1, _ = fe.solve(E1, fixed, force)
    _, c2, _ = fe.solve(E2, fixed, force)

    ratio = float(c2 / c1) if c1 > 0 and np.isfinite(c1) and np.isfinite(c2) else float("nan")
    rows.append(
        {
            "check": "FE linear stiffness scaling",
            "metric": "C(2E) / C(E)",
            "value": ratio,
            "expected": 0.5,
            "status": "PASS" if np.isfinite(ratio) and abs(ratio - 0.5) < 0.05 else "REVIEW",
            "notes": "For linear elasticity, doubling Young's modulus should halve compliance.",
        }
    )

    spec = replace(
        cantilever_spec(),
        name="Validation cantilever",
        nx=24,
        ny=10,
        length=48.0,
        height=20.0,
        thickness=5.0,
        iterations=8,
        volume_fraction=0.35,
        penal=3.0,
        filter_radius=1.5,
        move_limit=0.2,
        oc_eta=0.5,
        fixed_edges=["left"],
        point_constraints=[],
        loads=[(48.0, 10.0, 0.0, -100.0)],
        max_permitted_displacement=5.0,
        threshold=0.5,
        seed=0,
    )

    res, _ = run_topology_optimization(spec)

    c_first = float(res.history["compliance"].iloc[0])
    c_last = float(res.history["compliance"].iloc[-1])
    improvement = (
        float((c_first - c_last) / c_first * 100.0)
        if c_first > 0 and np.isfinite(c_first) and np.isfinite(c_last)
        else float("nan")
    )

    rows.append(
        {
            "check": "SIMP compliance trend",
            "metric": "Compliance reduction percent",
            "value": improvement,
            "expected": ">= 0%",
            "status": "PASS" if np.isfinite(improvement) and c_last <= c_first * 1.05 else "REVIEW",
            "notes": "Compliance should generally decrease or remain stable during minimization.",
        }
    )

    vol_err = float(abs(res.volume_fraction - spec.volume_fraction))
    rows.append(
        {
            "check": "Volume constraint",
            "metric": "Absolute volume error",
            "value": vol_err,
            "expected": "<= 0.08",
            "status": "PASS" if vol_err <= 0.08 else "REVIEW",
            "notes": "Filtered volume should remain close to the target volume fraction.",
        }
    )

    sens = run_mesh_sensitivity(replace(spec, iterations=6))
    iou = float(sens["iou"])

    rows.append(
        {
            "check": "Mesh sensitivity topology IoU",
            "metric": "Coarse vs fine IoU",
            "value": iou,
            "expected": ">= 0.25",
            "status": "PASS" if iou >= 0.25 else "REVIEW",
            "notes": "Low IoU can indicate mesh dependence, insufficient iterations, or a highly sensitive topology.",
        }
    )

    runtime = float(res.runtime_s)
    rows.append(
        {
            "check": "Small-case runtime",
            "metric": "Seconds",
            "value": runtime,
            "expected": "< 30 s",
            "status": "PASS" if runtime < 30.0 else "REVIEW",
            "notes": "Validation cases are intentionally small and should run quickly on a CPU.",
        }
    )

    return {
        "table": pd.DataFrame(rows),
        "optimization_result": res,
        "sensitivity": sens,
    }


# ==============================================================================
# Streamlit UI state helpers
# ==============================================================================

def ui_state_from_spec(spec: DesignSpec) -> Dict[str, Any]:
    load = spec.loads[0] if spec.loads else (spec.length * 0.5, spec.height * 0.5, 0.0, -1000.0)
    force_mag = math.hypot(float(load[2]), float(load[3]))
    if force_mag <= 0.0:
        force_mag = 1000.0

    return {
        "ui_preset": spec.name if spec.name in UI_PRESETS else "Cantilever",
        "ui_nx": int(np.clip(spec.nx, 4, 160)),
        "ui_ny": int(np.clip(spec.ny, 4, 80)),
        "ui_length": float(spec.length),
        "ui_height": float(spec.height),
        "ui_thickness": float(spec.thickness),
        "ui_volfrac": float(np.clip(spec.volume_fraction, 0.05, 0.90)),
        "ui_penal": float(np.clip(spec.penal, 1.0, 5.0)),
        "ui_filter": float(np.clip(spec.filter_radius, 0.0, 5.0)),
        "ui_iters": int(np.clip(spec.iterations, 1, 200)),
        "ui_move": float(np.clip(spec.move_limit, 0.01, 0.50)),
        "ui_eta": float(np.clip(spec.oc_eta, 0.05, 1.0)),
        "ui_material": spec.material_name if spec.material_name in MATERIALS else "Aluminum 6061",
        "ui_E": float(spec.E0),
        "ui_nu": float(np.clip(spec.nu, 0.0, 0.49)),
        "ui_rho": float(spec.rho),
        "ui_force": float(force_mag),
        "ui_load_x": float(np.clip(load[0] / spec.length, 0.0, 1.0)) if spec.length > 0 else 0.5,
        "ui_load_y": float(np.clip(load[1] / spec.height, 0.0, 1.0)) if spec.height > 0 else 0.5,
        "ui_fixed_edge": spec.fixed_edges[0] if spec.fixed_edges else "None",
        "ui_maxdisp": float(spec.max_permitted_displacement),
        "ui_threshold": float(np.clip(spec.threshold, 0.05, 0.95)),
        "ui_seed": int(spec.seed),
    }


def apply_spec_to_session(spec: DesignSpec, include_preset: bool = True) -> None:
    d = ui_state_from_spec(spec)
    if not include_preset:
        d.pop("ui_preset", None)
    st.session_state.update(d)


def initialize_session_state() -> None:
    if "sdop_initialized" in st.session_state:
        return

    auto = auto_demo_spec()
    apply_spec_to_session(auto, include_preset=True)

    st.session_state["sdop_initialized"] = True
    st.session_state["auto_demo_done"] = False
    st.session_state["result"] = None
    st.session_state["video_path"] = None
    st.session_state["sensitivity"] = None
    st.session_state["validation"] = None
    st.session_state["last_spec_obj"] = auto
    st.session_state["last_preset"] = auto.name
    st.session_state["last_material"] = auto.material_name


def scaled_point_constraints(base: DesignSpec, length: float, height: float) -> List[Tuple[float, float, bool, bool]]:
    out: List[Tuple[float, float, bool, bool]] = []
    for x, y, fx, fy in base.point_constraints:
        xf = float(x) / float(base.length) if base.length > 0 else 0.0
        yf = float(y) / float(base.height) if base.height > 0 else 0.0
        out.append((xf * float(length), yf * float(height), bool(fx), bool(fy)))
    return out


def spec_from_ui() -> DesignSpec:
    p = st.session_state.get("ui_preset", "Live Demo Cantilever")
    base = PRESETS.get(p, auto_demo_spec)()

    length = float(st.session_state.get("ui_length", base.length))
    height = float(st.session_state.get("ui_height", base.height))
    thickness = float(st.session_state.get("ui_thickness", base.thickness))

    load_x = float(st.session_state.get("ui_load_x", 0.5)) * length
    load_y = float(st.session_state.get("ui_load_y", 0.5)) * height
    force = float(st.session_state.get("ui_force", 1000.0))
    loads = [(load_x, load_y, 0.0, -abs(force))]

    if p == "Custom":
        edge = st.session_state.get("ui_fixed_edge", "left")
        fixed_edges = [edge] if edge != "None" else []
        point_constraints = []
    else:
        fixed_edges = list(base.fixed_edges)
        point_constraints = scaled_point_constraints(base, length, height)

    return DesignSpec(
        name=p,
        nx=int(st.session_state.get("ui_nx", base.nx)),
        ny=int(st.session_state.get("ui_ny", base.ny)),
        length=length,
        height=height,
        thickness=thickness,
        volume_fraction=float(st.session_state.get("ui_volfrac", base.volume_fraction)),
        penal=float(st.session_state.get("ui_penal", base.penal)),
        filter_radius=float(st.session_state.get("ui_filter", base.filter_radius)),
        iterations=int(st.session_state.get("ui_iters", base.iterations)),
        move_limit=float(st.session_state.get("ui_move", base.move_limit)),
        oc_eta=float(st.session_state.get("ui_eta", base.oc_eta)),
        E0=float(st.session_state.get("ui_E", base.E0)),
        Emin=float(base.Emin),
        nu=float(st.session_state.get("ui_nu", base.nu)),
        rho=float(st.session_state.get("ui_rho", base.rho)),
        material_name=str(st.session_state.get("ui_material", base.material_name)),
        fixed_edges=fixed_edges,
        point_constraints=point_constraints,
        loads=loads,
        max_permitted_displacement=float(st.session_state.get("ui_maxdisp", base.max_permitted_displacement)),
        threshold=float(st.session_state.get("ui_threshold", base.threshold)),
        seed=int(st.session_state.get("ui_seed", base.seed)),
    )


# ==============================================================================
# Streamlit rendering
# ==============================================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.header("Design inputs")

        preset = st.selectbox(
            "Benchmark preset",
            UI_PRESETS,
            key="ui_preset",
        )

        if st.session_state.get("last_preset") != preset:
            spec = PRESETS.get(preset, auto_demo_spec)()
            apply_spec_to_session(spec, include_preset=False)
            st.session_state["last_preset"] = preset
            st.session_state["last_material"] = spec.material_name

        c1, c2 = st.columns(2)
        c1.number_input("Elements X", min_value=4, max_value=160, step=1, key="ui_nx")
        c2.number_input("Elements Y", min_value=4, max_value=80, step=1, key="ui_ny")

        c1, c2, c3 = st.columns(3)
        c1.number_input("Length [mm]", min_value=1.0, step=1.0, key="ui_length")
        c2.number_input("Height [mm]", min_value=1.0, step=1.0, key="ui_height")
        c3.number_input("Thickness [mm]", min_value=0.1, step=0.5, key="ui_thickness")

        material = st.selectbox(
            "Material",
            list(MATERIALS.keys()),
            key="ui_material",
        )

        if st.session_state.get("last_material") != material:
            mat = MATERIALS.get(material)
            if mat is not None:
                st.session_state["ui_E"] = float(mat.E)
                st.session_state["ui_nu"] = float(mat.nu)
                st.session_state["ui_rho"] = float(mat.rho)
            st.session_state["last_material"] = material

        c1, c2, c3 = st.columns(3)
        c1.number_input("Young's modulus [MPa]", min_value=1.0, step=100.0, key="ui_E")
        c2.number_input("Poisson's ratio", min_value=0.0, max_value=0.49, step=0.01, key="ui_nu")
        c3.number_input("Density [kg/m^3]", min_value=1.0, step=10.0, key="ui_rho")

        st.subheader("Loads and limits")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Load magnitude [N]", min_value=1.0, step=10.0, key="ui_force")
        c2.slider("Load X fraction", min_value=0.0, max_value=1.0, step=0.01, key="ui_load_x")
        c3.slider("Load Y fraction", min_value=0.0, max_value=1.0, step=0.01, key="ui_load_y")

        st.selectbox(
            "Fixed edge for Custom preset",
            ["None", "left", "right", "bottom", "top"],
            key="ui_fixed_edge",
        )

        st.number_input(
            "Maximum permitted displacement [mm]",
            min_value=0.0,
            step=0.1,
            key="ui_maxdisp",
        )

        with st.expander("Advanced solver settings"):
            st.slider("Target volume fraction", min_value=0.05, max_value=0.90, step=0.01, key="ui_volfrac")
            st.number_input("Iterations", min_value=1, max_value=200, step=1, key="ui_iters")
            st.number_input("SIMP penalization", min_value=1.0, max_value=5.0, step=0.1, key="ui_penal")
            st.number_input("Density filter radius [elements]", min_value=0.0, max_value=5.0, step=0.1, key="ui_filter")
            st.number_input("Move limit", min_value=0.01, max_value=0.50, step=0.01, key="ui_move")
            st.number_input("OC exponent eta", min_value=0.05, max_value=1.00, step=0.05, key="ui_eta")
            st.slider("Reconstruction threshold", min_value=0.05, max_value=0.95, step=0.01, key="ui_threshold")
            st.number_input("Random seed", min_value=0, max_value=100000, step=1, key="ui_seed")

        st.divider()

        col1, col2 = st.columns(2)
        run_btn = col1.button("Run optimization", use_container_width=True, key="btn_run_opt")
        video_btn = col2.button("Generate video", use_container_width=True, key="btn_gen_video")

        if run_btn:
            spec = spec_from_ui()
            st.session_state["last_spec_obj"] = spec
            with st.spinner("Running SIMP topology optimization..."):
                result, _ = run_topology_optimization(spec)
            st.session_state["result"] = result
            st.session_state["sensitivity"] = None
            st.session_state["recon_threshold"] = float(result.spec.get("threshold", 0.5))

        if video_btn:
            spec = spec_from_ui()
            st.session_state["last_spec_obj"] = spec
            video_file = os.path.join(get_output_dir(), f"demo_{int(time.time())}.mp4")
            with st.spinner("Rendering live demo video..."):
                result, video_path = run_topology_optimization(spec, record_video_path=video_file)
            st.session_state["result"] = result
            st.session_state["video_path"] = video_path
            st.session_state["sensitivity"] = None
            st.session_state["recon_threshold"] = float(result.spec.get("threshold", 0.5))

        reset_btn = st.button("Reset live demo", use_container_width=True, key="btn_reset_demo")
        if reset_btn:
            auto = auto_demo_spec()
            apply_spec_to_session(auto, include_preset=True)
            st.session_state["last_spec_obj"] = auto
            st.session_state["last_preset"] = auto.name
            st.session_state["last_material"] = auto.material_name
            st.session_state["auto_demo_done"] = False
            st.session_state["result"] = None
            st.session_state["video_path"] = None
            st.session_state["sensitivity"] = None
            st.session_state["validation"] = None
            st.rerun()


def render_video_block(video_path: Optional[str], key_prefix: str = "video") -> None:
    if not video_path or not os.path.exists(video_path):
        st.warning("No local video file was generated. Install OpenCV headless and rerun video generation.")
        return

    is_mp4 = str(video_path).lower().endswith(".mp4")
    fmt = "video/mp4" if is_mp4 else "video/x-msvideo"
    video_bytes = None

    try:
        with open(video_path, "rb") as fh:
            video_bytes = fh.read()
    except Exception as exc:
        st.error(f"Could not read video file: {exc}")
        return

    if not video_bytes:
        st.warning("Video file exists but is empty.")
        return

    displayed = False

    try:
        b64 = base64.b64encode(video_bytes).decode("ascii")
        html = (
            '<video width="100%" height="360" controls autoplay muted playsinline>'
            f'<source src="data:{fmt};base64,{b64}" type="{fmt}">'
            "Your browser does not support embedded video."
            "</video>"
        )
        st.components.v1.html(html, height=380, scrolling=False)
        displayed = True
    except Exception:
        displayed = False

    if not displayed:
        try:
            st.video(video_bytes, format=fmt)
        except Exception as exc:
            st.error(f"Could not display video: {exc}")

    st.download_button(
        "Download live demo video",
        data=video_bytes,
        file_name=os.path.basename(video_path),
        mime=fmt,
        key=f"{key_prefix}_download_video",
    )


def render_dashboard(result: OptimizationResult) -> None:
    spec = result.spec

    if st.session_state.get("video_path"):
        st.subheader("Live demo video")
        render_video_block(st.session_state.get("video_path"), key_prefix="dashboard")

    st.subheader("Key metrics")

    volume_mm3 = (
        float(result.volume_fraction)
        * float(spec.get("length", 1.0))
        * float(spec.get("height", 1.0))
        * float(spec.get("thickness", 1.0))
    )
    mass_g = volume_mm3 * float(spec.get("rho", 2700.0)) * 1e-6

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Compliance", f"{result.compliance:.3g} N-mm")
    c2.metric("Volume fraction", f"{result.volume_fraction:.3f}")
    c3.metric("Max displacement", f"{result.max_displacement:.3f} mm")
    c4.metric("Runtime", f"{result.runtime_s:.2f} s")
    c5.metric("Iterations", str(result.iterations_run))
    c6.metric("Estimated mass", f"{mass_g:.1f} g")

    if result.max_displacement > float(spec.get("max_permitted_displacement", 1e9)):
        st.warning(
            "Maximum displacement exceeds the user-defined limit. "
            "This is a numerical indicator only, not a safety certification."
        )

    if not result.converged:
        st.info("The solver reached the iteration limit before the convergence tolerances were satisfied.")

    st.subheader("Density and structural response")
    render_figure(make_density_fig(result.physical_density, spec), key="dash_density")

    c1, c2 = st.columns(2)
    with c1:
        render_figure(make_displacement_fig(result.nodal_displacement, spec), key="dash_disp")
    with c2:
        render_figure(make_energy_fig(result.element_energy, spec), key="dash_energy")

    st.subheader("Convergence monitoring")
    render_figure(make_convergence_fig(result.history), key="dash_conv")


def render_reconstruction_tab(result: OptimizationResult) -> None:
    spec = result.spec
    hx = float(spec.get("length", 1.0)) / float(max(1, int(spec.get("nx", 1))))
    hy = float(spec.get("height", 1.0)) / float(max(1, int(spec.get("ny", 1))))
    thickness = float(spec.get("thickness", 1.0))

    threshold = st.slider(
        "Reconstruction threshold",
        min_value=0.05,
        max_value=0.95,
        step=0.01,
        value=float(st.session_state.get("recon_threshold", spec.get("threshold", 0.5))),
        key="recon_threshold",
    )

    verts, faces, mesh = build_surface_mesh(result.physical_density, threshold, hx, hy, thickness)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vertices", str(len(verts)))
    c2.metric("Triangles", str(len(faces)))
    c3.metric("Threshold", f"{threshold:.2f}")
    c4.metric("Watertight", "Yes" if mesh is not None and bool(getattr(mesh, "is_watertight", False)) else "Unknown/No")

    if faces.size == 0:
        st.warning("No solid voxels above threshold. Lower the threshold or rerun optimization.")
        return

    fig = make_mesh3d_fig(verts, faces)
    if fig is not None:
        render_figure(fig, key="recon_mesh")

    if mesh is not None:
        try:
            st.write(f"Mesh volume estimate: {float(mesh.volume):.3f} mm^3")
            st.write(f"Mesh surface area estimate: {float(mesh.area):.3f} mm^2")
        except Exception:
            pass


def render_export_tab(result: OptimizationResult) -> None:
    spec = result.spec
    threshold = float(st.session_state.get("recon_threshold", spec.get("threshold", 0.5)))
    hx = float(spec.get("length", 1.0)) / float(max(1, int(spec.get("nx", 1))))
    hy = float(spec.get("height", 1.0)) / float(max(1, int(spec.get("ny", 1))))
    thickness = float(spec.get("thickness", 1.0))

    verts, faces, mesh = build_surface_mesh(result.physical_density, threshold, hx, hy, thickness)
    hex_nodes, hex_elems = build_voxel_hex_mesh(result.physical_density, threshold, hx, hy, thickness)

    st.subheader("Numerical data exports")

    st.download_button(
        "Download density CSV",
        data=density_csv_bytes(result.physical_density),
        file_name="physical_density.csv",
        mime="text/csv",
        key="dl_density_csv",
    )

    st.download_button(
        "Download convergence history CSV",
        data=history_csv_bytes(result.history),
        file_name="convergence_history.csv",
        mime="text/csv",
        key="dl_history_csv",
    )

    st.download_button(
        "Download summary JSON",
        data=summary_json_bytes(result),
        file_name="optimization_summary.json",
        mime="application/json",
        key="dl_summary_json",
    )

    st.subheader("CAD and FEM exports")

    st.download_button(
        "Download STL",
        data=stl_bytes_from_mesh(verts, faces, mesh),
        file_name="optimized_structure.stl",
        mime="model/stl",
        key="dl_stl",
    )

    st.download_button(
        "Download OBJ",
        data=obj_bytes(verts, faces),
        file_name="optimized_structure.obj",
        mime="text/plain",
        key="dl_obj",
    )

    st.download_button(
        "Download Gmsh .msh surface mesh",
        data=gmsh_msh_bytes(verts, faces),
        file_name="optimized_structure.msh",
        mime="text/plain",
        key="dl_gmsh_msh",
    )

    st.download_button(
        "Download CalculiX .inp voxel volume mesh",
        data=calculix_inp_bytes(hex_nodes, hex_elems, spec),
        file_name="optimized_structure.inp",
        mime="text/plain",
        key="dl_calculix_inp",
    )

    st.download_button(
        "Download FreeCAD macro placeholder",
        data=freecad_macro_bytes(),
        file_name="open_in_freecad.FCMacro",
        mime="text/plain",
        key="dl_freecad_macro",
    )

    st.subheader("External tool status")
    ccx_exe = shutil.which("ccx") or shutil.which("calculix")
    freecad_exe = shutil.which("freecad") or shutil.which("FreeCAD")
    gmsh_exe = shutil.which("gmsh")

    st.write(f"Gmsh Python module: {'available' if HAS_GMSH else 'not detected'}")
    st.write(f"Gmsh executable: {gmsh_exe or 'not detected'}")
    st.write(f"CalculiX executable: {ccx_exe or 'not detected'}")
    st.write(f"FreeCAD executable: {freecad_exe or 'not detected'}")

    st.info(
        "Exported geometry is a numerical reconstruction for review, downstream meshing, "
        "and independent verification. It is not a manufacturing release package."
    )


def render_sensitivity_tab(result: OptimizationResult) -> None:
    st.subheader("Mesh sensitivity check")
    st.write(
        "This runs the same design problem on coarse and fine structured meshes and compares "
        "compliance, volume, displacement, runtime, and topology overlap. It is a numerical "
        "convergence diagnostic, not a physical safety proof."
    )

    if st.button("Run mesh sensitivity check", key="btn_run_sens"):
        spec = st.session_state.get("last_spec_obj")
        if not isinstance(spec, DesignSpec):
            spec = design_spec_from_dict(result.spec)
        with st.spinner("Running coarse and fine optimization cases..."):
            st.session_state["sensitivity"] = run_mesh_sensitivity(spec)

    sens = st.session_state.get("sensitivity")
    if not sens:
        st.info("No sensitivity run yet.")
        return

    show_table(sens["table"], key="sens_table")
    st.metric("Topology IoU coarse vs fine", f"{sens['iou']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        render_figure(make_density_fig(sens["coarse_density"], sens["coarse_spec"], "Coarse mesh density"), key="sens_coarse")
    with c2:
        render_figure(make_density_fig(sens["fine_density"], sens["fine_spec"], "Fine mesh density"), key="sens_fine")


def render_validation_tab() -> None:
    st.subheader("Benchmark validation")
    st.write(
        "This tab runs small canonical checks for the finite element engine, SIMP optimizer, "
        "volume constraint, mesh sensitivity, and runtime. These are numerical verification "
        "checks, not certification of any optimized geometry for real-world use."
    )

    if st.button("Run benchmark validation", key="btn_run_val"):
        with st.spinner("Running small benchmark validation cases..."):
            st.session_state["validation"] = run_benchmark_validation()

    val = st.session_state.get("validation")
    if not val:
        st.info("No validation run yet.")
        return

    show_table(val["table"], key="val_table")

    val_res = val["optimization_result"]
    st.subheader("Validation cantilever result")
    render_figure(make_density_fig(val_res.physical_density, val_res.spec, "Validation cantilever density"), key="val_density")

    sens = val["sensitivity"]
    st.metric("Validation topology IoU", f"{sens['iou']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        render_figure(make_density_fig(sens["coarse_density"], sens["coarse_spec"], "Validation coarse mesh"), key="val_coarse")
    with c2:
        render_figure(make_density_fig(sens["fine_density"], sens["fine_spec"], "Validation fine mesh"), key="val_fine")


def render_docs_tab() -> None:
    st.markdown(
        r"""
### Platform architecture

This single-file application implements the requested modules:

1. Geometry input manager
   Structured rectangular design domains, preset benchmark geometries, custom dimensions,
   design volume, thickness, and user-controlled resolution.

2. Material database
   Isotropic engineering materials with Young's modulus, Poisson's ratio, density, and
   optional yield strength fields. Custom numeric overrides are supported.

3. Load and boundary-condition manager
   Fixed edges, point constraints, point forces, load position fractions, and displacement limits.

4. Finite element mesh generator
   Structured 4-node bilinear quadrilateral mesh for 2D plane-stress elasticity.

5. Structural analysis engine
   Sparse linear elastic FE solver using SciPy sparse matrices and direct/iterative fallback.

6. Topology optimization engine
   Density-based SIMP method with compliance minimization, volume constraint, density filtering,
   Optimality Criteria update, convergence monitoring, and mesh sensitivity workflow.

7. Geometry reconstruction engine
   Thresholded density field converted to an extruded 3D surface mesh and voxel hexahedral mesh.

8. Result visualization system
   Streamlit dashboard with Plotly heatmaps, convergence curves, 3D mesh view, metrics,
   and automatically generated local live demo video.

9. CAD export engine
   CSV, JSON, STL, OBJ, Gmsh .msh, CalculiX .inp, and FreeCAD macro placeholder exports.

10. Benchmark validation engine
   Small canonical checks for linear FE scaling, SIMP compliance trend, volume constraint,
   mesh sensitivity, and runtime.

### Milestone mapping

- Milestone 1: Rectangular design domain is implemented through StructuredMesh.
- Milestone 2: Basic numerical loop is implemented through the OC/SIMP update.
- Milestone 3: Boundary conditions and loads are managed by build_fixed_dofs and build_force_vector.
- Milestone 4: Mesh generation is implemented by StructuredMesh.create.
- Milestone 5: FE analysis is implemented by FEEngine.
- Milestone 6: Topology optimization is implemented by run_topology_optimization.
- Milestone 7: Exportable 3D reconstruction is implemented by build_surface_mesh and build_voxel_hex_mesh.
- Milestone 8: Visual comparison dashboard, live video, validation, and documentation are implemented in the Streamlit UI.

### Numerical method

The SIMP interpolation is:

E(rho_e) = E_min + rho_e^p * (E_0 - E_min)

The objective is compliance minimization:

C = f^T u

subject to:

mean(filtered rho) <= V_star

A density filter is used to reduce checkerboarding and mesh dependence. The Optimality Criteria
method updates design variables while enforcing the volume constraint by bisection on the Lagrange
multiplier.

### Testing

Embedded Pytest-compatible tests are included in this file. Run:

pytest app.py -v

The tests check:
- material database integrity,
- filter constant preservation,
- element stiffness symmetry,
- small-scale optimization execution,
- compliance trend,
- volume constraint behavior,
- reconstruction,
- STL export,
- Gmsh and CalculiX export bytes.

### Evaluation metrics reported by the app

- compliance,
- volume fraction,
- maximum displacement,
- convergence history,
- design-variable change,
- runtime,
- mesh sensitivity,
- topology intersection-over-union between coarse and fine meshes,
- benchmark validation status.

### Safety and validation disclaimer

This platform is a research and concept-generation tool. It does not include:
- nonlinear material behavior,
- fatigue,
- buckling,
- contact,
- dynamics,
- thermal effects,
- manufacturing constraints,
- code-compliant safety factors,
- physical validation.

Do not use an optimized geometry for real-world loading without independent engineering
verification, appropriate standards review, and qualified professional sign-off.
"""
    )


def render_main_tabs() -> None:
    result = st.session_state.get("result")
    if result is None:
        st.info("No optimization result is available yet. Use the sidebar to run a case.")
        return

    tabs = st.tabs(
        [
            "Dashboard",
            "3D reconstruction",
            "Exports and CAD",
            "Mesh sensitivity",
            "Validation",
            "Documentation",
        ]
    )

    with tabs[0]:
        render_dashboard(result)

    with tabs[1]:
        render_reconstruction_tab(result)

    with tabs[2]:
        render_export_tab(result)

    with tabs[3]:
        render_sensitivity_tab(result)

    with tabs[4]:
        render_validation_tab()

    with tabs[5]:
        render_docs_tab()


def main() -> None:
    st.set_page_config(
        page_title="Structural Design Optimization Platform",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    st.title("Structural Design Optimization Platform")
    st.caption(
        "Density-based SIMP topology optimization with live demo video, 3D reconstruction, "
        "exports, mesh sensitivity, and benchmark validation. Research/concept use only."
    )

    st.warning(
        "Not certified for real-world structural use. Any physical implementation requires "
        "independent engineering verification and professional oversight."
    )

    if not st.session_state.get("auto_demo_done", False):
        auto = auto_demo_spec()
        st.session_state["last_spec_obj"] = auto
        video_file = os.path.join(get_output_dir(), "live_demo.mp4")

        result = None
        video_path = None

        try:
            with st.spinner("Automatically loading realistic demo data and generating live demo video..."):
                result, video_path = run_topology_optimization(auto, record_video_path=video_file)
        except Exception:
            try:
                with st.spinner("Video generation failed. Running fallback optimization without video..."):
                    result, video_path = run_topology_optimization(auto, record_video_path=None)
            except Exception as exc:
                st.exception(exc)

        if result is not None:
            st.session_state["result"] = result
            st.session_state["video_path"] = video_path
            st.session_state["recon_threshold"] = float(result.spec.get("threshold", 0.5))

        st.session_state["auto_demo_done"] = True

    render_sidebar()
    render_main_tabs()


# ==============================================================================
# Embedded Pytest-compatible tests
# ==============================================================================

def test_material_database_has_common_materials() -> None:
    assert "Aluminum 6061" in MATERIALS
    assert "Steel A36" in MATERIALS
    assert MATERIALS["Aluminum 6061"].E > 0.0
    assert 0.0 <= MATERIALS["Aluminum 6061"].nu <= 0.49


def test_density_filter_preserves_constant_field() -> None:
    filt = DensityFilter(1.5)
    arr = np.full((10, 12), 0.4, dtype=float)
    out = filt(arr)
    assert np.allclose(out, 0.4, atol=1e-5)


def test_q4_base_stiffness_is_symmetric() -> None:
    ke = q4_plane_stress_base_stiffness(1.0, 0.5, 0.3)
    assert np.allclose(ke, ke.T, atol=1e-10)
    eig = np.linalg.eigvalsh(ke)
    assert float(np.min(eig)) > -1e-8


def test_small_cantilever_optimization_runs() -> None:
    spec = DesignSpec(
        name="Pytest Cantilever",
        nx=16,
        ny=8,
        length=32.0,
        height=16.0,
        thickness=2.0,
        volume_fraction=0.35,
        penal=3.0,
        filter_radius=1.5,
        iterations=4,
        E0=68900.0,
        Emin=1e-6,
        nu=0.33,
        rho=2700.0,
        fixed_edges=["left"],
        loads=[(32.0, 8.0, 0.0, -100.0)],
        seed=0,
    )

    result, video_path = run_topology_optimization(spec)
    assert video_path is None
    assert result.history.shape[0] >= 1
    assert np.isfinite(result.compliance)
    assert 0.20 <= result.volume_fraction <= 0.55


def test_compliance_does_not_increase_significantly() -> None:
    spec = DesignSpec(
        name="Pytest Trend",
        nx=16,
        ny=8,
        length=32.0,
        height=16.0,
        thickness=2.0,
        volume_fraction=0.35,
        iterations=6,
        fixed_edges=["left"],
        loads=[(32.0, 8.0, 0.0, -100.0)],
        seed=1,
    )

    result, _ = run_topology_optimization(spec)
    c0 = float(result.history["compliance"].iloc[0])
    c1 = float(result.history["compliance"].iloc[-1])
    assert c1 <= c0 * 1.20 + 1e-9


def test_reconstruction_and_stl_export() -> None:
    density = np.ones((4, 5), dtype=float)
    verts, faces, mesh = build_surface_mesh(density, threshold=0.5, hx=1.0, hy=1.0, thickness=1.0)

    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(faces) > 0

    stl = stl_bytes_from_mesh(verts, faces, mesh)
    assert len(stl) > 80


def test_gmsh_and_calculix_export_bytes() -> None:
    density = np.ones((3, 3), dtype=float)
    verts, faces, _ = build_surface_mesh(density, 0.5, 1.0, 1.0, 1.0)
    nodes, elems = build_voxel_hex_mesh(density, 0.5, 1.0, 1.0, 1.0)

    msh = gmsh_msh_bytes(verts, faces)
    inp = calculix_inp_bytes(nodes, elems, {"E0": 70000.0, "nu": 0.33})

    assert b"$MeshFormat" in msh
    assert b"*NODE" in inp
    assert b"*ELEMENT" in inp


if __name__ == "__main__":
    main()