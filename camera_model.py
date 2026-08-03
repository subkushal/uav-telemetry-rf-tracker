"""
camera_model.py — Phase 2 | Step 1
Camera Calibration & Intrinsic Matrix Model

Replaces the Phase 1 simple FOV approximation with a proper
computer-vision camera model following the standard OpenCV/Hartley-Zisserman
pinhole + radial/tangential distortion pipeline.

Theory
------
Standard pinhole camera model:

    [u]   [fx  0  cx] [Xc/Zc]
    [v] = [0  fy  cy] [Yc/Zc]   (in pixels)
    [1]   [0   0   1] [  1  ]

where (fx, fy) are focal lengths in pixels and (cx, cy) is the
principal point (image centre).

Lens distortion (Brown–Conrady model, same as OpenCV):

    r²  = x²  + y²            (x, y are normalised undistorted coords)
    x'  = x(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*xy + p2*(r² + 2x²)
    y'  = y(1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2y²) + 2*p2*xy

    u   = fx*x' + cx
    v   = fy*y' + cy

Usage
-----
    from camera_model import CameraIntrinsics, CameraPresets

    # Use a pre-built drone camera profile
    cam = CameraPresets.dji_zenmuse_x7()

    # Or build a custom one from FOV
    cam = CameraIntrinsics.from_fov(hfov_deg=84, frame_w=1920, frame_h=1080)

    # Project a normalised camera-frame ray
    u, v, valid = cam.project_ray(cam_x, cam_y, cam_z)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core intrinsic model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """
    Full camera intrinsic model: focal lengths, principal point, distortion.

    Attributes
    ----------
    fx, fy   : focal length in pixels (fx = f / sensor_pixel_width_m)
    cx, cy   : principal point in pixels (usually ≈ frame centre)
    k1,k2,k3 : radial distortion coefficients
    p1, p2   : tangential distortion coefficients
    frame_w  : image width  in pixels
    frame_h  : image height in pixels
    """
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    frame_w: int = 1280
    frame_h: int = 720

    # ── Derived / convenience ─────────────────────────────────────────

    @property
    def K(self) -> np.ndarray:
        """3×3 camera intrinsic matrix K."""
        return np.array([
            [self.fx,     0.0, self.cx],
            [    0.0, self.fy, self.cy],
            [    0.0,     0.0,     1.0],
        ], dtype=float)

    @property
    def dist_coeffs(self) -> np.ndarray:
        """OpenCV-style distortion vector [k1, k2, p1, p2, k3]."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3],
                        dtype=float)

    @property
    def hfov_deg(self) -> float:
        """Horizontal FOV derived from fx and frame width."""
        return math.degrees(2.0 * math.atan2(self.frame_w / 2.0, self.fx))

    @property
    def vfov_deg(self) -> float:
        """Vertical FOV derived from fy and frame height."""
        return math.degrees(2.0 * math.atan2(self.frame_h / 2.0, self.fy))

    # ── Factory constructors ──────────────────────────────────────────

    @classmethod
    def from_fov(
        cls,
        hfov_deg: float,
        frame_w: int = 1280,
        frame_h: int = 720,
        vfov_deg: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
    ) -> "CameraIntrinsics":
        """
        Build intrinsics from horizontal FOV (most common drone spec).
        Vertical FOV auto-derived from aspect ratio when vfov_deg=0.
        """
        fx = (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)

        if vfov_deg > 0:
            fy = (frame_h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        else:
            # Square pixels: fy = fx (standard assumption)
            fy = fx

        cx = frame_w / 2.0
        cy = frame_h / 2.0

        return cls(fx=fx, fy=fy, cx=cx, cy=cy,
                   k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
                   frame_w=frame_w, frame_h=frame_h)

    @classmethod
    def from_sensor_spec(
        cls,
        focal_length_mm: float,
        sensor_w_mm: float,
        sensor_h_mm: float,
        frame_w: int = 1920,
        frame_h: int = 1080,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
    ) -> "CameraIntrinsics":
        """
        Build from physical sensor dimensions (data-sheet values).

        fx = focal_length_mm / sensor_w_mm * frame_w  (pixels)
        fy = focal_length_mm / sensor_h_mm * frame_h  (pixels)
        """
        fx = (focal_length_mm / sensor_w_mm) * frame_w
        fy = (focal_length_mm / sensor_h_mm) * frame_h
        cx = frame_w / 2.0
        cy = frame_h / 2.0

        return cls(fx=fx, fy=fy, cx=cx, cy=cy,
                   k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
                   frame_w=frame_w, frame_h=frame_h)

    # ── Projection ────────────────────────────────────────────────────

    def project_ray(
        self,
        cam_x: float,
        cam_y: float,
        cam_z: float,
        apply_distortion: bool = True,
    ) -> Tuple[float, float, bool]:
        """
        Project a 3-D camera-frame ray onto the image plane.

        Parameters
        ----------
        cam_x, cam_y : lateral components (right / down)
        cam_z        : depth along optical axis (must be > 0 for visibility)
        apply_distortion : if True, applies Brown-Conrady lens distortion

        Returns
        -------
        (u, v, visible)
            u, v     : pixel coordinates (float, sub-pixel accurate)
            visible  : True if (u,v) lands inside the image frame
        """
        if cam_z <= 1e-6:
            return 0.0, 0.0, False

        # Step 1 — Normalised (undistorted) image coordinates
        x = cam_x / cam_z
        y = cam_y / cam_z

        # Step 2 — Apply Brown–Conrady distortion
        if apply_distortion and self._has_distortion():
            x, y = self._distort(x, y)

        # Step 3 — Apply intrinsic matrix
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        # Step 4 — Visibility test (with ±10% margin for edge label drawing)
        margin_u = self.frame_w * 0.10
        margin_v = self.frame_h * 0.10
        visible = (
            -margin_u <= u <= self.frame_w + margin_u and
            -margin_v <= v <= self.frame_h + margin_v
        )

        return u, v, visible

    def undistort_point(
        self,
        u: float,
        v: float,
        max_iter: int = 20,
    ) -> Tuple[float, float]:
        """
        Inverse: pixel (u,v) → undistorted normalised coordinates.
        Uses iterative Newton method (same as OpenCV undistortPoints).
        """
        # Initial guess (undistorted)
        x0 = (u - self.cx) / self.fx
        y0 = (v - self.cy) / self.fy

        if not self._has_distortion():
            return x0, y0

        x, y = x0, y0
        for _ in range(max_iter):
            r2 = x * x + y * y
            k_radial = 1.0 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
            dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
            dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
            x = (x0 - dx) / k_radial
            y = (y0 - dy) / k_radial

        return x, y

    def reprojection_error(
        self,
        cam_x: float, cam_y: float, cam_z: float,
        expected_u: float, expected_v: float,
    ) -> float:
        """
        Euclidean pixel distance between projected and expected point.
        Used for projection validation.
        """
        u, v, vis = self.project_ray(cam_x, cam_y, cam_z)
        if not vis:
            return float("inf")
        return math.hypot(u - expected_u, v - expected_v)

    # ── Private helpers ───────────────────────────────────────────────

    def _has_distortion(self) -> bool:
        return any(abs(c) > 1e-10 for c in
                   [self.k1, self.k2, self.k3, self.p1, self.p2])

    def _distort(self, x: float, y: float) -> Tuple[float, float]:
        """Apply Brown–Conrady distortion to normalised coords."""
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        k_radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        x_dist = x * k_radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        y_dist = y * k_radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y

        return x_dist, y_dist

    def __repr__(self) -> str:
        return (
            f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
            f"hfov={self.hfov_deg:.1f}deg, vfov={self.vfov_deg:.1f}deg, "
            f"res={self.frame_w}x{self.frame_h})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built camera presets (common drone cameras)
# ─────────────────────────────────────────────────────────────────────────────

class CameraPresets:
    """
    Factory methods for common drone surveillance camera profiles.
    All distortion values are representative of real lenses from published
    calibration data and research papers.
    """

    @staticmethod
    def dji_zenmuse_x7() -> CameraIntrinsics:
        """
        DJI Zenmuse X7, 24mm lens, 6016×4008 sensor, typical aerial survey config.
        Downscaled to 1920×1080 for simulation.
        Ref: DJI official specs + OpenDroneMap calibration datasets.
        """
        return CameraIntrinsics.from_sensor_spec(
            focal_length_mm=24.0,
            sensor_w_mm=23.5,
            sensor_h_mm=15.7,
            frame_w=1920, frame_h=1080,
            k1=-0.0412, k2=0.0148, k3=-0.0021,
            p1=0.0002,  p2=-0.0003,
        )

    @staticmethod
    def dji_mavic3_wide() -> CameraIntrinsics:
        """
        DJI Mavic 3 wide camera, 24mm equiv, 4/3 CMOS.
        Calibrated at 1280×720 for real-time simulation.
        Ref: DJI Mavic 3 data sheet + community calibration.
        """
        return CameraIntrinsics.from_fov(
            hfov_deg=84.0,
            frame_w=1280, frame_h=720,
            k1=-0.0320, k2=0.0090, k3=-0.0008,
            p1=0.0001,  p2=0.0002,
        )

    @staticmethod
    def generic_surveillance_90() -> CameraIntrinsics:
        """
        Generic 90-deg wide-angle surveillance camera (no distortion).
        Matches Phase 1 simulation config exactly — useful as a baseline
        for before/after distortion comparison.
        """
        return CameraIntrinsics.from_fov(
            hfov_deg=90.0,
            frame_w=1280, frame_h=720,
        )

    @staticmethod
    def fisheye_120() -> CameraIntrinsics:
        """
        Fisheye-like 120-deg lens with heavy barrel distortion.
        For testing distortion correction pipeline.
        """
        return CameraIntrinsics.from_fov(
            hfov_deg=120.0,
            frame_w=1280, frame_h=720,
            k1=-0.2800, k2=0.1200, k3=-0.0180,
            p1=0.0008,  p2=-0.0005,
        )
    
    @staticmethod
    def eo_daylight_10x() -> CameraIntrinsics:
        """
        Custom EO Daylight Payload
        Native 1280x720 resolution. 
        Assuming a baseline wide FOV of ~65 degrees at 1x zoom.
        """
        return CameraIntrinsics.from_fov(
            hfov_deg=65.0,  # We can tweak this if the reticles drift slightly
            frame_w=1280, 
            frame_h=720,
            # We will assume zero distortion for now unless the image looks warped
            k1=0.0, k2=0.0, p1=0.0, p2=0.0
        )

    @staticmethod
    def list_presets() -> list:
        return ["dji_zenmuse_x7", "dji_mavic3_wide",
                "generic_surveillance_90", "fisheye_120"]
    
    # In camera_model.py
    @staticmethod
    def drone_fixed_46() -> CameraIntrinsics:
        return CameraIntrinsics.from_fov(
            hfov_deg=46.0,
            frame_w=1280, frame_h=720,
        )
