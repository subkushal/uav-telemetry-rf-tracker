"""
confidence.py  Phase 2 | Step 3
Multi-Factor Confidence Scoring Engine

Fuses five independent evidence sources into a single
confidence score [0.0 – 1.0] for each projected RF source position.

Evidence sources
----------------
1. Signal strength      — strong RSSI -> high confidence
2. Kalman filter state  — low KF covariance -> high confidence
3. Filter consistency   — NIS near expected value (2.0) -> trusted
4. Track age            — more observations -> more stable
5. Geometry             — low drone-target angle off boresight -> better fix

Fusion method
-------------
Weighted product of individual factor scores, each in [0, 1]:

    confidence = (w1*S1 + w2*S2 + w3*S3 + w4*S4 + w5*S5) / sum(w)

Individual scores are smooth sigmoid / linear mappings calibrated to
produce intuitive confidence levels:
    >= 0.80  HIGH    (green)   — location is reliable
    >= 0.55  MEDIUM  (orange)  — use with caution
    >= 0.30  LOW     (red)     — approximate location only
    <  0.30  POOR    (grey)    — not usable

Temporal smoothing
------------------
An exponential moving average prevents confidence from flickering
between frames while still responding to genuine signal loss.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

from signal_model import SignalMeasurement, strength_to_level
from kalman_filter import KalmanEstimate


# ---------------------------------------------------------------------------
# Confidence level thresholds and styling
# ---------------------------------------------------------------------------

_CONF_TABLE = [
    (0.80, "HIGH",   (0,   200,   0)),   # green
    (0.55, "MEDIUM", (0,   165, 255)),   # orange
    (0.30, "LOW",    (30,   30, 220)),   # red
    (0.00, "POOR",   (80,   80,  80)),   # grey
]

def confidence_to_level(c: float) -> Tuple[str, Tuple[int,int,int]]:
    """Map a confidence score to (label, BGR_color)."""
    for thr, lbl, col in _CONF_TABLE:
        if c >= thr:
            return lbl, col
    return "POOR", (80, 80, 80)


# ---------------------------------------------------------------------------
# Per-factor score dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceBreakdown:
    """
    Explains exactly how the overall confidence score was computed.
    Shown in the operator interface for transparency / debugging.
    """
    overall:          float    # final fused score [0, 1]
    level:            str      # HIGH / MEDIUM / LOW / POOR
    color_bgr:        Tuple[int, int, int]

    # Individual factor scores [0, 1]
    score_signal:     float    # from RSSI / signal strength
    score_kf_cov:     float    # from KF position uncertainty
    score_nis:        float    # from NIS filter consistency check
    score_track_age:  float    # from number of observations
    score_geometry:   float    # from angle-off-boresight

    # Weights used (for display)
    weight_signal:    float
    weight_kf_cov:    float
    weight_nis:       float
    weight_track_age: float
    weight_geometry:  float

    # Smoothed vs raw
    raw_confidence:   float
    smoothed_confidence: float

    # Per-factor labels for display
    def factor_lines(self):
        """Returns list of (label, score) tuples for overlay display."""
        return [
            ("Signal",   self.score_signal),
            ("KF Cov",   self.score_kf_cov),
            ("Filter",   self.score_nis),
            ("Track",    self.score_track_age),
            ("Geometry", self.score_geometry),
        ]


# ---------------------------------------------------------------------------
# Confidence engine
# ---------------------------------------------------------------------------

class ConfidenceEngine:
    """
    Fuses signal, Kalman, and geometry data into a single confidence score.

    Parameters
    ----------
    weights            : dict of {factor: weight} — default tuning shown below
    smoothing_alpha    : EMA coefficient [0,1], higher = faster response
    kf_std_ref_m       : KF std deviation that maps to score = 0.5 (metres)
    max_track_frames   : track age at which score_track_age saturates to 1.0
    max_angle_deg      : angle off boresight at which geometry score = 0.0
    """

    DEFAULT_WEIGHTS = {
        "signal":    0.35,   # most important — directly measures signal quality
        "kf_cov":    0.25,   # how certain the Kalman filter is
        "nis":       0.15,   # is the filter behaving consistently
        "track_age": 0.15,   # how long have we been tracking this source
        "geometry":  0.10,   # angle off optical axis (less important for RF)
    }

    def __init__(
        self,
        weights: dict = None,
        smoothing_alpha: float = 0.25,
        kf_std_ref_m:    float = 8.0,
        max_track_frames: int  = 60,
        max_angle_deg:   float = 40.0,
    ):
        self.w = weights or self.DEFAULT_WEIGHTS.copy()
        self._alpha       = smoothing_alpha
        self._kf_ref      = kf_std_ref_m
        self._max_frames  = max_track_frames
        self._max_angle   = max_angle_deg

        # Per-source state
        self._smoothed: dict = {}       # source_id -> smoothed confidence
        self._frame_count: dict = {}    # source_id -> observation count

    # ----------------------------------------------------------------- public

    def compute(
        self,
        source_id:   str,
        signal:      SignalMeasurement,
        kf_estimate: KalmanEstimate,
        fov_margin_h: float = 0.0,   # from ProjectionResultV2.fov_margin_h
    ) -> ConfidenceBreakdown:
        """
        Compute and return full confidence breakdown for one source.

        Parameters
        ----------
        source_id    : unique RF source identifier (e.g. "MOTO_01")
        signal       : SignalMeasurement from RFSignalModel
        kf_estimate  : KalmanEstimate from RFSourceKalmanFilter
        fov_margin_h : horizontal FOV margin [0-1], 0=centre, 1=frame edge
        """
        # Track observation count
        self._frame_count[source_id] = self._frame_count.get(source_id, 0) + 1
        n_obs = self._frame_count[source_id]

        # ── Factor 1: Signal strength ──────────────────────────────────
        # Sigmoid curve: ss=0.5 -> score=0.5; ss=0.75 -> score~0.88
        ss = signal.signal_strength
        s_signal = self._sigmoid(ss, centre=0.45, steepness=6.0)

        # ── Factor 2: Kalman covariance ────────────────────────────────
        # KF std = kf_ref_m -> score = 0.5; lower std -> higher score
        kf_std = kf_estimate.position_uncertainty_m
        # Inverse sigmoid: small uncertainty = high score
        s_kf = self._sigmoid_inv(kf_std, centre=self._kf_ref, steepness=0.25)

        # ── Factor 3: NIS filter consistency ──────────────────────────
        # For 2-D measurement, NIS should be chi2(2) distributed.
        # 95% confidence interval: [0.05, 5.99]
        # Score is high when NIS is near the expected value of 2.0
        nis = kf_estimate.nis
        # Score = 1 at NIS=2, drops as NIS moves away from 2
        nis_deviation = abs(nis - 2.0) / 2.0   # 0 at ideal, 1 at NIS=0 or 4
        s_nis = max(0.0, 1.0 - nis_deviation ** 0.6)
        # Penalise heavily if NIS > 10 (filter divergence)
        if nis > 10.0:
            s_nis *= 0.3
        s_nis = float(np.clip(s_nis, 0.0, 1.0))

        # ── Factor 4: Track age ────────────────────────────────────────
        # Grows from 0 to 1 as observation count reaches max_track_frames
        s_track = min(1.0, n_obs / max(self._max_frames, 1))
        # Sqrt curve: reaches 0.5 at 25% of max frames (fast early growth)
        s_track = math.sqrt(s_track)

        # ── Factor 5: Geometry / angle off boresight ──────────────────
        # fov_margin_h: 0 = dead centre, 1 = at edge of FOV
        # For RF projection, extreme angles reduce confidence
        angle_fraction = float(np.clip(fov_margin_h, 0.0, 1.0))
        s_geom = 1.0 - 0.6 * (angle_fraction ** 2)   # quadratic falloff

        # ── Weighted average ───────────────────────────────────────────
        w = self.w
        total_w = sum(w.values())
        raw_conf = (
            w["signal"]    * s_signal +
            w["kf_cov"]    * s_kf     +
            w["nis"]       * s_nis    +
            w["track_age"] * s_track  +
            w["geometry"]  * s_geom
        ) / total_w
        raw_conf = float(np.clip(raw_conf, 0.0, 1.0))

        # ── Exponential moving average smoothing ───────────────────────
        prev = self._smoothed.get(source_id, raw_conf)
        smoothed = self._alpha * raw_conf + (1.0 - self._alpha) * prev
        self._smoothed[source_id] = smoothed

        level, color = confidence_to_level(smoothed)

        return ConfidenceBreakdown(
            overall=round(smoothed, 4),
            level=level,
            color_bgr=color,
            score_signal=round(s_signal, 3),
            score_kf_cov=round(s_kf, 3),
            score_nis=round(s_nis, 3),
            score_track_age=round(s_track, 3),
            score_geometry=round(s_geom, 3),
            weight_signal=w["signal"],
            weight_kf_cov=w["kf_cov"],
            weight_nis=w["nis"],
            weight_track_age=w["track_age"],
            weight_geometry=w["geometry"],
            raw_confidence=round(raw_conf, 4),
            smoothed_confidence=round(smoothed, 4),
        )

    def reset(self, source_id: str = None):
        """Reset state for one source (or all if source_id is None)."""
        if source_id:
            self._smoothed.pop(source_id, None)
            self._frame_count.pop(source_id, None)
        else:
            self._smoothed.clear()
            self._frame_count.clear()

    def get_smoothed(self, source_id: str) -> float:
        """Return current smoothed confidence without updating."""
        return self._smoothed.get(source_id, 0.0)

    # ---------------------------------------------------------------- private

    @staticmethod
    def _sigmoid(x: float, centre: float, steepness: float) -> float:
        """Logistic sigmoid: maps x -> [0,1], centred at centre."""
        return 1.0 / (1.0 + math.exp(-steepness * (x - centre)))

    @staticmethod
    def _sigmoid_inv(x: float, centre: float, steepness: float) -> float:
        """Inverse sigmoid: maps x -> [0,1], high x -> low score."""
        return 1.0 / (1.0 + math.exp(steepness * (x - centre)))
