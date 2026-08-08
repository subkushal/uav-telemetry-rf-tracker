# =============================================================================
# coordinate_projector.py
# Drone Telemetry Matrix - 3D Pixel-to-GPS Raycasting Engine
# =============================================================================

import math

class CoordinateProjector:
    def __init__(self, horizontal_fov_deg=60.0, vertical_fov_deg=45.0):
        """
        horizontal_fov_deg: The camera's horizontal field of view (check drone specs).
        vertical_fov_deg: The camera's vertical field of view.
        """
        self.hfov = math.radians(horizontal_fov_deg)
        self.vfov = math.radians(vertical_fov_deg)

    def project_pixel_to_gps(self, x_pixel, y_pixel, frame_w, frame_h, telemetry):
        """
        Projects a video pixel coordinate down to a real-world GPS position.
        
        telemetry dict must contain:
            - "drone_lat": Current latitude of the aircraft
            - "drone_lon": Current longitude of the aircraft
            - "altitude_m": Relative altitude above ground level (AGL) in meters
            - "gimbal_pitch_deg": Camera angle down from horizon (0 = horizontal, -90 = straight down)
            - "drone_yaw_deg": The drone's true compass heading (0 = North, 90 = East)
        """
        drone_lat = telemetry["drone_lat"]
        drone_lon = telemetry["drone_lon"]
        alt = telemetry["altitude_m"]
        
        # Convert critical angles to radians
        gimbal_pitch = math.radians(telemetry["gimbal_pitch_deg"])
        drone_yaw = math.radians(telemetry["drone_yaw_deg"])

        # ─── STEP 1: CONVERT PIXEL TO CAMERA ANGLE OFFSETS ──────────────────
        # Find out how many degrees left/right and up/down the target is from the screen center
        normalized_x = (x_pixel / frame_w) - 0.5  # Range: -0.5 (left edge) to +0.5 (right edge)
        normalized_y = 0.5 - (y_pixel / frame_h)  # Range: -0.5 (bottom edge) to +0.5 (top edge)

        angle_offset_x = normalized_x * self.hfov
        angle_offset_y = normalized_y * self.vfov

        # ─── STEP 2: CALCULATE LOCAL PHYSICAL RAYCAST DISTANCES ─────────────
        # Combined look angle including gimbal inclination and pixel position
        total_pitch = gimbal_pitch + angle_offset_y

        # Prevent division by zero if looking exactly parallel to the horizon
        if abs(total_pitch) < 0.05:
            total_pitch = -0.05 if total_pitch <= 0 else 0.05

        # Trigonometric ground projection math (Flat Earth approximation for local radius)
        # Distance along the ground plane from directly beneath the drone to the target point
        ground_distance_forward = alt / math.tan(abs(total_pitch))
        ground_distance_sideways = ground_distance_forward * math.tan(angle_offset_x)

        # ─── STEP 3: ROTATE RAY DIRECTION TO TRUE COMPASS HEADING ───────────
        # Combine local forward/sideways vectors with the drone's current compass yaw
        # This converts aircraft coordinates into true planetary alignments (North/East)
        total_yaw = drone_yaw + math.atan2(ground_distance_sideways, ground_distance_forward)
        true_ground_range = math.sqrt(ground_distance_forward**2 + ground_distance_sideways**2)

        meters_north = true_ground_range * math.cos(total_yaw)
        meters_east = true_ground_range * math.sin(total_yaw)

        # ─── STEP 4: TRANSLATE METERS TO GLOBAL LATITUDE / LONGITUDE ────────
        # Earth scaling parameters
        meters_per_degree_lat = 111132.0
        # Longitude scaling collapses dynamically as you move closer to the poles
        meters_per_degree_lon = 111412.0 * math.cos(math.radians(drone_lat))

        target_lat = drone_lat + (meters_north / meters_per_degree_lat)
        target_lon = drone_lon + (meters_east / meters_per_degree_lon)

        return {
            "lat": round(target_lat, 6),
            "lon": round(target_lon, 6),
            "distance_from_drone_m": round(true_ground_range, 1)
        }

# ─── MATHEMATICAL INTEGRATION VERIFICATION TEST ──────────────────────────────
if __name__ == "__main__":
    projector = CoordinateProjector(horizontal_fov_deg=60.0, vertical_fov_deg=45.0)

    # Simulated flight data received from drone's autopilot system
    mock_telemetry = {
        "drone_lat": 32.894000,
        "drone_lon": 74.762000,
        "altitude_m": 20.0,             # Drone is flying at 20 meters high
        "gimbal_pitch_deg": -45.0,       # Camera is tilted 45 degrees downward
        "drone_yaw_deg": 0.0             # Drone is facing straight North
    }

    # Assume a standard 1080p video stream, target found slightly off-center
    result = projector.project_pixel_to_gps(
        x_pixel=700, y_pixel=540, 
        frame_w=1920, frame_h=1080, 
        telemetry=mock_telemetry
    )

    print("\n[COORDINATE PROJECTOR] Raycast Simulation Results:")
    print(f"  Estimated Target Position -> LAT: {result['lat']} | LON: {result['lon']}")
    print(f"  Physical Ground Separation -> {result['distance_from_drone_m']} meters away from aircraft footprint.")