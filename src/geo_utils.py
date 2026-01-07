"""
Geographic utility functions for earthquake analysis.
"""

import numpy as np

try:
    from .config import EARTH_RADIUS_KM
except ImportError:
    from config import EARTH_RADIUS_KM


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point (degrees)
    lat2, lon2 : float
        Latitude and longitude of second point (degrees)

    Returns
    -------
    float
        Distance in kilometers
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def point_to_line_distance(point_lat, point_lon, line_lat1, line_lon1, line_lat2, line_lon2):
    """
    Calculate minimum distance from a point to a line segment.
    Uses simplified approach: minimum of distances to endpoints and midpoint.

    Parameters
    ----------
    point_lat, point_lon : float
        Point coordinates (degrees)
    line_lat1, line_lon1 : float
        Line segment start coordinates (degrees)
    line_lat2, line_lon2 : float
        Line segment end coordinates (degrees)

    Returns
    -------
    float
        Minimum distance in kilometers
    """
    d1 = haversine_distance(point_lat, point_lon, line_lat1, line_lon1)
    d2 = haversine_distance(point_lat, point_lon, line_lat2, line_lon2)
    mid_lat = (line_lat1 + line_lat2) / 2
    mid_lon = (line_lon1 + line_lon2) / 2
    d_mid = haversine_distance(point_lat, point_lon, mid_lat, mid_lon)
    return min(d1, d2, d_mid)


def project_point_to_line(point_lat, point_lon, line_lat1, line_lon1, line_lat2, line_lon2):
    """
    Project a point onto a line segment and return the projection parameter.

    Parameters
    ----------
    point_lat, point_lon : float
        Point coordinates (degrees)
    line_lat1, line_lon1 : float
        Line segment start coordinates (degrees)
    line_lat2, line_lon2 : float
        Line segment end coordinates (degrees)

    Returns
    -------
    tuple
        (t, projected_lat, projected_lon, distance)
        t: projection parameter (0 = start, 1 = end)
        projected_lat, projected_lon: projected point coordinates
        distance: distance from point to projected point
    """
    # Convert to simple Cartesian approximation (valid for small areas)
    dx = line_lon2 - line_lon1
    dy = line_lat2 - line_lat1

    if dx == 0 and dy == 0:
        return 0, line_lat1, line_lon1, haversine_distance(point_lat, point_lon, line_lat1, line_lon1)

    # Calculate projection parameter
    t = ((point_lon - line_lon1) * dx + (point_lat - line_lat1) * dy) / (dx**2 + dy**2)
    t = max(0, min(1, t))  # Clamp to [0, 1]

    # Calculate projected point
    proj_lon = line_lon1 + t * dx
    proj_lat = line_lat1 + t * dy

    # Calculate distance
    dist = haversine_distance(point_lat, point_lon, proj_lat, proj_lon)

    return t, proj_lat, proj_lon, dist


def calculate_along_fault_distance(fault_lat1, fault_lon1, fault_lat2, fault_lon2, t):
    """
    Calculate along-fault distance from start point given projection parameter.

    Parameters
    ----------
    fault_lat1, fault_lon1 : float
        Fault segment start coordinates (degrees)
    fault_lat2, fault_lon2 : float
        Fault segment end coordinates (degrees)
    t : float
        Projection parameter (0 = start, 1 = end)

    Returns
    -------
    float
        Along-fault distance in kilometers
    """
    total_length = haversine_distance(fault_lat1, fault_lon1, fault_lat2, fault_lon2)
    return t * total_length


def get_fault_segments(fault_lat1, fault_lon1, fault_lat2, fault_lon2, segment_length_km=10):
    """
    Divide a fault line into segments of specified length.

    Parameters
    ----------
    fault_lat1, fault_lon1 : float
        Fault start coordinates (degrees)
    fault_lat2, fault_lon2 : float
        Fault end coordinates (degrees)
    segment_length_km : float
        Desired segment length in kilometers

    Returns
    -------
    list of tuples
        List of (start_lat, start_lon, end_lat, end_lon, distance_from_start) for each segment
    """
    total_length = haversine_distance(fault_lat1, fault_lon1, fault_lat2, fault_lon2)
    n_segments = max(1, int(np.ceil(total_length / segment_length_km)))

    segments = []
    for i in range(n_segments):
        t_start = i / n_segments
        t_end = (i + 1) / n_segments

        seg_lat1 = fault_lat1 + t_start * (fault_lat2 - fault_lat1)
        seg_lon1 = fault_lon1 + t_start * (fault_lon2 - fault_lon1)
        seg_lat2 = fault_lat1 + t_end * (fault_lat2 - fault_lat1)
        seg_lon2 = fault_lon1 + t_end * (fault_lon2 - fault_lon1)

        distance_from_start = t_start * total_length

        segments.append((seg_lat1, seg_lon1, seg_lat2, seg_lon2, distance_from_start))

    return segments


def create_turkey_grid(resolution=0.1):
    """
    Create a lat/lon grid covering Turkey.

    Parameters
    ----------
    resolution : float
        Grid resolution in degrees

    Returns
    -------
    tuple
        (lon_grid, lat_grid) meshgrids
    """
    try:
        from .config import TURKEY_BOUNDS
    except ImportError:
        from config import TURKEY_BOUNDS

    lons = np.arange(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'], resolution)
    lats = np.arange(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'], resolution)

    return np.meshgrid(lons, lats)
