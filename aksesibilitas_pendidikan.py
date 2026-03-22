import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium, folium_static
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
import warnings
import time
import math
from folium.plugins import MiniMap, Fullscreen
from shapely.ops import unary_union, linemerge
from pyproj import CRS, Transformer
from scipy.spatial import ConvexHull
from collections import defaultdict
from shapely import concave_hull, convex_hull

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# INISIALISASI SESSION STATE
# ============================================================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'map_created' not in st.session_state:
    st.session_state.map_created = False
if 'accessibility_zones' not in st.session_state:
    st.session_state.accessibility_zones = None
if 'education_facilities' not in st.session_state:
    st.session_state.education_facilities = None
if 'reachable_edges' not in st.session_state:
    st.session_state.reachable_edges = None
if 'last_location' not in st.session_state:
    st.session_state.last_location = None
if 'analysis_params' not in st.session_state:
    st.session_state.analysis_params = None
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = 0
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False
if 'network_cache' not in st.session_state:
    st.session_state.network_cache = {}

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Aksesbilitas Fasilitas Pendidikan dengan Network Coverage",
    page_icon="🎓📍",
    layout="wide"
)

# Judul aplikasi
st.title("🎓📍 Analisis Spasial Aksesbilitas Fasilitas Pendidikan")
st.markdown("**Analisis zona Aksesbilitas dengan dua metode coverage dari titik analisis**")

# ============================================================
# PEMETAAN JENIS FASILITAS PENDIDIKAN
# ============================================================
EDUCATION_FACILITY_TYPES = {
    'SD': {
        'label': 'Sekolah Dasar (SD)',
        'tags': [
            {'amenity': 'school'},
            {'isced:level': '1'},
        ],
        'keywords': ['sekolah dasar', 'sd ', 'elementary', 'primary school', 'madrasah ibtidaiyah', 'mi '],
        'icon': 'graduation-cap',
        'color': 'blue'
    },
    'SMP': {
        'label': 'Sekolah Menengah Pertama (SMP)',
        'tags': [
            {'amenity': 'school'},
            {'isced:level': '2'},
        ],
        'keywords': ['sekolah menengah pertama', 'smp ', 'junior high', 'madrasah tsanawiyah', 'mts '],
        'icon': 'book',
        'color': 'green'
    },
    'SMA': {
        'label': 'Sekolah Menengah Atas (SMA)',
        'tags': [
            {'amenity': 'school'},
            {'isced:level': '3'},
        ],
        'keywords': ['sekolah menengah atas', 'sma ', 'smk ', 'senior high', 'madrasah aliyah', 'ma '],
        'icon': 'certificate',
        'color': 'orange'
    },
    'Universitas': {
        'label': 'Universitas',
        'tags': [
            {'amenity': 'university'},
        ],
        'keywords': ['universitas', 'university', 'univ '],
        'icon': 'university',
        'color': 'red'
    },
    'Sekolah Tinggi': {
        'label': 'Sekolah Tinggi / Institut / Akademi',
        'tags': [
            {'amenity': 'college'},
        ],
        'keywords': ['sekolah tinggi', 'institut ', 'akademi ', 'politeknik', 'college', 'stie ', 'stik ', 'stia '],
        'icon': 'building',
        'color': 'purple'
    },
    'Lembaga Kursus': {
        'label': 'Lembaga Kursus / Pelatihan',
        'tags': [
            {'amenity': 'language_school'},
            {'amenity': 'driving_school'},
            {'amenity': 'music_school'},
            {'amenity': 'training'},
        ],
        'keywords': ['kursus', 'pelatihan', 'bimbel', 'bimbingan belajar', 'les ', 'training center', 'lkp '],
        'icon': 'pencil',
        'color': 'cadetblue'
    }
}

def classify_education_facility(row):
    """Mengklasifikasikan fasilitas pendidikan berdasarkan nama dan tag OSM"""
    name = str(row.get('name', '')).lower()
    amenity = str(row.get('amenity', '')).lower()
    isced = str(row.get('isced:level', '')).lower()

    # Klasifikasi berdasarkan isced:level
    if isced == '1':
        return 'SD'
    if isced == '2':
        return 'SMP'
    if isced == '3':
        return 'SMA'

    # Klasifikasi berdasarkan amenity
    if amenity == 'university':
        return 'Universitas'
    if amenity == 'college':
        return 'Sekolah Tinggi'
    if amenity in ['language_school', 'driving_school', 'music_school', 'training']:
        return 'Lembaga Kursus'

    # Klasifikasi berdasarkan nama
    for ftype, info in EDUCATION_FACILITY_TYPES.items():
        for kw in info['keywords']:
            if kw in name:
                return ftype

    # Default untuk amenity=school
    if amenity == 'school':
        return 'SD/SMP/SMA'

    return 'Lainnya'

# ============================================================
# FUNGSI KONVERSI MODA TRANSPORTASI
# ============================================================
def convert_transport_mode(mode_bahasa):
    conversion_map = {
        "jalan kaki": "walk",
        "sepeda": "bike",
        "mobil/motor": "drive"
    }
    return conversion_map.get(mode_bahasa, "walk")

def get_default_speed(mode_bahasa):
    speed_map = {
        "jalan kaki": 5.0,
        "sepeda": 15.0,
        "mobil/motor": 40.00
    }
    return speed_map.get(mode_bahasa, 5.0)

# ============================================================
# FUNGSI UTILITAS UMUM
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_education_facilities(bbox):
    """Mendapatkan fasilitas pendidikan dari OSM dalam bounding box"""
    north, south, east, west = bbox

    try:
        all_tags = [
            {'amenity': 'school'},
            {'amenity': 'university'},
            {'amenity': 'college'},
            {'amenity': 'language_school'},
            {'amenity': 'driving_school'},
            {'amenity': 'music_school'},
            {'amenity': 'kindergarten'},
        ]

        all_facilities = []

        for tags in all_tags:
            try:
                facilities = ox.features_from_bbox(
                    north, south, east, west,
                    tags=tags
                )
                if not facilities.empty:
                    all_facilities.append(facilities)
            except:
                continue

        if all_facilities:
            combined = gpd.GeoDataFrame(pd.concat(all_facilities, ignore_index=True))
            if 'geometry' in combined.columns:
                combined = combined.drop_duplicates(subset=['geometry'])

            if 'name' not in combined.columns:
                combined['name'] = 'Fasilitas Pendidikan'

            # Tambahkan klasifikasi
            combined['edu_type'] = combined.apply(classify_education_facility, axis=1)

            return combined
        else:
            try:
                all_amenities = ox.features_from_bbox(
                    north, south, east, west,
                    tags={'amenity': True}
                )
                edu_amenities = ['school', 'university', 'college', 'language_school',
                                 'driving_school', 'music_school', 'kindergarten', 'training']
                mask = all_amenities['amenity'].astype(str).str.lower().isin(edu_amenities)
                filtered = all_amenities[mask].copy()

                if 'name' not in filtered.columns:
                    filtered['name'] = 'Fasilitas Pendidikan'

                filtered['edu_type'] = filtered.apply(classify_education_facility, axis=1)
                return filtered
            except:
                return gpd.GeoDataFrame()

    except Exception as e:
        st.warning(f"Tidak bisa mengambil data fasilitas: {str(e)}")
        return gpd.GeoDataFrame()

# ============================================================
# FUNGSI NETWORK ANALYSIS
# ============================================================
def get_network_from_point(location_point, network_type, radius):
    try:
        cache_key = f"{location_point}_{network_type}_{radius}"

        if cache_key in st.session_state.network_cache:
            return st.session_state.network_cache[cache_key]

        graph = ox.graph_from_point(
            location_point,
            dist=radius,
            network_type=network_type,
            simplify=True,
            truncate_by_edge=True,
            retain_all=True
        )

        if len(graph.nodes()) == 0:
            st.error(f"❌ Tidak ada jaringan jalan ditemukan dari titik {location_point}")
            return None

        graph_proj = ox.project_graph(graph)
        st.session_state.network_cache[cache_key] = graph_proj
        return graph_proj

    except Exception as e:
        st.error(f"❌ Gagal mengambil jaringan dari titik: {str(e)}")
        return None

def find_start_node_from_point(graph_proj, location_point):
    try:
        lat, lon = location_point
        start_node = ox.distance.nearest_nodes(graph_proj, lon, lat)
        node_data = graph_proj.nodes[start_node]
        start_coords = (node_data['x'], node_data['y'])
        transformer = Transformer.from_crs(graph_proj.graph['crs'], 'EPSG:4326', always_xy=True)
        lon_wgs, lat_wgs = transformer.transform(start_coords[0], start_coords[1])
        start_wgs84 = (lat_wgs, lon_wgs)
        return start_node, start_coords, start_wgs84
    except Exception as e:
        st.error(f"❌ Gagal menemukan node dari titik: {str(e)}")
        return None, None, None

# ============================================================
# FUNGSI CONCAVE HULL
# ============================================================
def calculate_alpha_shape(points, alpha=None):
    try:
        if len(points) < 3:
            return None
        points_array = np.array(points)
        from shapely.geometry import MultiPoint
        multipoint = MultiPoint([(x, y) for x, y in points_array])

        for ratio in [0.05, 0.1, 0.2, 0.3]:
            try:
                concave_shape = concave_hull(multipoint, ratio=ratio)
                if concave_shape and not concave_shape.is_empty and concave_shape.geom_type == 'Polygon':
                    return concave_shape
            except:
                continue

        convex_shape = convex_hull(multipoint)
        if convex_shape and not convex_shape.is_empty:
            return convex_shape
        else:
            hull = ConvexHull(points_array)
            return Polygon(points_array[hull.vertices])

    except Exception as e:
        st.warning(f"Error dalam alpha shape: {str(e)}")
        try:
            points_array = np.array(points)
            hull = ConvexHull(points_array)
            return Polygon(points_array[hull.vertices])
        except:
            return None

def optimize_alpha_value(points):
    try:
        if len(points) < 3:
            return None, 0.1
        return calculate_alpha_shape(points), 0.1
    except:
        return None, 0.1

# ============================================================
# METODE 1: NETWORK SERVICE AREA
# ============================================================
def calculate_network_service_area(graph_proj, start_node, start_coords, max_distance,
                                   service_buffer=100):
    try:
        distances = nx.single_source_dijkstra_path_length(
            graph_proj, start_node, weight='length', cutoff=max_distance
        )
        if not distances:
            return None, [], {}

        reachable_nodes = []
        reachable_edges = []
        edge_distances = {}
        reachable_node_points = []
        all_edge_points = []

        for u, v, data in graph_proj.edges(data=True):
            u_dist = distances.get(u, float('inf'))
            v_dist = distances.get(v, float('inf'))

            if u_dist <= max_distance or v_dist <= max_distance:
                if u not in reachable_nodes:
                    reachable_nodes.append(u)
                    reachable_node_points.append((graph_proj.nodes[u]['x'], graph_proj.nodes[u]['y']))
                if v not in reachable_nodes:
                    reachable_nodes.append(v)
                    reachable_node_points.append((graph_proj.nodes[v]['x'], graph_proj.nodes[v]['y']))

                if 'geometry' in data:
                    edge_geom = data['geometry']
                    if hasattr(edge_geom, 'coords'):
                        all_edge_points.extend(list(edge_geom.coords))
                else:
                    edge_geom = LineString([
                        (graph_proj.nodes[u]['x'], graph_proj.nodes[u]['y']),
                        (graph_proj.nodes[v]['x'], graph_proj.nodes[v]['y'])
                    ])
                    all_edge_points.extend(list(edge_geom.coords))

                reachable_edges.append(edge_geom)
                edge_distances[len(reachable_edges)-1] = min(u_dist, v_dist)

        if not reachable_edges:
            return None, [], {}

        all_points = reachable_node_points + all_edge_points
        seen = set()
        unique_points = []
        for point in all_points:
            pt = tuple(point)
            if pt not in seen:
                seen.add(pt)
                unique_points.append(point)

        service_area = None

        if len(unique_points) >= 3:
            try:
                concave_shape, _ = optimize_alpha_value(unique_points)
                if concave_shape and not concave_shape.is_empty:
                    service_area = concave_shape
                    st.info("✅ Menggunakan Concave Hull")
                else:
                    pts = np.array(unique_points)
                    hull = ConvexHull(pts)
                    service_area = Polygon(pts[hull.vertices])
                    st.info("⚠️ Menggunakan Convex Hull (fallback)")
            except Exception as hull_error:
                st.warning(f"Error dalam concave hull: {hull_error}")

        if service_area is None or service_area.is_empty:
            try:
                buffered_polygons = []
                for i, edge in enumerate(reachable_edges):
                    try:
                        edge_dist = edge_distances.get(i, max_distance)
                        buffer_size = max(10, min(service_buffer * (1 - edge_dist/max_distance), service_buffer * 1.5))
                        buffered_polygons.append(edge.buffer(buffer_size, resolution=8))
                    except:
                        continue
                if buffered_polygons:
                    service_area = unary_union(buffered_polygons)
            except Exception as buffer_error:
                st.warning(f"Buffer fallback error: {buffer_error}")

        if service_area is None or service_area.is_empty:
            service_area = Point(start_coords[0], start_coords[1]).buffer(max_distance * 0.8)
            st.info("ℹ️ Menggunakan buffer dari titik awal")

        try:
            if service_area and not service_area.is_empty:
                service_area = service_area.buffer(service_buffer, resolution=16)
                start_point = Point(start_coords[0], start_coords[1])
                if not service_area.contains(start_point):
                    service_area = unary_union([service_area, start_point.buffer(service_buffer * 2)])
                service_area = service_area.simplify(service_buffer/2, preserve_topology=True)
        except Exception as e:
            st.warning(f"Error dalam smoothing service area: {e}")

        return service_area, reachable_edges, edge_distances

    except Exception as e:
        st.error(f"Error dalam network service area: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, [], {}

# ============================================================
# METODE 2: BUFFER DARI TITIK
# ============================================================
def calculate_buffer_coverage(location_point, max_distance, shape='Lingkaran'):
    try:
        lat, lon = location_point
        buffer_deg_lat = max_distance / 111320
        buffer_deg_lon = max_distance / (111320 * math.cos(math.radians(lat)))

        if shape == 'Lingkaran':
            polygon = Point(lon, lat).buffer(max_distance / 111000, resolution=32)
        elif shape == 'Persegi':
            polygon = box(lon - buffer_deg_lon, lat - buffer_deg_lat,
                          lon + buffer_deg_lon, lat + buffer_deg_lat)
        elif shape == 'Kapsul':
            c1 = Point(lon - buffer_deg_lon/2, lat).buffer(buffer_deg_lat/2, resolution=16)
            c2 = Point(lon + buffer_deg_lon/2, lat).buffer(buffer_deg_lat/2, resolution=16)
            rect = box(lon - buffer_deg_lon/2, lat - buffer_deg_lat/3,
                       lon + buffer_deg_lon/2, lat + buffer_deg_lat/3)
            polygon = unary_union([c1, c2, rect])
        else:
            polygon = Point(lon, lat).buffer(max_distance / 111000, resolution=16)

        if polygon.is_empty or not polygon.is_valid:
            polygon = Point(lon, lat).buffer(max_distance / 111000, resolution=16)

        return polygon

    except Exception as e:
        st.error(f"Error dalam buffer coverage: {str(e)}")
        return Point(location_point[1], location_point[0]).buffer(0.01)

# ============================================================
# FUNGSI NETWORK COVERAGE UTAMA
# ============================================================
def calculate_network_coverage_from_point(graph_proj, start_node, start_coords,
                                          max_distance, location_point, method="Service Area",
                                          **kwargs):
    try:
        if method == "Buffer dari Titik":
            shape = kwargs.get('buffer_shape', 'Lingkaran')
            return calculate_buffer_coverage(location_point, max_distance, shape), []

        if method == "Service Area":
            service_buffer = kwargs.get('service_buffer', 100)
            coverage_polygon, reachable_edges, edge_distances = calculate_network_service_area(
                graph_proj, start_node, start_coords, max_distance, service_buffer=service_buffer
            )
            if coverage_polygon is None:
                return calculate_buffer_coverage(location_point, max_distance,
                                                  kwargs.get('buffer_shape', 'Lingkaran')), []
            return coverage_polygon, reachable_edges

        return calculate_buffer_coverage(location_point, max_distance,
                                          kwargs.get('buffer_shape', 'Lingkaran')), []

    except Exception as e:
        st.error(f"Error dalam calculate_network_coverage: {str(e)}")
        return calculate_buffer_coverage(location_point, max_distance,
                                          kwargs.get('buffer_shape', 'Lingkaran')), []

# ============================================================
# KONVERSI KOORDINAT
# ============================================================
def convert_polygon_to_wgs84(polygon, source_crs, location_point):
    try:
        transformer = Transformer.from_crs(source_crs, 'EPSG:4326', always_xy=True)

        if hasattr(polygon, 'exterior'):
            exterior_coords = list(polygon.exterior.coords)
            wgs84_coords = [transformer.transform(x, y) for x, y in exterior_coords]
            return Polygon(wgs84_coords)
        else:
            bounds = polygon.bounds
            minx, miny, maxx, maxy = bounds
            corners = [
                transformer.transform(minx, miny),
                transformer.transform(maxx, miny),
                transformer.transform(maxx, maxy),
                transformer.transform(minx, maxy)
            ]
            return Polygon([(lon, lat) for lon, lat in corners])

    except Exception as e:
        st.warning(f"Konversi polygon gagal: {e}. Menggunakan fallback buffer.")
        lat, lon = location_point
        return Point(lon, lat).buffer(0.01)

# ============================================================
# FUNGSI ANALISIS UTAMA
# ============================================================
def analyze_from_point_main(location_point, network_type, speed_kmh, radius_m, time_limits_min,
                            method="Service Area", **kwargs):
    current_params = {
        'location': location_point,
        'network_type': network_type,
        'speed': speed_kmh,
        'radius': radius_m,
        'time_limits': tuple(sorted(time_limits_min)),
        'method': method,
        **kwargs
    }

    if (st.session_state.analysis_params == current_params and
            st.session_state.analysis_results is not None):
        st.info("✅ Menggunakan hasil analisis sebelumnya...")
        return st.session_state.analysis_results

    try:
        st.session_state.analysis_in_progress = True
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state.last_location = location_point

        if method != "Buffer dari Titik":
            status_text.text("📍 Mendapatkan jaringan dari titik analisis...")
            graph_proj = get_network_from_point(location_point, network_type, radius_m)
            if graph_proj is None:
                st.session_state.analysis_in_progress = False
                return None, None, None, None
            progress_bar.progress(25)

            status_text.text("📍 Mencari node awal dari titik analisis...")
            start_node, start_coords, start_wgs84 = find_start_node_from_point(graph_proj, location_point)
            if start_node is None:
                st.session_state.analysis_in_progress = False
                return None, None, None, None
            progress_bar.progress(40)
        else:
            graph_proj = None
            start_node = None
            start_coords = None
            start_wgs84 = location_point
            progress_bar.progress(40)

        status_text.text("🎓 Mencari fasilitas pendidikan dari titik analisis...")

        lat, lon = location_point
        R = 6378137
        lat_offset = (radius_m / R) * (180 / math.pi)
        lon_offset = (radius_m / (R * math.cos(math.pi * lat / 180))) * (180 / math.pi)
        bbox = (lat + lat_offset, lat - lat_offset, lon + lon_offset, lon - lon_offset)

        education_facilities = get_education_facilities(bbox)
        progress_bar.progress(50)

        status_text.text("🧮 Menghitung coverage dari titik analisis...")
        accessibility_zones = {}
        reachable_edges_dict = {}
        speed_m_per_min = (speed_kmh * 1000) / 60

        for idx, time_limit in enumerate(sorted(time_limits_min)):
            max_distance = speed_m_per_min * time_limit

            coverage_polygon, reachable_edges = calculate_network_coverage_from_point(
                graph_proj, start_node, start_coords, max_distance, location_point, method, **kwargs
            )

            if coverage_polygon and not coverage_polygon.is_empty:
                area_km2 = coverage_polygon.area / 1000000

                wgs84_polygon = None
                if method == "Buffer dari Titik":
                    wgs84_polygon = coverage_polygon
                else:
                    try:
                        wgs84_polygon = convert_polygon_to_wgs84(
                            coverage_polygon, graph_proj.graph['crs'], location_point
                        )
                    except:
                        wgs84_polygon = Point(lon, lat).buffer(max_distance / 111320)

                accessible_facilities = []
                if not education_facilities.empty and wgs84_polygon:
                    for idx_fac, facility in education_facilities.iterrows():
                        try:
                            if hasattr(facility.geometry, 'within'):
                                if facility.geometry.within(wgs84_polygon):
                                    fac_name = facility.get('name', 'Fasilitas Pendidikan')
                                    fac_amenity = facility.get('amenity', 'Pendidikan')
                                    fac_edu_type = facility.get('edu_type', 'Lainnya')

                                    if hasattr(facility.geometry, 'x'):
                                        fac_lat = facility.geometry.y
                                        fac_lon = facility.geometry.x
                                        distance = haversine_distance(lat, lon, fac_lat, fac_lon)
                                        travel_time = distance / speed_m_per_min

                                        accessible_facilities.append({
                                            'name': str(fac_name),
                                            'amenity': str(fac_amenity),
                                            'edu_type': str(fac_edu_type),
                                            'geometry': facility.geometry,
                                            'distance_m': distance,
                                            'travel_time_min': travel_time,
                                            'coordinates': (fac_lat, fac_lon)
                                        })
                        except:
                            continue

                accessibility_zones[time_limit] = {
                    'geometry': wgs84_polygon,
                    'geometry_projected': coverage_polygon,
                    'max_distance': max_distance,
                    'area_sqkm': area_km2,
                    'calculation_method': method,
                    'accessible_facilities': accessible_facilities,
                    'facilities_count': len(accessible_facilities),
                    'time_limit': time_limit
                }

                reachable_edges_dict[time_limit] = reachable_edges

            progress_bar.progress(50 + int((idx + 1) * 50 / len(time_limits_min)))

        progress_bar.progress(100)
        status_text.text("✅ Analisis dari titik selesai!")
        time.sleep(0.5)
        status_text.empty()

        result = (graph_proj, education_facilities, accessibility_zones, reachable_edges_dict)
        st.session_state.analysis_results = result
        st.session_state.analysis_params = current_params
        st.session_state.accessibility_zones = accessibility_zones
        st.session_state.education_facilities = education_facilities
        st.session_state.reachable_edges = reachable_edges_dict
        st.session_state.last_analysis_time = time.time()
        st.session_state.analysis_in_progress = False

        return result

    except Exception as e:
        st.error(f"❌ Error dalam analisis dari titik: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.session_state.analysis_in_progress = False
        return None, None, None, None

# ============================================================
# FUNGSI PEMBUATAN PETA
# ============================================================
def create_comprehensive_map(location_point, accessibility_zones, education_facilities,
                             reachable_edges_dict=None):
    try:
        m = folium.Map(location=location_point, zoom_start=14,
                       tiles='OpenStreetMap', control_scale=True)

        folium.Marker(
            location=location_point,
            popup='<b>📍 Titik Analisis</b><br>Lokasi awal perhitungan jangkauan',
            tooltip='Titik Analisis',
            icon=folium.Icon(color='red', icon='bullseye', prefix='fa')
        ).add_to(m)

        zone_colors = {
            5: '#E3F2FD',
            10: '#BBDEFB',
            15: '#90CAF9',
            20: '#64B5F6',
            25: '#42A5F5',
            30: '#2196F3',
            45: '#1565C0',
            60: '#0D47A1'
        }

        if reachable_edges_dict:
            for time_limit, reachable_edges in reachable_edges_dict.items():
                if reachable_edges and len(reachable_edges) > 0:
                    edge_color = zone_colors.get(time_limit, '#90CAF9')
                    for edge in reachable_edges:
                        if hasattr(edge, 'coords'):
                            coords = list(edge.coords)
                            if len(coords) >= 2:
                                folium_coords = [[y, x] for x, y in coords]
                                if len(folium_coords) >= 2:
                                    folium.PolyLine(
                                        locations=folium_coords,
                                        color=edge_color,
                                        weight=1,
                                        opacity=0.3,
                                        popup=f'Edge Jaringan ({time_limit} menit)'
                                    ).add_to(m)

        for time_limit, zone_data in accessibility_zones.items():
            color = zone_colors.get(time_limit, '#90CAF9')
            try:
                if 'geometry' in zone_data and zone_data['geometry']:
                    polygon = zone_data['geometry']
                    if hasattr(polygon, 'exterior'):
                        coords = list(polygon.exterior.coords)
                        folium_coords = [[lat, lon] for lon, lat in coords]
                        if len(folium_coords) >= 3:
                            folium.Polygon(
                                locations=folium_coords,
                                color=color,
                                fill=True,
                                fill_color=color,
                                fill_opacity=0.3,
                                weight=2,
                                popup=f'<b>Zona {time_limit} menit</b><br>'
                                      f'Metode: {zone_data.get("calculation_method", "Network")}<br>'
                                      f'Luas: {zone_data["area_sqkm"]:.2f} km²<br>'
                                      f'Jarak maks: {zone_data["max_distance"]:.0f} m<br>'
                                      f'Fasilitas: {zone_data.get("facilities_count", 0)}',
                                tooltip=f'Zona Aksesibilitas {time_limit} menit'
                            ).add_to(m)

                            center_lat = sum(c[0] for c in folium_coords) / len(folium_coords)
                            center_lon = sum(c[1] for c in folium_coords) / len(folium_coords)
                            folium.Marker(
                                location=[center_lat, center_lon],
                                icon=folium.DivIcon(
                                    html=f'<div style="font-size: 10pt; font-weight: bold; color: {color};">{time_limit}m</div>'
                                ),
                                popup=f'Label Zona {time_limit} menit'
                            ).add_to(m)
                    else:
                        folium.Circle(
                            location=location_point,
                            radius=zone_data['max_distance'],
                            color=color, fill=True, fill_color=color, fill_opacity=0.3,
                            popup=f'<b>Zona {time_limit} menit</b><br>Luas: {zone_data["area_sqkm"]:.2f} km²'
                        ).add_to(m)
            except Exception as e:
                st.warning(f"Error menambahkan zona {time_limit} menit: {str(e)}")
                folium.Circle(
                    location=location_point,
                    radius=zone_data.get('max_distance', 1000),
                    color=color, fill=True, fill_color=color, fill_opacity=0.3,
                    popup=f'Zona {time_limit} menit (fallback)'
                ).add_to(m)

        # Marker fasilitas pendidikan
        edu_marker_config = {
            'SD': ('blue', 'graduation-cap'),
            'SMP': ('green', 'book'),
            'SMA': ('orange', 'certificate'),
            'SD/SMP/SMA': ('lightblue', 'school'),
            'Universitas': ('red', 'university'),
            'Sekolah Tinggi': ('purple', 'building'),
            'Lembaga Kursus': ('cadetblue', 'pencil'),
            'Lainnya': ('gray', 'info-circle'),
        }

        added_facilities = set()

        for time_limit, zone_data in accessibility_zones.items():
            if 'accessible_facilities' in zone_data:
                for facility in zone_data['accessible_facilities']:
                    try:
                        fac_key = f"{facility['coordinates'][0]:.5f},{facility['coordinates'][1]:.5f}"
                        if fac_key in added_facilities:
                            continue
                        added_facilities.add(fac_key)

                        edu_type = facility.get('edu_type', 'Lainnya')
                        color, icon = edu_marker_config.get(edu_type, ('gray', 'info-circle'))

                        popup_content = f"""
                        <div style="min-width: 200px;">
                            <h4 style="margin-bottom: 5px; color: #2c3e50;">{facility['name']}</h4>
                            <hr style="margin: 5px 0;">
                            <p style="margin: 2px 0;"><b>Jenis:</b> {edu_type}</p>
                            <p style="margin: 2px 0;"><b>Amenity OSM:</b> {facility.get('amenity', '-')}</p>
                            <p style="margin: 2px 0;"><b>Waktu tempuh:</b> {facility['travel_time_min']:.1f} menit</p>
                            <p style="margin: 2px 0;"><b>Jarak:</b> {facility['distance_m']:.0f} m</p>
                            <p style="margin: 2px 0;"><b>Zona:</b> {time_limit} menit</p>
                        </div>
                        """

                        folium.Marker(
                            location=facility['coordinates'],
                            popup=folium.Popup(popup_content, max_width=300),
                            tooltip=f"{facility['name']} — {edu_type} ({facility['travel_time_min']:.1f} menit)",
                            icon=folium.Icon(color=color, icon=icon, prefix='fa')
                        ).add_to(m)
                    except:
                        continue

        folium.LayerControl().add_to(m)
        MiniMap(toggle_display=True).add_to(m)
        Fullscreen().add_to(m)

        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 240px;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:13px; padding: 12px; border-radius: 8px; opacity: 0.95;">
            <b>LEGENDA</b><br>
            <i class="fa fa-bullseye" style="color:red"></i>&nbsp; Titik Analisis<br>
            <i class="fa fa-graduation-cap" style="color:blue"></i>&nbsp; SD<br>
            <i class="fa fa-book" style="color:green"></i>&nbsp; SMP<br>
            <i class="fa fa-certificate" style="color:orange"></i>&nbsp; SMA / SMK<br>
            <i class="fa fa-university" style="color:red"></i>&nbsp; Universitas<br>
            <i class="fa fa-building" style="color:purple"></i>&nbsp; Sekolah Tinggi / Institut<br>
            <i class="fa fa-pencil" style="color:cadetblue"></i>&nbsp; Lembaga Kursus<br>
            <hr style="margin:5px 0;">
            <div style="background:#E3F2FD;width:18px;height:14px;display:inline-block;border:1px solid #ccc;margin-right:4px;"></div>5 menit<br>
            <div style="background:#BBDEFB;width:18px;height:14px;display:inline-block;border:1px solid #ccc;margin-right:4px;"></div>10 menit<br>
            <div style="background:#90CAF9;width:18px;height:14px;display:inline-block;border:1px solid #ccc;margin-right:4px;"></div>15 menit<br>
            <div style="background:#64B5F6;width:18px;height:14px;display:inline-block;border:1px solid #ccc;margin-right:4px;"></div>20 menit<br>
            <div style="background:#42A5F5;width:18px;height:14px;display:inline-block;border:1px solid #ccc;margin-right:4px;"></div>25 menit<br>
            <div style="background:#2196F3;width:18px;height:14px;display:inline-block;border:1px solid #ccc;margin-right:4px;"></div>30 menit<br>
            <div style="background:#1565C0;width:18px;height:14px;display:inline-block;border:1px solid #ccc;margin-right:4px;"></div>45 menit<br>
            <div style="background:#0D47A1;width:18px;height:14px;display:inline-block;border:1px solid #ccc;margin-right:4px;"></div>60 menit<br>
            <div style="margin-top:5px;"><small><i>Garis tipis: Jaringan jalan yang terjangkau</i></small></div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        st.session_state.map_created = True
        st.session_state.last_map = m
        return m

    except Exception as e:
        st.error(f"Error membuat peta: {str(e)}")
        m = folium.Map(location=location_point, zoom_start=14)
        folium.Marker(location_point, popup='Titik Analisis').add_to(m)
        return m

# ============================================================
# FUNGSI DISPLAY HASIL
# ============================================================
def display_results(location_point, graph, education_facilities, accessibility_zones,
                    reachable_edges, area_calculation_method, travel_speed,
                    search_radius, time_limits, **kwargs):

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Interaktif", "📊 Dashboard", "🎓 Fasilitas", "📈 Analisis"])

    with tab1:
        st.subheader("🗺️ Peta Jangkauan Fasilitas Pendidikan")

        if accessibility_zones:
            st.info(f"**📍 Titik Analisis:** {location_point[0]:.6f}, {location_point[1]:.6f}")

            if area_calculation_method == "Service Area":
                hull_method = kwargs.get('hull_method', 'Concave Hull (Alpha Shape)')
                st.info(f"**🔧 Metode:** {hull_method}")

            m = create_comprehensive_map(location_point, accessibility_zones,
                                         education_facilities, reachable_edges)
            folium_static(m, width=1200, height=650)

            st.markdown("### 📍 Informasi Peta")
            st.info("""
            **Layer Peta:**
            - **Area berwarna biru**: Zona aksesibilitas berdasarkan waktu tempuh
            - **Garis tipis**: Jaringan jalan yang terjangkau
            - **Marker berwarna**: Fasilitas pendidikan (warna menunjukkan jenjang)
            - **Titik merah**: Lokasi analisis awal

            **Interaksi:**
            - Klik pada area/polygon untuk melihat detail
            - Klik pada fasilitas untuk informasi lengkap
            - Gunakan kontrol di pojok kanan atas untuk zoom dan fullscreen
            """)
        else:
            st.warning("⚠️ Tidak ada zona jangkauan yang dapat dihitung.")

    with tab2:
        st.subheader("📊 Dashboard Analisis Network Coverage")

        col1, col2, col3 = st.columns(3)

        with col1:
            if graph:
                st.metric("📌 Node Jaringan", f"{len(graph.nodes()):,}",
                          help="Jumlah simpul dalam jaringan jalan")
            elif area_calculation_method == "Buffer dari Titik":
                st.metric("📌 Metode", "Buffer Langsung")

        with col2:
            if graph:
                st.metric("🛣️ Edge Jaringan", f"{len(graph.edges()):,}",
                          help="Jumlah segmen jalan")
            elif area_calculation_method == "Buffer dari Titik":
                st.metric("🛣️ Bentuk Buffer", kwargs.get('buffer_shape', 'Lingkaran'))

        with col3:
            total_fac = education_facilities.shape[0] if not education_facilities.empty else 0
            st.metric("🎓 Total Fasilitas Pendidikan", total_fac,
                      help="Total fasilitas pendidikan dalam area")

        col4, col5 = st.columns(2)
        with col4:
            st.metric("🚗 Mode Transportasi",
                      kwargs.get('mode_bahasa', 'mobil/motor').capitalize(), delta="Terpilih")
        with col5:
            st.metric("⚡ Kecepatan", f"{travel_speed} km/jam")

        # Statistik per jenis fasilitas pendidikan
        if not education_facilities.empty and 'edu_type' in education_facilities.columns:
            st.subheader("🎓 Distribusi Fasilitas Pendidikan dalam Area")
            type_counts = education_facilities['edu_type'].value_counts().reset_index()
            type_counts.columns = ['Jenis Fasilitas', 'Jumlah']
            st.dataframe(type_counts, use_container_width=True, hide_index=True)

        if area_calculation_method == "Service Area" and accessibility_zones:
            st.subheader("📊 Statistik Service Area")
            network_stats = []
            for time_limit, zone_data in accessibility_zones.items():
                if zone_data.get('calculation_method') == "Service Area":
                    edges_count = len(reachable_edges.get(time_limit, [])) if reachable_edges else 0
                    network_stats.append({
                        "⏱️ Waktu": f"{time_limit} menit",
                        "📏 Jarak Maks": f"{zone_data['max_distance']:.0f} m",
                        "🛣️ Edges Terjangkau": edges_count,
                        "📐 Luas Area": f"{zone_data['area_sqkm']:.2f} km²",
                        "🎓 Fasilitas Pendidikan": zone_data.get('facilities_count', 0)
                    })

            if network_stats:
                st.dataframe(pd.DataFrame(network_stats), use_container_width=True, hide_index=True)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                tl_list = list(accessibility_zones.keys())
                areas = [accessibility_zones[t]['area_sqkm'] for t in tl_list]
                fac_counts = [accessibility_zones[t].get('facilities_count', 0) for t in tl_list]

                ax1.bar([str(t) for t in tl_list], areas, color='steelblue', edgecolor='black')
                ax1.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Luas Area (km²)', fontsize=12, fontweight='bold')
                ax1.set_title('Luas Service Area vs Waktu', fontsize=14, fontweight='bold')
                ax1.grid(axis='y', alpha=0.3)

                ax2.bar([str(t) for t in tl_list], fac_counts, color='teal', edgecolor='black')
                ax2.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Jumlah Fasilitas Pendidikan', fontsize=12, fontweight='bold')
                ax2.set_title('Fasilitas Pendidikan Terjangkau vs Waktu', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

        st.subheader("🔧 Informasi Metode Analisis")

        if area_calculation_method == "Service Area":
            method_info = f"""
            **🔍 Service Area (Concave Hull / Alpha Shape)**

            **Algoritma yang digunakan:**
            1. **Dijkstra Algorithm**: Menghitung jarak terpendek dari titik awal ke semua nodes
            2. **Identifikasi Edges Terjangkau**: Edges yang memiliki minimal satu node terjangkau
            3. **Kumpulkan Points**: Mengumpulkan semua titik dari nodes dan edges
            4. **Concave Hull**: Membentuk concave hull menggunakan shapely.concave_hull
            5. **Buffer Smoothing**: Buffer sebesar {kwargs.get('service_buffer', 100)}m untuk bentuk yang smooth

            **Parameter:**
            - Buffer Service Area: `{kwargs.get('service_buffer', 100)}` meter
            - Radius Jaringan: `{search_radius}` meter
            """
        else:
            method_info = f"""
            **🔍 Buffer dari Titik**

            **Algoritma:**
            1. **Direct Buffering** dari titik analisis ({location_point[0]:.6f}, {location_point[1]:.6f})
            2. **Shape**: {kwargs.get('buffer_shape', 'Lingkaran')}

            **Parameter:**
            - Bentuk Buffer: `{kwargs.get('buffer_shape', 'Lingkaran')}`
            - Analisis sederhana tanpa jaringan
            """

        st.markdown(method_info)

    with tab3:
        st.subheader("🎓 Fasilitas Pendidikan yang Dapat Diakses")

        if accessibility_zones and any(
            'accessible_facilities' in zone and zone['accessible_facilities']
            for zone in accessibility_zones.values()
        ):
            for time_limit in sorted(time_limits):
                if time_limit in accessibility_zones:
                    facilities = accessibility_zones[time_limit].get('accessible_facilities', [])

                    with st.expander(
                        f"🎓 Fasilitas dalam {time_limit} menit ({len(facilities)})", expanded=False
                    ):
                        if facilities:
                            fac_data = []
                            for fac in facilities:
                                fac_data.append({
                                    "🏷️ Nama": fac['name'][:50] + "..." if len(fac['name']) > 50 else fac['name'],
                                    "📚 Jenjang": fac.get('edu_type', '-'),
                                    "🔧 Amenity OSM": fac.get('amenity', '-'),
                                    "📏 Jarak (m)": f"{fac['distance_m']:.0f}",
                                    "⏱️ Waktu (menit)": f"{fac['travel_time_min']:.1f}",
                                    "📍 Latitude": f"{fac['coordinates'][0]:.4f}",
                                    "📍 Longitude": f"{fac['coordinates'][1]:.4f}"
                                })

                            fac_df = pd.DataFrame(fac_data)
                            st.dataframe(fac_df, use_container_width=True, hide_index=True)

                            # Ringkasan per jenjang
                            if fac_data:
                                type_summary = pd.DataFrame(fac_data).groupby('📚 Jenjang').size().reset_index()
                                type_summary.columns = ['Jenjang', 'Jumlah']
                                st.markdown("**Ringkasan per jenjang:**")
                                st.dataframe(type_summary, use_container_width=True, hide_index=True)

                            csv = fac_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Data CSV",
                                data=csv,
                                file_name=f"fasilitas_pendidikan_{time_limit}menit.csv",
                                mime="text/csv",
                                key=f"download_{time_limit}"
                            )
                        else:
                            st.info(f"ℹ️ Tidak ada fasilitas pendidikan yang dapat diakses dalam {time_limit} menit.")
        else:
            st.warning("⚠️ Tidak ada fasilitas pendidikan yang dapat diakses dalam area ini.")

    with tab4:
        st.subheader("📈 Analisis dan Visualisasi")

        if accessibility_zones:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            tl_list = list(accessibility_zones.keys())
            areas = [accessibility_zones[t]['area_sqkm'] for t in tl_list]
            fac_counts = [accessibility_zones[t].get('facilities_count', 0) for t in tl_list]

            colors_area = plt.cm.Blues(np.linspace(0.4, 0.9, len(tl_list)))
            bars1 = ax1.bar([str(t) for t in tl_list], areas, color=colors_area, edgecolor='black')
            ax1.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Luas Area (km²)', fontsize=12, fontweight='bold')
            ax1.set_title('Luas Service Area vs Waktu', fontsize=14, fontweight='bold')
            for bar in bars1:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., h,
                         f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)

            if sum(fac_counts) > 0:
                colors_fac = plt.cm.Greens(np.linspace(0.4, 0.9, len(tl_list)))
                bars2 = ax2.bar([str(t) for t in tl_list], fac_counts,
                                color=colors_fac, edgecolor='black')
                ax2.set_xlabel('Batas Waktu (menit)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Jumlah Fasilitas Pendidikan', fontsize=12, fontweight='bold')
                ax2.set_title('Fasilitas Pendidikan yang Dapat Diakses', fontsize=14, fontweight='bold')
                for bar in bars2:
                    h = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., h,
                             f'{h}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Tidak ada data fasilitas pendidikan',
                         ha='center', va='center', transform=ax2.transAxes,
                         fontsize=12, fontweight='bold')
                ax2.set_title('Tidak Ada Fasilitas Ditemukan', fontsize=14, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)

            # Analisis per jenjang pendidikan
            all_facilities_flat = []
            for zone_data in accessibility_zones.values():
                if 'accessible_facilities' in zone_data:
                    all_facilities_flat.extend(zone_data['accessible_facilities'])

            total_unique = len({f['name'] for f in all_facilities_flat})

            if all_facilities_flat:
                all_travel_times = [f['travel_time_min'] for f in all_facilities_flat]
                all_distances = [f['distance_m'] for f in all_facilities_flat]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🏷️ Fasilitas Unik", total_unique)
                with col2:
                    st.metric("⏱️ Rata-rata Waktu", f"{np.mean(all_travel_times):.1f} menit")
                with col3:
                    st.metric("📏 Rata-rata Jarak", f"{np.mean(all_distances):.0f} m")
                with col4:
                    st.metric("⚡ Waktu Terdekat", f"{min(all_travel_times):.1f} menit")

                # Distribusi per jenjang
                edu_type_counts = {}
                for f in all_facilities_flat:
                    et = f.get('edu_type', 'Lainnya')
                    edu_type_counts[et] = edu_type_counts.get(et, 0) + 1

                if edu_type_counts:
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    edu_labels = list(edu_type_counts.keys())
                    edu_values = list(edu_type_counts.values())
                    colors_edu = plt.cm.Set2(np.linspace(0, 1, len(edu_labels)))
                    bars3 = ax3.bar(edu_labels, edu_values, color=colors_edu, edgecolor='black')
                    ax3.set_xlabel('Jenjang Pendidikan', fontsize=12, fontweight='bold')
                    ax3.set_ylabel('Jumlah Fasilitas', fontsize=12, fontweight='bold')
                    ax3.set_title('Distribusi Fasilitas Pendidikan per Jenjang', fontsize=14, fontweight='bold')
                    for bar in bars3:
                        h = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., h,
                                 f'{h}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                    ax3.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig3)

                # Histogram waktu tempuh
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.hist(all_travel_times, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
                ax2.set_xlabel('Waktu Tempuh (menit)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Jumlah Fasilitas', fontsize=12, fontweight='bold')
                ax2.set_title('Distribusi Waktu Tempuh ke Fasilitas Pendidikan', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("ℹ️ Tidak ada data fasilitas untuk ditampilkan.")

            st.subheader("💾 Ekspor Hasil Analisis")
            summary_data = []
            for time_limit, zone_data in accessibility_zones.items():
                summary_data.append({
                    'time_limit_min': time_limit,
                    'method': zone_data.get('calculation_method', area_calculation_method),
                    'max_distance_m': zone_data['max_distance'],
                    'area_sqkm': zone_data['area_sqkm'],
                    'facilities_count': zone_data.get('facilities_count', 0)
                })

            csv_summary = pd.DataFrame(summary_data).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Ringkasan Analisis (CSV)",
                data=csv_summary,
                file_name="ringkasan_analisis_aksesibilitas_pendidikan.csv",
                mime="text/csv",
                key="download_summary"
            )

# ============================================================
# HALAMAN AWAL
# ============================================================
def display_welcome_page():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 📋 Panduan Penggunaan Network Analysis Coverage

        1. **📍 Pilih lokasi** di sidebar (kota atau koordinat manual)
        2. **⚙️ Atur parameter**:
           - Mode transportasi (jalan kaki / sepeda / mobil/motor)
           - Kecepatan perjalanan (km/jam)
           - Radius pencarian (500-30000m)
           - Batas waktu jangkauan (menit)
           - Metode coverage area (Service Area / Buffer dari Titik)
        3. **🚀 Klik "Jalankan Analisis"** untuk memulai

        ## 🎯 Jenis Fasilitas Pendidikan yang Dianalisis

        | Jenjang | Keterangan |
        |---|---|
        | 🏫 **SD** | Sekolah Dasar, Madrasah Ibtidaiyah |
        | 📘 **SMP** | Sekolah Menengah Pertama, Madrasah Tsanawiyah |
        | 📗 **SMA/SMK** | Sekolah Menengah Atas/Kejuruan, Madrasah Aliyah |
        | 🏛️ **Universitas** | Universitas |
        | 🏢 **Sekolah Tinggi** | Sekolah Tinggi, Institut, Politeknik, Akademi |
        | ✏️ **Lembaga Kursus** | Kursus, Bimbel, Pelatihan, LKP |
        """)

    with col2:
        st.markdown("""
        ## 💡 Tips Network Analysis

        ### Untuk Hasil Terbaik:
        - **Urban Area**: Radius 2.000-5.000m, Buffer 100-500m
        - **Suburban Area**: Radius 5.000-15.000m, Buffer 200-1.000m
        - **Rural Area**: Radius 15.000-30.000m, Buffer 500-3.000m

        ### Perbedaan Hull Methods:
        - **Concave Hull**: Mengikuti bentuk jaringan, hasil lebih akurat
        - **Buffer Edges**: Smooth dan cepat, baik untuk area urban

        ### Visualisasi Peta:
        - 🔵 **Area biru**: Zona aksesibilitas
        - 〰️ **Garis tipis**: Jaringan jalan terjangkau
        - 🎓 **Marker warna**: Fasilitas pendidikan per jenjang
        - 🔴 **Titik merah**: Lokasi analisis awal
        """)

# ============================================================
# SIDEBAR INPUT
# ============================================================
with st.sidebar:
    st.header("📍 Parameter Titik Analisis")

    input_method = st.radio(
        "Metode Input Titik Analisis:",
        ["Pilih Kota", "Input Manual"],
        key="input_method"
    )

    if input_method == "Pilih Kota":
        kota_options = {
            "Malang": (-7.9819, 112.6200),
            "Jakarta": (-6.2088, 106.8456),
            "Bandung": (-6.9175, 107.6191),
            "Surabaya": (-7.2575, 112.7521),
            "Yogyakarta": (-7.7956, 110.3695),
            "Semarang": (-6.9667, 110.4167),
            "Denpasar": (-8.6705, 115.2126),
            "Medan": (3.5952, 98.6722),
            "Makassar": (-5.1477, 119.4327)
        }
        selected_city = st.selectbox("Pilih Kota:", list(kota_options.keys()), key="selected_city")
        location_point = kota_options[selected_city]
        st.success(f"📍 **Titik Analisis:** {selected_city}: {location_point}")
    else:
        st.write("Masukkan koordinat titik analisis:")
        lat = st.number_input("Latitude:", value=-6.2088, format="%.6f", key="lat_input")
        lon = st.number_input("Longitude:", value=106.8456, format="%.6f", key="lon_input")
        location_point = (lat, lon)
        st.success(f"📍 **Titik Analisis:** {lat}, {lon}")

    st.subheader("⚙️ Pengaturan Analisis")

    network_type_bahasa = st.selectbox(
        "Mode Transportasi:",
        ["jalan kaki", "sepeda", "mobil/motor"],
        index=2,
        help="Jenis jaringan yang digunakan untuk analisis",
        key="network_type_bahasa"
    )
    network_type_osmnx = convert_transport_mode(network_type_bahasa)

    travel_speed = st.slider(
        "Kecepatan (km/jam):",
        min_value=1.0, max_value=100.0,
        value=get_default_speed(network_type_bahasa),
        step=0.5, key="travel_speed"
    )

    search_radius = st.slider(
        "Radius Jaringan (meter):",
        min_value=500, max_value=30000, value=2000, step=100,
        help="Radius untuk mengambil jaringan jalan dari titik analisis",
        key="search_radius"
    )

    time_limits = st.multiselect(
        "Batas Waktu (menit):",
        [5, 10, 15, 20, 25, 30, 45, 60],
        default=[15, 25],
        help="Waktu tempuh maksimum untuk menghitung jangkauan",
        key="time_limits"
    )

    area_calculation_method = st.selectbox(
        "Metode Coverage Area:",
        ["Service Area", "Buffer dari Titik"],
        index=0,
        key="area_calculation_method"
    )

    if area_calculation_method == "Service Area":
        st.info("**Metode Concave Hull**: Service area akan mengikuti bentuk jaringan jalan")
        service_buffer = st.slider(
            "Buffer Service Area (meter):",
            20, 5000, 100, 10,
            help="Buffer untuk smoothing service area dari jaringan jalan (maks: 5000m)",
            key="service_buffer"
        )
        hull_method = st.selectbox(
            "Metode Hull:",
            ["Concave Hull (Alpha Shape)", "Buffer Edges"],
            index=0,
            key="hull_method"
        )
    elif area_calculation_method == "Buffer dari Titik":
        buffer_shape = st.selectbox(
            "Bentuk Buffer:",
            ["Lingkaran", "Persegi", "Kapsul"],
            index=0,
            key="buffer_shape"
        )

    analyze_button = st.button("🚀 Jalankan Analisis", type="primary",
                               use_container_width=True, key="analyze_button")

    if st.button("🔄 Reset Analisis", use_container_width=True, key="reset_button"):
        for key in ['analysis_results', 'map_created', 'accessibility_zones',
                    'education_facilities', 'reachable_edges', 'last_location',
                    'analysis_params', 'last_analysis_time', 'analysis_in_progress',
                    'network_cache']:
            if key in ['map_created', 'analysis_in_progress']:
                st.session_state[key] = False
            elif key in ['last_analysis_time']:
                st.session_state[key] = 0
            elif key == 'network_cache':
                st.session_state[key] = {}
            else:
                st.session_state[key] = None
        st.rerun()

    st.markdown("---")
    st.info("""
    **📌 Panduan Metode Network Analysis:**

    1. **Service Area (Concave Hull)**:
       - Analisis jaringan sebenarnya berdasarkan struktur jalan
       - Dijkstra Algorithm menghitung jarak dari titik awal
       - Concave Hull untuk bentuk yang mengikuti jaringan

    2. **Buffer dari Titik**:
       - Buffer sederhana dari titik analisis
       - Tidak memerlukan analisis jaringan jalan
       - Cepat untuk estimasi awal

    **🎯 Rekomendasi Buffer:**
    - Jalan kaki: 50-200 meter
    - Sepeda: 100-500 meter
    - Mobil/motor: 200-2000 meter
    """)

# ============================================================
# MAIN APPLICATION
# ============================================================
main_container = st.container()

with main_container:
    if analyze_button and time_limits:
        if st.session_state.analysis_in_progress:
            st.warning("⏳ Analisis sedang berjalan...")
        else:
            kwargs = {}
            if area_calculation_method == "Service Area":
                kwargs['service_buffer'] = service_buffer
                kwargs['hull_method'] = hull_method
                kwargs['mode_bahasa'] = network_type_bahasa
            elif area_calculation_method == "Buffer dari Titik":
                kwargs['buffer_shape'] = buffer_shape
                kwargs['mode_bahasa'] = network_type_bahasa

            current_params = {
                'location': location_point,
                'network_type': network_type_osmnx,
                'speed': travel_speed,
                'radius': search_radius,
                'time_limits': tuple(sorted(time_limits)),
                'method': area_calculation_method,
                **kwargs
            }

            if st.session_state.analysis_params != current_params:
                with st.spinner("📍 Sedang menganalisis dari titik..."):
                    result = analyze_from_point_main(
                        location_point, network_type_osmnx, travel_speed,
                        search_radius, time_limits, area_calculation_method, **kwargs
                    )

                if result[0] is not None or area_calculation_method == "Buffer dari Titik":
                    graph, education_facilities, accessibility_zones, reachable_edges = result
                    display_results(location_point, graph, education_facilities,
                                    accessibility_zones, reachable_edges,
                                    area_calculation_method, travel_speed,
                                    search_radius, time_limits, **kwargs)
                else:
                    st.error("❌ Analisis gagal. Periksa parameter dan coba lagi.")
            else:
                st.info("📊 **Menampilkan hasil analisis sebelumnya**")
                if st.session_state.analysis_results:
                    graph, education_facilities, accessibility_zones, reachable_edges = st.session_state.analysis_results
                    display_results(location_point, graph, education_facilities,
                                    accessibility_zones, reachable_edges,
                                    area_calculation_method, travel_speed,
                                    search_radius, time_limits, **kwargs)

    elif st.session_state.analysis_results and st.session_state.accessibility_zones:
        st.info("📊 **Menampilkan hasil analisis sebelumnya**")
        graph, education_facilities, accessibility_zones, reachable_edges = st.session_state.analysis_results

        if st.session_state.analysis_params:
            params = st.session_state.analysis_params
            display_results(
                params.get('location', location_point),
                graph, education_facilities, accessibility_zones, reachable_edges,
                params.get('method', area_calculation_method),
                params.get('speed', travel_speed),
                params.get('radius', search_radius),
                params.get('time_limits', time_limits),
                **{k: v for k, v in params.items()
                   if k not in ['location', 'network_type', 'speed', 'radius', 'time_limits', 'method']}
            )

    elif not analyze_button:
        display_welcome_page()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <p style='font-size: 1.1em; font-weight: bold; color: #2c3e50;'>
        🎓📍 <b>Network Analysis Coverage Area — Fasilitas Pendidikan</b> v6.0
        </p>
        <p style='font-size: 0.9em; color: #7f8c8d;'>
        Developer: <b>Adipandang Yudono, S.Si., MURP., PhD</b> (Spatial Analysis, Architecture System, Scrypt Developer, WebGIS Analytics)
        <br>
        <b>Algoritma Network Analysis:</b> Dijkstra + Concave Hull (Shapely) + Buffer Smoothing
        <br>
        Data sumber: © OpenStreetMap contributors
        <br>
        2026
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(45deg, #1565C0, #0D47A1);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(21, 101, 192, 0.3);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #1565C0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.3s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; padding: 0 10px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: bold;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(21, 101, 192, 0.3);
    }
    .dataframe { border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .dataframe thead th {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white; font-weight: bold;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%);
    }
    .folium-map {
        width: 100% !important; height: 650px !important;
        border: 1px solid #ddd; border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
