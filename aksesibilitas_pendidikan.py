import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon, LineString, box, MultiPolygon
from shapely.ops import unary_union
from shapely import concave_hull, convex_hull
from pyproj import Transformer
from folium.plugins import MiniMap, Fullscreen
import warnings
import time
import math

warnings.filterwarnings('ignore')

# ============================================================
# SESSION STATE
# ============================================================
_defaults = {
    'analysis_results': None,
    'map_created': False,
    'accessibility_zones': None,
    'education_facilities': None,
    'reachable_edges': None,
    'last_location': None,
    'analysis_params': None,
    'last_analysis_time': 0,
    'analysis_in_progress': False,
    'network_cache': {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Aksesibilitas Fasilitas Pendidikan",
    page_icon="🎓",
    layout="wide"
)
st.title("🎓📍 Analisis Spasial Aksesibilitas Fasilitas Pendidikan")
st.markdown("**Analisis zona aksesibilitas dengan Network Coverage dari titik analisis**")

# ============================================================
# KLASIFIKASI FASILITAS PENDIDIKAN
# ============================================================
EDU_MARKER = {
    'SD':             ('blue',      'graduation-cap'),
    'SMP':            ('green',     'book'),
    'SMA':            ('orange',    'certificate'),
    'SD/SMP/SMA':     ('lightblue', 'school'),
    'Universitas':    ('red',       'university'),
    'Sekolah Tinggi': ('purple',    'building'),
    'Lembaga Kursus': ('cadetblue', 'pencil'),
    'Lainnya':        ('gray',      'info-circle'),
}

def classify_edu(row):
    name    = str(row.get('name', '')).lower()
    amenity = str(row.get('amenity', '')).lower()
    isced   = str(row.get('isced:level', '')).lower()
    if isced == '1': return 'SD'
    if isced == '2': return 'SMP'
    if isced == '3': return 'SMA'
    if amenity == 'university':  return 'Universitas'
    if amenity == 'college':     return 'Sekolah Tinggi'
    if amenity in ('language_school','driving_school','music_school','training'): return 'Lembaga Kursus'
    kw_map = {
        'SD':  ['sekolah dasar','sd ','elementary','primary','madrasah ibtidaiyah','mi '],
        'SMP': ['sekolah menengah pertama','smp ','junior high','madrasah tsanawiyah','mts '],
        'SMA': ['sekolah menengah atas','sma ','smk ','senior high','madrasah aliyah','ma '],
        'Universitas':    ['universitas','university','univ '],
        'Sekolah Tinggi': ['sekolah tinggi','institut ','akademi ','politeknik','stie ','stik '],
        'Lembaga Kursus': ['kursus','pelatihan','bimbel','bimbingan belajar','les ','lkp '],
    }
    for ftype, keywords in kw_map.items():
        for kw in keywords:
            if kw in name:
                return ftype
    if amenity == 'school': return 'SD/SMP/SMA'
    return 'Lainnya'

# ============================================================
# TRANSPORT HELPERS
# ============================================================
def convert_mode(mode_id):
    return {'jalan kaki':'walk','sepeda':'bike','mobil/motor':'drive'}.get(mode_id,'drive')

def default_speed(mode_id):
    return {'jalan kaki':5.0,'sepeda':15.0,'mobil/motor':40.0}.get(mode_id,40.0)

# ============================================================
# HAVERSINE
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    r = math.pi/180
    dlat = (lat2-lat1)*r
    dlon = (lon2-lon1)*r
    a = math.sin(dlat/2)**2 + math.cos(lat1*r)*math.cos(lat2*r)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ============================================================
# BUG FIX: Dapatkan centroid WGS84 dari geometry apapun
# (Point, Polygon, MultiPolygon, dll dari OSM)
# ============================================================
def get_centroid_latlon(geom):
    """
    Kembalikan (lat, lon) dari geometry apapun.
    OSM sering mengembalikan Polygon/MultiPolygon untuk bangunan sekolah.
    Tanpa ini, hasattr(geom, 'x') False → fasilitas tidak pernah terhitung.
    """
    try:
        if geom is None or geom.is_empty:
            return None, None
        # Untuk Point langsung ambil koordinat
        if geom.geom_type == 'Point':
            return geom.y, geom.x
        # Untuk Polygon, MultiPolygon, dll → ambil centroid
        c = geom.centroid
        return c.y, c.x
    except:
        return None, None

# ============================================================
# BUG FIX: Hitung luas polygon WGS84 dalam km²
# Shapely .area pada WGS84 menghasilkan derajat² → perlu konversi
# ============================================================
def calc_area_sqkm_wgs84(polygon, ref_lat):
    """
    Konversi luas polygon WGS84 ke km² menggunakan faktor skala lokal.
    """
    try:
        # Proyeksikan ke meter menggunakan UTM lokal
        import pyproj
        zone = int((ref_lat + 180) / 6) + 1
        # Untuk Indonesia (lintang selatan) pakai zone selatan
        epsg_utm = 32700 + zone if ref_lat < 0 else 32600 + zone
        proj_in  = pyproj.CRS('EPSG:4326')
        proj_out = pyproj.CRS(f'EPSG:{epsg_utm}')
        transformer = Transformer.from_crs(proj_in, proj_out, always_xy=True)

        if polygon.geom_type == 'Polygon':
            coords_proj = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
            poly_proj   = Polygon(coords_proj)
            return poly_proj.area / 1e6
        elif polygon.geom_type == 'MultiPolygon':
            total = 0
            for p in polygon.geoms:
                coords_proj = [transformer.transform(x, y) for x, y in p.exterior.coords]
                total += Polygon(coords_proj).area
            return total / 1e6
        else:
            # Fallback kasar
            deg_area = polygon.area
            m_per_deg_lat = 111320
            m_per_deg_lon = 111320 * math.cos(math.radians(ref_lat))
            return deg_area * m_per_deg_lat * m_per_deg_lon / 1e6
    except:
        return 0.0

# ============================================================
# CONVEX HULL (no scipy)
# ============================================================
def simple_convex_hull(points):
    from shapely.geometry import MultiPoint
    mp = MultiPoint([(float(x), float(y)) for x, y in points])
    return convex_hull(mp)

# ============================================================
# BUG FIX: AMBIL FASILITAS PENDIDIKAN OSM
# osmnx versi baru: features_from_bbox pakai (west,south,east,north) via bbox dict
# atau tetap (north,south,east,west) positional — tapi perlu error handling lebih baik
# ============================================================
def get_education_facilities(bbox):
    """
    bbox = (north, south, east, west)
    Coba berbagai format untuk kompatibilitas osmnx lama & baru.
    """
    north, south, east, west = bbox
    tags_list = [
        {'amenity': 'school'},
        {'amenity': 'university'},
        {'amenity': 'college'},
        {'amenity': 'language_school'},
        {'amenity': 'driving_school'},
        {'amenity': 'music_school'},
        {'amenity': 'kindergarten'},  # tambah TK
    ]
    collected = []

    for tags in tags_list:
        try:
            # Coba format baru osmnx (bbox sebagai dict/tuple WSEN)
            try:
                f = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
            except TypeError:
                # Fallback format lama (north, south, east, west)
                f = ox.features_from_bbox(north, south, east, west, tags=tags)

            if not f.empty:
                collected.append(f)
        except Exception:
            continue

    if not collected:
        return gpd.GeoDataFrame()

    combined = gpd.GeoDataFrame(pd.concat(collected, ignore_index=True))
    combined = combined.drop_duplicates(subset=['geometry'])

    if 'name' not in combined.columns:
        combined['name'] = 'Fasilitas Pendidikan'
    else:
        combined['name'] = combined['name'].fillna('Fasilitas Pendidikan')

    combined['edu_type'] = combined.apply(classify_edu, axis=1)

    # BUG FIX: Normalisasi semua geometry ke centroid Point WGS84
    # Ini krusial karena OSM mengembalikan campuran Point + Polygon + MultiPolygon
    centroids = []
    for geom in combined.geometry:
        lat, lon = get_centroid_latlon(geom)
        if lat is not None:
            centroids.append(Point(lon, lat))
        else:
            centroids.append(None)
    combined['centroid_geom'] = centroids
    combined = combined[combined['centroid_geom'].notna()].copy()
    combined['centroid_geom'] = gpd.GeoSeries(combined['centroid_geom'], crs='EPSG:4326')

    return combined

# ============================================================
# NETWORK HELPERS
# ============================================================
def get_network(location_point, network_type, radius):
    key = f"{location_point}_{network_type}_{radius}"
    if key in st.session_state.network_cache:
        return st.session_state.network_cache[key]
    try:
        G = ox.graph_from_point(location_point, dist=radius,
                                network_type=network_type,
                                simplify=True, retain_all=True)
        G = ox.project_graph(G)
        st.session_state.network_cache[key] = G
        return G
    except Exception as e:
        st.error(f"❌ Gagal ambil jaringan: {e}")
        return None

def nearest_node(G, location_point):
    lat, lon = location_point
    try:
        node = ox.distance.nearest_nodes(G, lon, lat)
        nd   = G.nodes[node]
        transformer = Transformer.from_crs(G.graph['crs'], 'EPSG:4326', always_xy=True)
        lon_w, lat_w = transformer.transform(nd['x'], nd['y'])
        return node, (nd['x'], nd['y']), (lat_w, lon_w)
    except Exception as e:
        st.error(f"❌ Gagal cari node: {e}")
        return None, None, None

# ============================================================
# SERVICE AREA (CONCAVE / CONVEX HULL)
# ============================================================
def calc_service_area(G, start_node, start_coords, max_dist, service_buffer=100):
    try:
        dist_map = nx.single_source_dijkstra_path_length(
            G, start_node, weight='length', cutoff=max_dist)
        if not dist_map:
            return None, []

        edges_geom  = []
        all_pts     = []
        seen_nodes  = set()

        for u, v, data in G.edges(data=True):
            du = dist_map.get(u, float('inf'))
            dv = dist_map.get(v, float('inf'))
            if du > max_dist and dv > max_dist:
                continue

            for n in (u, v):
                if n not in seen_nodes:
                    seen_nodes.add(n)
                    all_pts.append((G.nodes[n]['x'], G.nodes[n]['y']))

            if 'geometry' in data:
                geom = data['geometry']
                all_pts.extend(list(geom.coords))
            else:
                geom = LineString([
                    (G.nodes[u]['x'], G.nodes[u]['y']),
                    (G.nodes[v]['x'], G.nodes[v]['y'])
                ])
            edges_geom.append(geom)

        if not edges_geom:
            return None, []

        seen_pts = set()
        unique_pts = []
        for p in all_pts:
            k = (round(p[0],2), round(p[1],2))
            if k not in seen_pts:
                seen_pts.add(k)
                unique_pts.append(p)

        area = None
        if len(unique_pts) >= 3:
            try:
                from shapely.geometry import MultiPoint
                mp = MultiPoint([(x,y) for x,y in unique_pts])
                for ratio in [0.05, 0.1, 0.2, 0.3]:
                    candidate = concave_hull(mp, ratio=ratio)
                    if candidate and not candidate.is_empty and candidate.geom_type == 'Polygon':
                        area = candidate
                        break
                if area is None:
                    area = convex_hull(mp)
            except:
                area = simple_convex_hull(unique_pts)

        if area is None or area.is_empty:
            buffers = []
            for eg in edges_geom:
                try: buffers.append(eg.buffer(service_buffer, resolution=4))
                except: pass
            if buffers:
                area = unary_union(buffers)

        if area is None or area.is_empty:
            area = Point(start_coords).buffer(max_dist * 0.8)

        try:
            area = area.buffer(service_buffer, resolution=8)
            sp = Point(start_coords)
            if not area.contains(sp):
                area = unary_union([area, sp.buffer(service_buffer*2)])
            area = area.simplify(service_buffer/2, preserve_topology=True)
        except:
            pass

        return area, edges_geom

    except Exception as e:
        st.error(f"Error service area: {e}")
        return None, []

# ============================================================
# BUG FIX: BUFFER LANGSUNG — kembalikan dalam WGS84 derajat
# dan simpan max_dist dalam meter untuk luas yang benar
# ============================================================
def calc_buffer(location_point, max_dist, shape='Lingkaran'):
    """
    Buat buffer dalam koordinat WGS84 (derajat).
    max_dist dalam meter.
    """
    lat, lon = location_point
    deg_lat  = max_dist / 111320
    deg_lon  = max_dist / (111320 * math.cos(math.radians(lat)))

    if shape == 'Persegi':
        return box(lon-deg_lon, lat-deg_lat, lon+deg_lon, lat+deg_lat)
    if shape == 'Kapsul':
        c1   = Point(lon-deg_lon/2, lat).buffer(deg_lat/2, resolution=8)
        c2   = Point(lon+deg_lon/2, lat).buffer(deg_lat/2, resolution=8)
        rect = box(lon-deg_lon/2, lat-deg_lat/3, lon+deg_lon/2, lat+deg_lat/3)
        return unary_union([c1, c2, rect])
    # Lingkaran — buffer dalam derajat (rata-rata lat/lon)
    deg_avg = (deg_lat + deg_lon) / 2
    return Point(lon, lat).buffer(deg_avg, resolution=32)

# ============================================================
# KONVERSI POLYGON KE WGS84 (untuk hasil Service Area projected)
# ============================================================
def to_wgs84(polygon, crs, fallback_loc):
    try:
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        if hasattr(polygon, 'exterior'):
            coords = [transformer.transform(x,y) for x,y in polygon.exterior.coords]
            return Polygon(coords)
        elif polygon.geom_type == 'MultiPolygon':
            polys = []
            for p in polygon.geoms:
                coords = [transformer.transform(x,y) for x,y in p.exterior.coords]
                polys.append(Polygon(coords))
            return MultiPolygon(polys)
    except:
        pass
    lat, lon = fallback_loc
    return Point(lon, lat).buffer(0.01)

# ============================================================
# ANALISIS UTAMA
# ============================================================
def run_analysis(location_point, network_type, speed_kmh, radius_m,
                 time_limits_min, method, **kwargs):

    params = {
        'location': location_point, 'network_type': network_type,
        'speed': speed_kmh, 'radius': radius_m,
        'time_limits': tuple(sorted(time_limits_min)),
        'method': method, **kwargs
    }
    if st.session_state.analysis_params == params and st.session_state.analysis_results:
        st.info("✅ Menggunakan hasil analisis sebelumnya...")
        return st.session_state.analysis_results

    st.session_state.analysis_in_progress = True
    pb = st.progress(0)
    status = st.empty()

    G = start_node = start_coords = None
    if method != "Buffer dari Titik":
        status.text("🛣️ Mengambil jaringan jalan...")
        G = get_network(location_point, network_type, radius_m)
        if G is None:
            st.session_state.analysis_in_progress = False
            return None, None, None, None
        pb.progress(20)
        status.text("📍 Mencari node terdekat...")
        start_node, start_coords, _ = nearest_node(G, location_point)
        if start_node is None:
            st.session_state.analysis_in_progress = False
            return None, None, None, None
        pb.progress(35)

    status.text("🎓 Mencari fasilitas pendidikan di OSM...")
    lat, lon = location_point
    R = 6378137
    dlat = (radius_m/R)*(180/math.pi)
    dlon = (radius_m/(R*math.cos(math.pi*lat/180)))*(180/math.pi)
    bbox = (lat+dlat, lat-dlat, lon+dlon, lon-dlon)
    edu_fac = get_education_facilities(bbox)

    n_fac = len(edu_fac) if not edu_fac.empty else 0
    status.text(f"🎓 Ditemukan {n_fac} fasilitas pendidikan di area pencarian...")
    pb.progress(50)

    status.text("🧮 Menghitung zona aksesibilitas...")
    speed_mpm  = (speed_kmh * 1000) / 60
    zones      = {}
    edges_dict = {}
    service_buf = kwargs.get('service_buffer', 100)

    for i, tl in enumerate(sorted(time_limits_min)):
        max_dist = speed_mpm * tl

        if method == "Buffer dari Titik":
            poly_wgs   = calc_buffer(location_point, max_dist, kwargs.get('buffer_shape', 'Lingkaran'))
            edges_geom = []
            # BUG FIX: Hitung luas dengan benar dalam km²
            area_sqkm  = calc_area_sqkm_wgs84(poly_wgs, lat)
        else:
            poly_proj, edges_geom = calc_service_area(G, start_node, start_coords, max_dist, service_buf)
            if poly_proj is None:
                poly_wgs  = calc_buffer(location_point, max_dist)
                edges_geom = []
                area_sqkm  = calc_area_sqkm_wgs84(poly_wgs, lat)
            else:
                # area_sqkm dari projected CRS (sudah dalam meter) — ini benar
                area_sqkm = poly_proj.area / 1e6
                poly_wgs  = to_wgs84(poly_proj, G.graph['crs'], location_point)

        if poly_wgs and not poly_wgs.is_empty:
            facs = []
            if not edu_fac.empty:
                for _, row in edu_fac.iterrows():
                    try:
                        # BUG FIX: Gunakan centroid_geom (sudah berupa Point WGS84)
                        # bukan row.geometry yang bisa Polygon/MultiPolygon
                        cgeom = row.get('centroid_geom')
                        if cgeom is None or cgeom.is_empty:
                            continue

                        # Cek apakah centroid berada dalam zona
                        if not cgeom.within(poly_wgs):
                            continue

                        fy, fx = cgeom.y, cgeom.x
                        d = haversine(lat, lon, fy, fx)
                        facs.append({
                            'name':             str(row.get('name', 'Fasilitas Pendidikan')),
                            'amenity':          str(row.get('amenity', '-')),
                            'edu_type':         str(row.get('edu_type', 'Lainnya')),
                            'distance_m':       d,
                            'travel_time_min':  d / speed_mpm,
                            'coordinates':      (fy, fx),
                        })
                    except Exception:
                        continue

            zones[tl] = {
                'geometry':              poly_wgs,
                'max_distance':          max_dist,
                'area_sqkm':             area_sqkm,
                'calculation_method':    method,
                'accessible_facilities': facs,
                'facilities_count':      len(facs),
            }
            edges_dict[tl] = edges_geom

        pb.progress(50 + int((i+1)*50/len(time_limits_min)))

    pb.progress(100)
    status.text("✅ Selesai!")
    time.sleep(0.3)
    status.empty()

    result = (G, edu_fac, zones, edges_dict)
    st.session_state.analysis_results    = result
    st.session_state.analysis_params     = params
    st.session_state.accessibility_zones = zones
    st.session_state.education_facilities = edu_fac
    st.session_state.reachable_edges     = edges_dict
    st.session_state.analysis_in_progress = False
    return result

# ============================================================
# BUAT PETA
# ============================================================
ZONE_COLORS = {5:'#E3F2FD',10:'#BBDEFB',15:'#90CAF9',20:'#64B5F6',
               25:'#42A5F5',30:'#2196F3',45:'#1565C0',60:'#0D47A1'}

def build_map(loc, zones, edu_fac, edges_dict):
    m = folium.Map(location=loc, zoom_start=14, tiles='OpenStreetMap', control_scale=True)

    folium.Marker(loc,
        popup='<b>📍 Titik Analisis</b>',
        tooltip='Titik Analisis',
        icon=folium.Icon(color='red', icon='bullseye', prefix='fa')
    ).add_to(m)

    if edges_dict:
        for tl, edges in edges_dict.items():
            color = ZONE_COLORS.get(tl, '#90CAF9')
            for eg in edges:
                try:
                    coords = [[y, x] for x, y in eg.coords]
                    if len(coords) >= 2:
                        folium.PolyLine(coords, color=color, weight=1, opacity=0.25).add_to(m)
                except:
                    continue

    for tl, zd in zones.items():
        color = ZONE_COLORS.get(tl, '#90CAF9')
        try:
            poly = zd['geometry']
            if poly.geom_type == 'Polygon' and len(list(poly.exterior.coords)) >= 3:
                pts = [[lat, lon] for lon, lat in poly.exterior.coords]
                folium.Polygon(
                    pts, color=color, fill=True,
                    fill_color=color, fill_opacity=0.3, weight=2,
                    popup=(f"<b>{tl} menit</b><br>"
                           f"Luas: {zd['area_sqkm']:.2f} km²<br>"
                           f"Fasilitas: {zd['facilities_count']}"),
                    tooltip=f"Zona {tl} menit"
                ).add_to(m)
            elif poly.geom_type == 'MultiPolygon':
                for p in poly.geoms:
                    pts = [[lat, lon] for lon, lat in p.exterior.coords]
                    folium.Polygon(
                        pts, color=color, fill=True,
                        fill_color=color, fill_opacity=0.3, weight=2,
                        tooltip=f"Zona {tl} menit"
                    ).add_to(m)
        except:
            folium.Circle(loc, radius=zd['max_distance'], color=color,
                          fill=True, fill_color=color, fill_opacity=0.3).add_to(m)

    added = set()
    for tl, zd in zones.items():
        for fac in zd.get('accessible_facilities', []):
            key = f"{fac['coordinates'][0]:.5f},{fac['coordinates'][1]:.5f}"
            if key in added:
                continue
            added.add(key)
            color, icon = EDU_MARKER.get(fac['edu_type'], ('gray', 'info-circle'))
            folium.Marker(
                fac['coordinates'],
                popup=folium.Popup(
                    f"<b>{fac['name']}</b><br>Jenjang: {fac['edu_type']}<br>"
                    f"Jarak: {fac['distance_m']:.0f} m<br>Waktu: {fac['travel_time_min']:.1f} mnt",
                    max_width=250),
                tooltip=f"{fac['name']} ({fac['edu_type']})",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)

    legend = """
    <div style="position:fixed;bottom:50px;left:50px;width:210px;
                background:white;border:2px solid grey;z-index:9999;
                font-size:12px;padding:10px;border-radius:8px;opacity:0.95;">
    <b>LEGENDA</b><br>
    <i class="fa fa-bullseye" style="color:red"></i> Titik Analisis<br>
    <i class="fa fa-graduation-cap" style="color:blue"></i> SD<br>
    <i class="fa fa-book" style="color:green"></i> SMP<br>
    <i class="fa fa-certificate" style="color:orange"></i> SMA/SMK<br>
    <i class="fa fa-university" style="color:red"></i> Universitas<br>
    <i class="fa fa-building" style="color:purple"></i> Sekolah Tinggi<br>
    <i class="fa fa-pencil" style="color:cadetblue"></i> Lembaga Kursus<br>
    <hr style="margin:4px 0">
    <div style="background:#E3F2FD;width:14px;height:10px;display:inline-block;border:1px solid #ccc"></div> 5 mnt &nbsp;
    <div style="background:#90CAF9;width:14px;height:10px;display:inline-block;border:1px solid #ccc"></div> 15 mnt<br>
    <div style="background:#2196F3;width:14px;height:10px;display:inline-block;border:1px solid #ccc"></div> 30 mnt &nbsp;
    <div style="background:#0D47A1;width:14px;height:10px;display:inline-block;border:1px solid #ccc"></div> 60 mnt
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    MiniMap(toggle_display=True).add_to(m)
    Fullscreen().add_to(m)
    return m

# ============================================================
# DISPLAY HASIL
# ============================================================
def show_results(loc, G, edu_fac, zones, edges_dict, method, speed, radius, time_limits, **kw):
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta","📊 Dashboard","🎓 Fasilitas","📈 Analisis"])

    with tab1:
        st.subheader("🗺️ Peta Jangkauan Fasilitas Pendidikan")
        st.info(f"**📍 Titik:** {loc[0]:.6f}, {loc[1]:.6f} | **Metode:** {method}")
        m = build_map(loc, zones, edu_fac, edges_dict)
        folium_static(m, width=1200, height=620)

    with tab2:
        st.subheader("📊 Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🎓 Total Fasilitas di Area",
                      edu_fac.shape[0] if not edu_fac.empty else 0)
        with c2:
            st.metric("⚡ Kecepatan", f"{speed} km/jam")
        with c3:
            total_reach = sum(z['facilities_count'] for z in zones.values())
            st.metric("✅ Fasilitas Terjangkau", total_reach)
        with c4:
            st.metric("🔍 Metode", method)

        if not edu_fac.empty and 'edu_type' in edu_fac.columns:
            st.subheader("Distribusi Jenis Fasilitas dalam Area Pencarian")
            tc = edu_fac['edu_type'].value_counts().reset_index()
            tc.columns = ['Jenjang', 'Jumlah']
            st.dataframe(tc, use_container_width=True, hide_index=True)

        if zones:
            st.subheader("Statistik per Zona Waktu")
            rows = []
            for tl, zd in sorted(zones.items()):
                rows.append({
                    "⏱️ Waktu":      f"{tl} menit",
                    "📏 Jarak Maks": f"{zd['max_distance']:.0f} m",
                    "📐 Luas":       f"{zd['area_sqkm']:.2f} km²",
                    "🎓 Fasilitas":  zd['facilities_count']
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            tl_list = sorted(zones.keys())
            areas   = [zones[t]['area_sqkm'] for t in tl_list]
            counts  = [zones[t]['facilities_count'] for t in tl_list]
            labels  = [str(t) for t in tl_list]

            ax1.bar(labels, areas, color='steelblue', edgecolor='black')
            ax1.set_title('Luas Area vs Waktu Tempuh', fontweight='bold')
            ax1.set_xlabel('Waktu (menit)'); ax1.set_ylabel('km²')
            ax1.grid(axis='y', alpha=0.3)
            for bar, v in zip(ax1.patches, areas):
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                         f'{v:.2f}', ha='center', va='bottom', fontsize=9)

            ax2.bar(labels, counts, color='teal', edgecolor='black')
            ax2.set_title('Fasilitas Pendidikan Terjangkau', fontweight='bold')
            ax2.set_xlabel('Waktu (menit)'); ax2.set_ylabel('Jumlah')
            ax2.grid(axis='y', alpha=0.3)
            for bar, v in zip(ax2.patches, counts):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                         str(v), ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    with tab3:
        st.subheader("🎓 Daftar Fasilitas yang Dapat Diakses")
        any_fac = any(z['accessible_facilities'] for z in zones.values())
        if any_fac:
            for tl in sorted(time_limits):
                if tl not in zones:
                    continue
                facs = zones[tl]['accessible_facilities']
                with st.expander(f"Zona {tl} menit — {len(facs)} fasilitas", expanded=False):
                    if facs:
                        df = pd.DataFrame([{
                            "Nama":      f['name'][:50],
                            "Jenjang":   f['edu_type'],
                            "Amenity":   f['amenity'],
                            "Jarak (m)": f"{f['distance_m']:.0f}",
                            "Waktu":     f"{f['travel_time_min']:.1f} mnt",
                        } for f in facs])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        summ = df.groupby('Jenjang').size().reset_index(name='Jumlah')
                        st.caption("Ringkasan per jenjang:")
                        st.dataframe(summ, use_container_width=True, hide_index=True)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(f"📥 Download CSV zona {tl} menit", csv,
                                           f"fasdik_{tl}mnt.csv", "text/csv",
                                           key=f"dl_{tl}")
                    else:
                        st.info("Tidak ada fasilitas dalam zona ini.")
        else:
            st.warning("Tidak ada fasilitas pendidikan yang ditemukan dalam area analisis.")

    with tab4:
        st.subheader("📈 Visualisasi Lanjutan")
        all_facs = [f for z in zones.values() for f in z['accessible_facilities']]
        if all_facs:
            times = [f['travel_time_min'] for f in all_facs]
            dists = [f['distance_m'] for f in all_facs]
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Fasilitas Unik",    len({f['name'] for f in all_facs}))
            with c2: st.metric("Rata-rata Waktu",   f"{np.mean(times):.1f} mnt")
            with c3: st.metric("Rata-rata Jarak",   f"{np.mean(dists):.0f} m")
            with c4: st.metric("Waktu Terdekat",    f"{min(times):.1f} mnt")

            type_cnt = {}
            for f in all_facs:
                type_cnt[f['edu_type']] = type_cnt.get(f['edu_type'], 0) + 1
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.bar(list(type_cnt.keys()), list(type_cnt.values()),
                    color=plt.cm.Set2(np.linspace(0, 1, len(type_cnt))), edgecolor='black')
            ax3.set_title('Distribusi Fasilitas per Jenjang', fontweight='bold')
            ax3.set_xlabel('Jenjang'); ax3.set_ylabel('Jumlah')
            ax3.grid(axis='y', alpha=0.3)
            plt.tight_layout(); st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.hist(times, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
            ax4.set_title('Distribusi Waktu Tempuh ke Fasilitas Pendidikan', fontweight='bold')
            ax4.set_xlabel('Waktu (menit)'); ax4.set_ylabel('Jumlah Fasilitas')
            ax4.grid(axis='y', alpha=0.3)
            plt.tight_layout(); st.pyplot(fig4)
        else:
            st.info("Tidak ada data fasilitas untuk divisualisasikan.")

        st.subheader("💾 Ekspor Ringkasan")
        summ = pd.DataFrame([{
            'Waktu_menit':        tl,
            'Metode':             z['calculation_method'],
            'Jarak_maks_m':       z['max_distance'],
            'Luas_km2':           z['area_sqkm'],
            'Jumlah_fasilitas':   z['facilities_count']
        } for tl, z in zones.items()])
        st.download_button("📥 Download Ringkasan CSV",
                           summ.to_csv(index=False).encode('utf-8'),
                           "ringkasan_aksesibilitas_pendidikan.csv",
                           "text/csv", key="dl_summary")

# ============================================================
# HALAMAN WELCOME
# ============================================================
def welcome():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## 📋 Panduan Penggunaan
        1. **Pilih lokasi** di sidebar (kota atau koordinat manual)
        2. **Atur parameter**: mode transportasi, kecepatan, radius, batas waktu, metode
        3. **Klik "🚀 Jalankan Analisis"**

        ## 🎓 Fasilitas Pendidikan yang Dianalisis
        | Ikon | Jenjang |
        |---|---|
        | 🎓 | SD / Madrasah Ibtidaiyah |
        | 📘 | SMP / Madrasah Tsanawiyah |
        | 📗 | SMA / SMK / Madrasah Aliyah |
        | 🏛️ | Universitas |
        | 🏢 | Sekolah Tinggi / Institut / Politeknik |
        | ✏️ | Lembaga Kursus / Bimbel / Pelatihan |
        """)
    with col2:
        st.markdown("""
        ## 💡 Tips
        **Radius yang disarankan:**
        - Urban: 1.000–3.000 m
        - Suburban: 3.000–8.000 m
        - Rural: 8.000–20.000 m

        **Buffer Service Area:**
        - Jalan kaki: 50–200 m
        - Sepeda: 100–500 m
        - Mobil: 200–1.000 m
        """)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📍 Titik Analisis")

    mode = st.radio("Input:", ["Pilih Kota", "Koordinat Manual"], key="input_mode")
    KOTA = {
        "Malang":     (-7.9819, 112.6200),
        "Jakarta":    (-6.2088, 106.8456),
        "Bandung":    (-6.9175, 107.6191),
        "Surabaya":   (-7.2575, 112.7521),
        "Yogyakarta": (-7.7956, 110.3695),
        "Semarang":   (-6.9667, 110.4167),
        "Denpasar":   (-8.6705, 115.2126),
        "Medan":      ( 3.5952,  98.6722),
        "Makassar":   (-5.1477, 119.4327),
    }
    if mode == "Pilih Kota":
        city = st.selectbox("Kota:", list(KOTA.keys()), key="city")
        loc  = KOTA[city]
        st.success(f"📍 {city}: {loc}")
    else:
        lat_in = st.number_input("Latitude:",  value=-6.2088, format="%.6f", key="lat")
        lon_in = st.number_input("Longitude:", value=106.8456, format="%.6f", key="lon")
        loc = (lat_in, lon_in)
        st.success(f"📍 {lat_in:.5f}, {lon_in:.5f}")

    st.subheader("⚙️ Pengaturan")
    mode_id  = st.selectbox("Mode Transportasi:",
                            ["jalan kaki", "sepeda", "mobil/motor"], index=2, key="mode_id")
    net_type = convert_mode(mode_id)
    speed    = st.slider("Kecepatan (km/jam):", 1.0, 100.0,
                         default_speed(mode_id), 0.5, key="speed")
    radius   = st.slider("Radius Pencarian (m):", 500, 20000, 2000, 100, key="radius")
    t_limits = st.multiselect("Batas Waktu (menit):",
                              [5, 10, 15, 20, 25, 30, 45, 60], default=[15, 25], key="t_limits")
    method   = st.selectbox("Metode Coverage:",
                            ["Service Area", "Buffer dari Titik"], key="method")

    extra = {}
    if method == "Service Area":
        extra['service_buffer'] = st.slider("Buffer Service Area (m):", 20, 2000, 100, 10, key="sbuf")
    else:
        extra['buffer_shape'] = st.selectbox("Bentuk Buffer:",
                                             ["Lingkaran", "Persegi", "Kapsul"], key="bshape")
    extra['mode_bahasa'] = mode_id

    run_btn   = st.button("🚀 Jalankan Analisis", type="primary",
                          use_container_width=True, key="run")
    reset_btn = st.button("🔄 Reset", use_container_width=True, key="reset")

    if reset_btn:
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.markdown("---")
    st.caption("Data: © OpenStreetMap contributors")

# ============================================================
# MAIN
# ============================================================
if run_btn and t_limits:
    if st.session_state.analysis_in_progress:
        st.warning("⏳ Analisis sedang berjalan...")
    else:
        with st.spinner("⏳ Menganalisis..."):
            result = run_analysis(loc, net_type, speed, radius, t_limits, method, **extra)

        if result and (result[0] is not None or method == "Buffer dari Titik"):
            G, edu_fac, zones, edges_dict = result
            show_results(loc, G, edu_fac, zones, edges_dict,
                         method, speed, radius, t_limits, **extra)
        else:
            st.error("❌ Analisis gagal. Coba kurangi radius atau ganti metode ke 'Buffer dari Titik'.")

elif st.session_state.analysis_results and st.session_state.accessibility_zones:
    G, edu_fac, zones, edges_dict = st.session_state.analysis_results
    p = st.session_state.analysis_params or {}
    show_results(
        p.get('location', loc), G, edu_fac, zones, edges_dict,
        p.get('method', method), p.get('speed', speed),
        p.get('radius', radius), list(p.get('time_limits', t_limits)),
        **{k: v for k, v in p.items()
           if k not in ('location', 'network_type', 'speed', 'radius', 'time_limits', 'method')}
    )
elif not run_btn:
    welcome()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:15px;color:#7f8c8d;font-size:0.85em;'>
🎓📍 <b>Aksesibilitas Fasilitas Pendidikan</b> v6.2 &nbsp;|&nbsp;
Dijkstra + Concave Hull (Shapely) &nbsp;|&nbsp;
Data: © OpenStreetMap contributors &nbsp;|&nbsp; 2026<br>
Developer: <b>Adipandang Yudono, S.Si., MURP., PhD</b>
</div>""", unsafe_allow_html=True)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.stButton>button{
    background:linear-gradient(45deg,#1565C0,#0D47A1);
    color:white;border:none;border-radius:8px;font-weight:bold;
    transition:all 0.2s;box-shadow:0 3px 6px rgba(0,0,0,.15);
}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 5px 10px rgba(21,101,192,.3);}
[data-testid="stMetric"]{
    background:linear-gradient(135deg,#f8f9fa,#e9ecef);
    border-radius:8px;padding:12px;border-left:4px solid #1565C0;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,#1565C0,#0D47A1)!important;
    color:white!important;
}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#e3f2fd,#bbdefb);}
</style>""", unsafe_allow_html=True)
