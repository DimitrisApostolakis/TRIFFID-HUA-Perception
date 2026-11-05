"""Position/heading sources and existing fallback priority, separate from feature delivery."""
import collections
import json
import math
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import URLError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32
try:
    from ugv_status_interfaces.msg import UnitreeStatus
except ImportError:
    UnitreeStatus = None
_GPS_WINDOW = 7
_R_EARTH = 6378137.0

class PositionProvider:
    def __init__(self, node):
        self.node = node
        self.position_api_url = node.get_parameter('position_api_url').value
        self.use_position_api = bool(node.get_parameter('use_position_api').value)
        self.ugv_status_topic = node.get_parameter('ugv_status_topic').value
        self.ugv_status_timeout_s = max(0.1, float(node.get_parameter('ugv_status_timeout_s').value))
        # GPS state
        # Sliding window for median filtering of noisy GPS fixes
        self._gps_lat_buf = collections.deque(maxlen=_GPS_WINDOW)
        self._gps_lon_buf = collections.deque(maxlen=_GPS_WINDOW)
        self._gps_alt_buf = collections.deque(maxlen=_GPS_WINDOW)

        # Selected robot position. Fresh ROS UGV status is primary, followed
        # by the HTTP status API, /fix, and finally gps_origin_* parameters.
        self.robot_lat = 0.0
        self.robot_lon = 0.0
        self.robot_alt = 0.0
        self.gps_valid = False
        self._position_source = 'none'
        self._position_lock = threading.Lock()

        self._status_lat = 0.0
        self._status_lon = 0.0
        self._status_position_valid = False
        self._status_received_monotonic = None

        self._api_lat = 0.0
        self._api_lon = 0.0
        self._api_position_valid = False

        self._fix_lat = 0.0
        self._fix_lon = 0.0
        self._fix_alt = 0.0
        self._fix_valid = False

        # Seed from parameters if provided
        param_lat = self.node.get_parameter('gps_origin_lat').value
        param_lon = self.node.get_parameter('gps_origin_lon').value
        param_alt = self.node.get_parameter('gps_origin_alt').value
        self._seed_lat = float(param_lat)
        self._seed_lon = float(param_lon)
        self._seed_alt = float(param_alt)
        self._seed_valid = param_lat != 0.0 and param_lon != 0.0
        if self._seed_valid:
            self.robot_lat = self._seed_lat
            self.robot_lon = self._seed_lon
            self.robot_alt = self._seed_alt
            self.gps_valid = True
            self._position_source = 'param'

        self.robot_heading = 0.0
        self.heading_valid = False

        self.sub_status = None
        if UnitreeStatus is not None:
            self.sub_status = self.node.create_subscription(
                UnitreeStatus,
                self.ugv_status_topic,
                self.ugv_status_callback,
                10,
            )
        else:
            self.node.get_logger().warn(
                'ugv_status_interfaces is not installed; ROS UGV status '
                'subscription disabled. Using API / /fix fallbacks.'
            )

        self.sub_gps = self.node.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10,
        )
        self.sub_heading = self.node.create_subscription(
            Float32,
            '/b2/nicla/magnetometer/heading_broadcasted',
            self.heading_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        # Poll the UGV status API on a background thread so network latency
        # never blocks ROS callbacks. Only latitude/longitude are consumed.
        self._position_stop = threading.Event()
        self._position_worker = None
        if self.use_position_api:
            self._position_worker = threading.Thread(
                target=self._position_api_worker_loop,
                daemon=True,
            )
            self._position_worker.start()
        else:
            self.node.get_logger().info(
                '  Position API disabled; using ROS UGV status / /fix fallbacks.'
            )
        self.node.create_timer(1.0, self._check_ugv_status_freshness)

    @staticmethod
    def _valid_lat_lon(lat, lon):
        return (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90.0 <= lat <= 90.0
            and -180.0 <= lon <= 180.0
            and not (lat == 0.0 and lon == 0.0)
        )

    def _select_position_locked(self):
        """Select the highest-priority currently available position source."""
        if self._status_position_valid:
            self.robot_lat = self._status_lat
            self.robot_lon = self._status_lon
            if self._fix_valid:
                self.robot_alt = self._fix_alt
            elif self._seed_valid:
                self.robot_alt = self._seed_alt
            else:
                self.robot_alt = 0.0
            self.gps_valid = True
            self._position_source = 'ugv_status'
        elif self._api_position_valid:
            self.robot_lat = self._api_lat
            self.robot_lon = self._api_lon
            if self._fix_valid:
                self.robot_alt = self._fix_alt
            elif self._seed_valid:
                self.robot_alt = self._seed_alt
            else:
                self.robot_alt = 0.0
            self.gps_valid = True
            self._position_source = 'api'
        elif self._fix_valid:
            self.robot_lat = self._fix_lat
            self.robot_lon = self._fix_lon
            self.robot_alt = self._fix_alt
            self.gps_valid = True
            self._position_source = 'fix'
        elif self._seed_valid:
            self.robot_lat = self._seed_lat
            self.robot_lon = self._seed_lon
            self.robot_alt = self._seed_alt
            self.gps_valid = True
            self._position_source = 'param'
        else:
            self.gps_valid = False
            self._position_source = 'none'

    def ugv_status_callback(self, msg):
        """Use fresh, GNSS-valid ROS UGV status as the primary position."""
        try:
            if hasattr(msg, 'gnss_valid') and not bool(msg.gnss_valid):
                raise ValueError('gnss_valid=false')

            lat = float(msg.latitude)
            lon = float(msg.longitude)

            # Some UnitreeStatus producers also carry a nested
            # global_position. Use it only if the top-level coordinates are
            # invalid and the producer explicitly reports that field present.
            if (not self._valid_lat_lon(lat, lon)
                    and bool(getattr(msg, 'has_global_position', False))):
                global_position = msg.global_position
                if (hasattr(global_position, 'gnss_valid')
                        and not bool(global_position.gnss_valid)):
                    raise ValueError('global_position.gnss_valid=false')
                lat = float(global_position.latitude)
                lon = float(global_position.longitude)

            if not self._valid_lat_lon(lat, lon):
                raise ValueError('invalid latitude/longitude')

        except (AttributeError, TypeError, ValueError) as e:
            with self._position_lock:
                previous_source = self._position_source
                self._status_position_valid = False
                self._select_position_locked()
                current_source = self._position_source

            if previous_source == 'ugv_status':
                self.node.get_logger().warn(
                    f'ROS UGV status invalid — falling back to '
                    f'{current_source}: {e}'
                )
            return

        with self._position_lock:
            previous_source = self._position_source
            self._status_lat = lat
            self._status_lon = lon
            self._status_position_valid = True
            self._status_received_monotonic = time.monotonic()
            self._select_position_locked()
            current_source = self._position_source

        if current_source == 'ugv_status' and previous_source != 'ugv_status':
            self.node.get_logger().info(
                f'UGV position acquired from ROS status: '
                f'({lat:.7f}, {lon:.7f})'
            )

    def _check_ugv_status_freshness(self):
        """Invalidate ROS UGV status when messages stop arriving."""
        with self._position_lock:
            if (not self._status_position_valid
                    or self._status_received_monotonic is None):
                return

            age_s = time.monotonic() - self._status_received_monotonic
            if age_s <= self.ugv_status_timeout_s:
                return

            previous_source = self._position_source
            self._status_position_valid = False
            self._select_position_locked()
            current_source = self._position_source

        if previous_source == 'ugv_status':
            self.node.get_logger().warn(
                f'ROS UGV status stale ({age_s:.1f}s) — falling back to '
                f'{current_source}.'
            )

    def _position_api_worker_loop(self):
        """Poll the UGV status API as the first fallback position source."""
        while not self._position_stop.is_set():
            previous_source = None
            try:
                req = Request(
                    self.position_api_url,
                    method='GET',
                    headers={'Accept': 'application/json'},
                )
                with urlopen(req, timeout=3) as resp:
                    if resp.status != 200:
                        raise URLError(f'HTTP {resp.status}')
                    payload = json.loads(resp.read().decode('utf-8'))

                message = payload.get('message') or {}
                if 'gnss_valid' in message and not bool(message['gnss_valid']):
                    raise ValueError('gnss_valid=false')

                lat = float(message['latitude'])
                lon = float(message['longitude'])
                if not self._valid_lat_lon(lat, lon):
                    raise ValueError('invalid latitude/longitude')

                with self._position_lock:
                    previous_source = self._position_source
                    self._api_lat = lat
                    self._api_lon = lon
                    self._api_position_valid = True
                    self._select_position_locked()
                    current_source = self._position_source

                if current_source == 'api' and previous_source != 'api':
                    self.node.get_logger().info(
                        f'UGV position acquired from API: '
                        f'({lat:.7f}, {lon:.7f})'
                    )

            except (URLError, OSError, KeyError, TypeError, ValueError,
                    json.JSONDecodeError, UnicodeDecodeError) as e:
                with self._position_lock:
                    previous_source = self._position_source
                    self._api_position_valid = False
                    self._select_position_locked()
                    current_source = self._position_source

                if previous_source == 'api':
                    if current_source == 'fix':
                        self.node.get_logger().warn(
                            f'Position API unavailable — falling back to /fix: {e}'
                        )
                    elif current_source == 'param':
                        self.node.get_logger().warn(
                            f'Position API unavailable — falling back to '
                            f'gps_origin_* parameters: {e}'
                        )
                    else:
                        self.node.get_logger().warn(
                            f'Position API unavailable and no /fix available: {e}'
                        )

            self._position_stop.wait(2.0)

    def gps_callback(self, msg: NavSatFix):
        """Track /fix as the median-filtered fallback position."""
        try:
            lat = float(msg.latitude)
            lon = float(msg.longitude)
            alt = float(msg.altitude)
        except (TypeError, ValueError):
            return

        if not self._valid_lat_lon(lat, lon):
            return

        self._gps_lat_buf.append(lat)
        self._gps_lon_buf.append(lon)
        if math.isfinite(alt):
            self._gps_alt_buf.append(alt)

        fix_lat = float(sorted(self._gps_lat_buf)[len(self._gps_lat_buf) // 2])
        fix_lon = float(sorted(self._gps_lon_buf)[len(self._gps_lon_buf) // 2])
        if self._gps_alt_buf:
            fix_alt = float(
                sorted(self._gps_alt_buf)[len(self._gps_alt_buf) // 2]
            )
        else:
            fix_alt = 0.0

        with self._position_lock:
            previous_source = self._position_source
            self._fix_lat = fix_lat
            self._fix_lon = fix_lon
            self._fix_alt = fix_alt
            self._fix_valid = True
            self._select_position_locked()
            current_source = self._position_source

        if current_source == 'fix' and previous_source != 'fix':
            self.node.get_logger().info(
                f'Using /fix fallback: ({fix_lat:.7f}, {fix_lon:.7f}, '
                f'{fix_alt:.1f}m)'
            )

    def heading_callback(self, msg: Float32):
        """Normalize compass heading to [-pi, pi), keeping clockwise-positive."""
        heading_deg = float(msg.data) % 360.0
        heading = math.radians(heading_deg)
        self.robot_heading = (heading + math.pi) % (2.0 * math.pi) - math.pi

        if not self.heading_valid:
            self.heading_valid = True
            self.node.get_logger().info(
                f'Heading acquired: {heading_deg:.1f}° magnetic, '
                f'{math.degrees(self.robot_heading):.1f}° normalized'
            )

    @staticmethod
    def _body_to_enu(x_fwd, y_left, z_up, heading):
        """Convert body-frame (fwd, left, up) offset to ENU metres."""
        angle_relative = math.atan2(-y_left, x_fwd)
        angle_absolute = heading + angle_relative
        angle_absolute = (
            (angle_absolute + math.pi) % (2.0 * math.pi)
        ) - math.pi

        distance = math.hypot(x_fwd, y_left)
        east = distance * math.sin(angle_absolute)
        north = distance * math.cos(angle_absolute)
        up = z_up
        return east, north, up

    def body_to_gps(self, x_fwd, y_left, z_up=0.0):
        """Convert a b2/base_link offset to (lon, lat, alt); raw body-frame
        coords if no global UGV position is available."""
        with self._position_lock:
            gps_valid = self.gps_valid
            robot_lat = self.robot_lat
            robot_lon = self.robot_lon
            robot_alt = self.robot_alt

        if not gps_valid:
            return (x_fwd, y_left, z_up)

        heading = self.robot_heading if self.heading_valid else 0.0

        # Step 1: body → ENU
        east, north, up = self._body_to_enu(
            x_fwd, y_left, z_up, heading
        )

        # Step 2: ENU metres → degree offsets
        lat_rad = math.radians(robot_lat)
        d_lat = north / _R_EARTH * (180.0 / math.pi)
        d_lon = east / (_R_EARTH * math.cos(lat_rad)) * (180.0 / math.pi)

        # Step 3: add to current robot position
        lat = robot_lat + d_lat
        lon = robot_lon + d_lon
        alt = robot_alt + up

        return (lon, lat, alt)  # GeoJSON order: [lon, lat, alt]

    def snapshot(self):
        with self._position_lock:
            return self.gps_valid, self.robot_lat, self.robot_lon, self.robot_alt, self._position_source

    def close(self):
        self._position_stop.set()
