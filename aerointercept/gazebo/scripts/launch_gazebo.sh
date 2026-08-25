#!/usr/bin/env bash
# Launch only processes created here; never edits or globally kills ros2_ws/PX4.
set -eo pipefail

PROJECT_ROOT="/home/verser/Python/AeroIntercept"
PX4_ROOT="/home/verser/PX4-Autopilot"
ROS_WORKSPACE="/home/verser/ros2_ws"
WORLD_FILE="$PROJECT_ROOT/assets/gazebo/worlds/aerointercept_park.sdf"
BRIDGE_CONFIG="$PROJECT_ROOT/configs/gazebo_camera_bridge.yaml"
TARGET_BINARY="$ROS_WORKSPACE/build/uav_target_sim/uav_target_sim"
RUNTIME_DIR="/tmp/aerointercept_gazebo"
SOCKET_PATH="$RUNTIME_DIR/bridge.sock"
HEADLESS=0
MODE="circle"
SEED=0
TARGET_SPAWN_NORTH_M=10.0

usage() {
  echo "usage: $0 [--headless] [--mode circle|sinusoidal|random_walk|mixed] [--seed N] [--socket PATH]"
}

while (($#)); do
  case "$1" in
    --headless) HEADLESS=1; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --socket) SOCKET_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  circle|sinusoidal|random_walk) ;;
  mixed)
    MODES=(circle sinusoidal random_walk)
    MODE="${MODES[$((SEED % 3))]}"
    echo "[AeroIntercept] mixed selected '$MODE' for seed $SEED"
    ;;
  *) echo "existing C++ target supports circle, sinusoidal, random_walk, or mixed" >&2; exit 2 ;;
esac

for required in "$WORLD_FILE" "$BRIDGE_CONFIG" "$TARGET_BINARY" "$PX4_ROOT/build/px4_sitl_default/bin/px4"; do
  if [[ ! -e "$required" ]]; then
    echo "required file is missing: $required" >&2
    exit 1
  fi
done

# Refuse a conflicting stack instead of deleting user processes or logs.
if pgrep -x px4 >/dev/null || pgrep -f "gz sim" >/dev/null || pgrep -x MicroXRCEAgent >/dev/null; then
  echo "an existing PX4/Gazebo/MicroXRCEAgent process is running; stop it explicitly first" >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR/ros_logs"
rm -f "$SOCKET_PATH"
PIDS=()

cleanup() {
  trap - INT TERM EXIT
  echo "[AeroIntercept] stopping owned Gazebo/PX4/ROS processes..."
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  rm -f "$SOCKET_PATH"
}
trap cleanup INT TERM EXIT

source /opt/ros/humble/setup.bash
source "$ROS_WORKSPACE/install/setup.bash"
export ROS_LOG_DIR="$RUNTIME_DIR/ros_logs"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PARK_MODEL_ROOT="$PROJECT_ROOT/assets/gazebo/models/aerointercept_park_surface"
export GZ_SIM_RESOURCE_PATH="$PROJECT_ROOT/assets/gazebo/models:$PARK_MODEL_ROOT:$PARK_MODEL_ROOT/materials/textures:$PX4_ROOT/Tools/simulation/gz/models:$PX4_ROOT/Tools/simulation/gz/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

echo "[AeroIntercept] logs: $RUNTIME_DIR"
if ((HEADLESS)); then
  gz sim -r -s "$WORLD_FILE" >"$RUNTIME_DIR/gazebo.log" 2>&1 &
else
  gz sim -r "$WORLD_FILE" >"$RUNTIME_DIR/gazebo.log" 2>&1 &
fi
PIDS+=("$!")

MicroXRCEAgent udp4 -p 8888 >"$RUNTIME_DIR/micro_xrce.log" 2>&1 &
PIDS+=("$!")
echo "[AeroIntercept] waiting for Gazebo rendering and create services..."
sleep 8

(
  cd "$PX4_ROOT"
  exec env PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=aerointercept_park \
    PX4_SYS_AUTOSTART=4002 PX4_GZ_MODEL_POSE="0,0,0" PX4_SIM_MODEL=gz_x500_depth \
    ./build/px4_sitl_default/bin/px4 -d -i 1
) >"$RUNTIME_DIR/px4_interceptor.log" 2>&1 &
PIDS+=("$!")
echo "[AeroIntercept] waiting for interceptor model and PX4 bridge..."
sleep 8

(
  cd "$PX4_ROOT"
  exec env PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=aerointercept_park \
    PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="$TARGET_SPAWN_NORTH_M,0,0" PX4_SIM_MODEL=gz_x500 \
    ./build/px4_sitl_default/bin/px4 -d -i 2
) >"$RUNTIME_DIR/px4_target.log" 2>&1 &
PIDS+=("$!")
echo "[AeroIntercept] waiting for target model and PX4 bridge..."
sleep 5

ros2 run ros_gz_bridge parameter_bridge \
  --ros-args -p config_file:="$BRIDGE_CONFIG" \
  >"$RUNTIME_DIR/camera_bridge.log" 2>&1 &
PIDS+=("$!")

"$TARGET_BINARY" --ros-args -p motion_mode:="$MODE" -p max_range:=10.0 \
  >"$RUNTIME_DIR/target_controller.log" 2>&1 &
PIDS+=("$!")

/usr/bin/python3 "$PROJECT_ROOT/aerointercept/gazebo/ros_bridge.py" \
  --socket "$SOCKET_PATH" --image-topic /camera/image \
  --target-origin-ned "$TARGET_SPAWN_NORTH_M" 0 0 --reset-position-ned 0 0 -6 \
  --camera-mount-yaw-offset -1.5707963267948966 \
  >"$RUNTIME_DIR/training_bridge.log" 2>&1 &
PIDS+=("$!")

echo "[AeroIntercept] Gazebo park started; waiting for camera and both PX4 odometry streams."
echo "[AeroIntercept] socket: $SOCKET_PATH"
echo "[AeroIntercept] target C++ mode: $MODE"
echo "[AeroIntercept] initial model separation: ${TARGET_SPAWN_NORTH_M} m"
echo "[AeroIntercept] reset controller: physical hover + camera look-at-target yaw"
echo "[AeroIntercept] Ctrl+C stops only this launcher's child processes."
wait
