#!/bin/bash
# Start the xtrainer stack from this client tree, without using anyone else's
# checkout of roboticsx_vla_client.
#
# Robot binaries and launch files still live in /data/robotics.install (that is
# the machine install). This script and scripts/dobot_config/ are ours, so
# flag/default/settings changes in ~/yihaowu/... do not affect us.
#
# Must be run from a desktop session with DISPLAY (the collect_agent GUI is
# required="true" in the launch file).
#
#   cd ~/arianliu/client
#   ./scripts/run_collection.sh
#
set -uo pipefail
WARN(){ echo -e "\033[0;33m[WARN] $* \033[0m"; }
ERROR(){ echo -e "\033[1;31m[ERROR] $* \033[0m"; }

HERE=$(dirname "$(readlink -f "$0")")
INSTALL_ROOT="${INSTALL_ROOT:-/data/robotics.install}"
DOBOT_SRC="$HERE/dobot_config"

START_INFERENCE=false
SYNC=false
RUN_HIL=false
AGENT_TYPE="collection"

options=$(getopt -l "sync,nosync,inference,hil,deploy" -o "snihd" -a -- "$@")
eval set -- "$options"
while true; do
  case $1 in
    -s | --sync)
        SYNC=true
        ;;
    -n | --nosync)
        SYNC=false
        ;;
    -i | --inference)
        START_INFERENCE=true
        ;;
    -h | --hil)
        RUN_HIL=true
        ;;
    -d | --deploy)
        AGENT_TYPE="deploy"
        ;;
    --)
        shift
        break
        ;;
  esac
  shift
done

if [[ ! -x "$INSTALL_ROOT/scripts/run_docker.sh" ]]; then
    ERROR "robot install not found at $INSTALL_ROOT"
    exit 1
fi

if $SYNC; then
    cd "$INSTALL_ROOT" || { ERROR "进入安装目录失败" && exit 1; }
    if ./scripts/sync_artifacts.sh xtrainer.install; then
        cp -f ./RUN.md "$HOME/Desktop"
    else
        WARN "更新部署件失败"
    fi
fi

PLATFORM=$(cat /etc/robotics_platform_name)
case "$PLATFORM" in
    xtrainer1) src=dobot_settings1.ini ;;
    xtrainer2) src=dobot_settings2.ini ;;
    xtrainer3) src=dobot_settings3.ini ;;
    *)
        ERROR "Unexpected platform name: $PLATFORM"
        exit 1
        ;;
esac

if [[ ! -f "$DOBOT_SRC/$src" ]]; then
    ERROR "missing $DOBOT_SRC/$src"
    exit 1
fi

# Apply our snapshot into the install tree the docker workdir uses.
cp -f "$DOBOT_SRC/$src" "$INSTALL_ROOT/lib/python3/dist-packages/scripts/dobot_config/dobot_settings.ini"
echo "[run_collection] platform=$PLATFORM using $src (from $DOBOT_SRC)"

if $START_INFERENCE; then
    WARN "starting vendor inference_service as well; omit -i to only bring up the robot"
fi

cd "$INSTALL_ROOT" || { ERROR "进入安装目录失败" && exit 1; }
./scripts/run_docker.sh -i ros-runtime -e \
    "COLLECT_AGENT_TYPE=$AGENT_TYPE roslaunch robotics run_xtrainer.launch run_inference:=$START_INFERENCE run_hil:=$RUN_HIL" \
    || { ERROR "启动Xtrainer失败" && exit 1; }

wait
