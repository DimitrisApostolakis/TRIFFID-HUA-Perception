FROM ros:jazzy-perception-noble

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=42

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Frozen SANSA runtime. NumPy 1.x remains compatible with ROS cv_bridge.
RUN pip3 install --no-cache-dir --break-system-packages \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
COPY src/triffid_ugv_perception/requirements*.txt /tmp/ugv-requirements/
RUN pip3 install --no-cache-dir --break-system-packages \
    -r /tmp/ugv-requirements/requirements-sansa.txt

# Workspace setup 
WORKDIR /ws
# src/ is bind-mounted at runtime via docker-compose

# Ensure ROS2 + workspace overlay are sourced in every shell
# (docker compose exec runs non-interactive bash, which skips .bashrc)
# BASH_ENV is sourced by every non-interactive bash invocation.
ENV BASH_ENV=/etc/triffid_ros_env.sh
RUN echo '#!/bin/bash'                                          >  /etc/triffid_ros_env.sh && \
    echo 'source /opt/ros/jazzy/setup.bash'                    >> /etc/triffid_ros_env.sh && \
    echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /etc/triffid_ros_env.sh && \
    chmod +x /etc/triffid_ros_env.sh

# Entrypoint 
COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["bash"]
