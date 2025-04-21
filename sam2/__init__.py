# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_dir  # initialize_config_module
from hydra.core.global_hydra import GlobalHydra

# if not GlobalHydra.instance().is_initialized():
#     initialize_config_module("sam2", version_base="1.2")

# Get absolute path to the `configs` directory inside your ROS package
import rospkg
import os

rospack = rospkg.RosPack()
pkg_path = rospack.get_path("lasr_vision_sam2")
config_dir = os.path.join(pkg_path, "sam2")

# Initialize Hydra with this config directory
if not GlobalHydra.instance().is_initialized():
    initialize_config_dir(config_dir=config_dir, version_base="1.2")

