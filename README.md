# Soft-Robot Self-Healing — Research Software Deliverables

Research code and supporting artifacts for experimental soft-robot control, model adaptation, and related simulations. This repository contains Python experiments, a ROS 2 experimental workspace, SOFA examples, and supporting analysis scripts.

> **Status:** Research software. This collection has not been validated as a standalone, one-command installation. Hardware-specific setup and external dependencies may be required. Verify all safety limits before commanding physical actuators.

## Repository layout

| Path | Contents |
| --- | --- |
| `experiment_code/` | Scripts for plotting controller, tracking, comparison, and experimental results |
| `exp_setup_ws/src/` | ROS 2 packages for robot control, pneumatic actuation, data recording, message definitions, and experiment controllers |
| `sofa/` | SOFA-related simulation and controller scripts |
| `scripts/` | Training, evaluation, plotting, model and adaptation utilities |

The ROS 2 workspace contains package-level READMEs in `exp_setup_ws/src/ros2-vtem_control-main/` and `exp_setup_ws/src/ros2-pneumatic_actuation-main/`. Consult those files for relevant package build and launch commands.

## Reproducing results

Reproduction requires determining the software versions, dependencies, robot/simulation configuration, calibration parameters, and data/model provenance for the specific experiment. These are **not fully documented** in this repository yet. Do not assume that all scripts run unmodified on another machine.

Some `.jax` files under `exp_setup_ws/src/controller/controller/models/` may be model artifacts rather than source scripts. CSV files also contain experiment artifacts. They have been preserved in this archive.

## Attribution and licensing

This repository includes components with their own notices and licenses, including the MIT license inside `exp_setup_ws/src/ros2-vtem_control-main/` and Apache License 2.0 inside `exp_setup_ws/src/data_recorder/`. Preserve the copyright notices and comply with the applicable terms. **No license is asserted for the collection as a whole** by this README. Obtain the appropriate permission before publishing any institution-owned, collaborator-owned, or third-party material.

## Publication and citation

Add the associated paper, project webpage, and recommended citation when the release is approved and the bibliographic details are finalized.
