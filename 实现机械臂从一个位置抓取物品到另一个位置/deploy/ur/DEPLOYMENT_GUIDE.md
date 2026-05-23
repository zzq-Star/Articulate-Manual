# Deployment Guide — UR

## Overview
Tool: tool0
Payload: 0.0 kg
Speed: 0.25
Waypoints: 8

## Waypoints
  1. [PTP] [0.0, 0.0, 0.0, 1.0]
  2. [LIN] [0.0, 0.0, 0.0, 0.2, 0.3, 0.5]
  3. [CIRC] [0.15, -0.1, 0.2, 0.3, 0.4, 0.5]
  4. [SPLINE] [0.3, -0.2, 0.4, 0.4, 0.5, 0.5]
  5. [PTP] [0.3, -0.2, 0.5, 0.5, 0.5, 0.5]
  6. [PTP] [0.15, -0.1, 0.5, 0.4, 0.4, 0.4]
  7. [PTP] [0.0, 0.0, 0.5, 0.3, 0.3, 0.3]
  8. [PTP] [0.0, 0.0, 0.0, 0.2, 0.3, 0.5]

## Safety
  - Verify tool TCP calibration before first run.
  - Set reduced speed mode (250 mm/s) during initial test.
  - Confirm payload does not exceed wrist rating.
  - Check joint limit software thresholds are enabled.
  - Verify emergency stop is functional.
  - Run program in single-step mode first.
  - Ensure no personnel inside safeguarded area.