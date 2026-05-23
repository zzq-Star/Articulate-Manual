# ROS2 Humble Coding Context
# ===========================
# This context is injected into code generation prompts.
# It defines conventions for ROS2 Humble Python node development.

## Package Structure
- package_name/
  - package.xml
  - setup.py
  - config/
    - controllers.yaml
  - launch/
    - *_launch.py
  - package_name/
    - __init__.py
    - *.py

## package.xml
```xml
<?xml version="1.0"?>
<package format="3">
  <name>arm_controller</name>
  <version>0.0.1</version>
  <description>Articulate generated arm controller</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>MIT</license>
  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>trajectory_msgs</exec_depend>
  <exec_depend>control_msgs</exec_depend>
  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Common Patterns
- Node names: snake_case
- Topic names: /arm_controller/command, /arm_controller/state
- Use async spin for multi-node coordination
- Always use __init__.py in Python source dirs
