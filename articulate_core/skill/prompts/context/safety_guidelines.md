# Safety Guidelines for Code Generation
# ======================================
# These guidelines MUST be followed when generating any robotic arm control code.

## Hard Constraints
1. Joint limits: ALL generated trajectories must stay within joint limit bounds.
2. Velocity limits: commanded velocities must not exceed 90% of rated limits.
3. Torque limits: commanded torques must not exceed 95% of rated limits.
4. Singularity: generated paths must include singularity detection and avoidance.

## Code Patterns
1. All motion commands must include pre-check validation.
2. All control nodes must implement a timeout/watchdog mechanism.
3. Emergency stop input must be checked before each motion command.
4. Path pre-computation must be completed before execution begins.
5. All numerical operations must use numpy for type safety.

## ROS2 Specific
1. Use QoSSettings with reliable delivery for command topics.
2. Configure controller with proper safety limits in YAML.
3. Lifecycle nodes preferred for production code.
4. Parameter declarations for all tunable values.

## Deployment
1. First run must be at 10% speed with no load.
2. Workspace boundaries must be validated before first motion.
3. E-stop must be tested before any automated motion.
4. Always include a safety pre-check routine.
