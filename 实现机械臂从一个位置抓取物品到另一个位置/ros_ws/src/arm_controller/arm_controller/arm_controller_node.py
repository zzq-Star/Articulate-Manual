import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from arm_controller.arm_kinematics import ArmKinematics

from arm_controller.trajectory_planner import TrajectoryPlanner


class ArmController(Node):
    """ROS2 node for six_dof_standard arm control."""

    def __init__(self):
        super().__init__('arm_controller_node')

        # Declare parameters
        self.declare_parameter('control_rate', 100.0)
        self.declare_parameter('joint_velocity_limit', 1.0)
        self.declare_parameter('joint_acceleration_limit', 0.5)
        self.declare_parameter('timeout_seconds', 5.0)

        
        self.kinematics = ArmKinematics()
        
        self.planner = TrajectoryPlanner()

        # QoS profile for reliable command delivery
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )

        # Publishers
        
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/command', cmd_qos)
        
        self.state_pub = self.create_publisher(
            JointState, '/arm_controller/state', cmd_qos)
        

        # Subscribers
        
        self.command_sub = self.create_subscription(
            JointTrajectory, '/arm_controller/goal',
            self.on_goal, cmd_qos)
        

        # Timer for control loop
        rate = self.get_parameter('control_rate').value
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        # State
        self.current_joint_state = None
        self.active_trajectory = None
        self.trajectory_start_time = None

        self.get_logger().info('ArmController initialized')

    def on_goal(self, msg: JointTrajectory):
        """Handle incoming trajectory goal."""
        if not msg.points:
            self.get_logger().warn('Received empty trajectory')
            return

        # Validate trajectory
        if not self._validate_trajectory(msg):
            self.get_logger().error('Trajectory validation failed')
            return

        self.active_trajectory = msg
        self.trajectory_start_time = self.get_clock().now()
        self.get_logger().info('Accepted new trajectory goal')

    def control_loop(self):
        """Main control loop - execute active trajectory."""
        if self.active_trajectory is None:
            return

        # Check timeout
        timeout = self.get_parameter('timeout_seconds').value
        elapsed = (self.get_clock().now() - self.trajectory_start_time).nanoseconds / 1e9
        if elapsed > timeout:
            self.get_logger().warn('Trajectory timeout, aborting')
            self.active_trajectory = None
            return

        # Find current trajectory point
        for i, point in enumerate(self.active_trajectory.points):
            if point.time_from_start.sec + point.time_from_start.nanosec / 1e9 >= elapsed:
                self._publish_command(point)
                return

        # Trajectory complete
        if self.active_trajectory.points:
            self._publish_command(self.active_trajectory.points[-1])
        self.get_logger().info('Trajectory complete')
        self.active_trajectory = None

    def _publish_command(self, point: JointTrajectoryPoint):
        """Publish joint command."""
        cmd = JointTrajectory()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.joint_names = [f'joint_{i}' for i in range(len(point.positions))]
        cmd.points = [point]
        self.trajectory_pub.publish(cmd)

    def _validate_trajectory(self, msg: JointTrajectory) -> bool:
        """Validate trajectory safety constraints."""
        vel_limit = self.get_parameter('joint_velocity_limit').value
        accel_limit = self.get_parameter('joint_acceleration_limit').value

        for point in msg.points:
            if point.velocities:
                if any(abs(v) > vel_limit for v in point.velocities):
                    self.get_logger().error(f'Velocity limit exceeded: {max(point.velocities)} > {vel_limit}')
                    return False
            if point.accelerations:
                if any(abs(a) > accel_limit for a in point.accelerations):
                    self.get_logger().error(f'Acceleration limit exceeded: {max(point.accelerations)} > {accel_limit}')
                    return False
        return True


def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
