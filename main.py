from src.classes import Robot, Environment, FastSLAM
import keyboard

load_map = "./assets/map_slam.txt"

def main():
    env = Environment(
        map=load_map,
    )
    
    robot = Robot(
        environment=env,
        location=(10, 11, 0),
        sensor_noise=0.0
    )
    
    slam = FastSLAM(
        env_size=env.grid.shape,
        particle_count=15,
        approx_pos=(10.5, 11.3, 15))
    
    print("""
    Press 'left' or 'right' arrow key to rotate.
    Press 'up' arrow key to move in facing direction.
    Press 'q' or CTRL-C to exit.
""")
    
    # initial update from observaitons
    slam.update_landmarks(robot.get_perception(ret_coords=True))
    slam.resample_particles(normalize=True)
    slam.display()
    while True:
        try:
            diff = None
            if keyboard.is_pressed('up'):
                diff = robot.move()
            elif keyboard.is_pressed('left'):
                diff = robot.turn('left')
            elif keyboard.is_pressed('right'):
                diff = robot.turn('right')
            elif keyboard.is_pressed('q'):
                break
            
            if diff:
                slam.update_movement(diff)
                slam.update_landmarks(robot.get_perception(ret_coords=True))
                slam.resample_particles(normalize=True)

            # update map
            env.display([robot])
            slam.display()

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()