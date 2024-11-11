from src.classes import Robot, MarkovLocalization, Environment
import random
# import sys
# sys.stdout.reconfigure(encoding='utf-32')
random.seed(123)
iter_count = 50
load_map = "./assets/map_15_15.txt"

def main():
    env = Environment(
        map=load_map,
        # only used if map == None
        p_obstacle=0.4,
        size=(15,15)
    )
    #env.save_map("map_15_15.txt")
    
    robot = Robot(
        environment=env,
        location=(6, 11, 270),
        sensor_noise=0.1,
        odom_noise=0.1
    )

    localization = MarkovLocalization(
        robot=robot,
        environment=env,
        perception_chance=0.75,
    )
    env.print(robots=[robot])
    localization.print()
    
    env.display([robot])
    dir = 'down'
    
    for _ in range(iter_count):
        # move unless cannot, then turn
        moved = False
        while not moved:
            moved = robot.move()
            turned = False
            if not moved:
                while not turned:
                    turned = robot.turn(dir)
                    dir = random.choice(['up', 'down', 'left', 'right'])
            if turned or moved:
                localization.update(print_robot_perception=True)
                env.display([robot])
                localization.display()
                env.print(robots=[robot])
                localization.print()
        
    input("Press enter to exit")

if __name__ == "__main__":
    main()