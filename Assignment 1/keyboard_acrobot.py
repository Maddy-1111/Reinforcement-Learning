import gymnasium as gym
import pygame
import sys

def main():
    env = gym.make("Acrobot-v1", render_mode="human")
    obs, _ = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    done = False

    print("Controls:")
    print("Left Arrow  -> torque -1")
    print("Right Arrow -> torque +1")
    print("No key      -> torque 0")
    print("Press ESC to quit")

    while True:

        action = 1  # default: 0 torque (action index 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            action = 0   # torque -1
        elif keys[pygame.K_RIGHT]:
            action = 2   # torque +1
        elif keys[pygame.K_ESCAPE]:
            env.close()
            pygame.quit()
            sys.exit()

        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(60)  # limit to 60 FPS

if __name__ == "__main__":
    main()