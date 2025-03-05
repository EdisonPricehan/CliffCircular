import gymnasium as gym
from cliffcircular.cliffcircular import CliffCircularEnv


def main():
    # Create the environment
    env = gym.make(
        'CliffCircular-v1',
        render_mode='human',  # or 'rgb_array' or 'ansi'
        extra_cliff_num=1,  # or 0 or 1
        cost_downscale_denom=5,
    )

    # Reset the environment
    observation, info = env.reset()
    done = False

    while not done:
        # Optionally, render the environment
        env.render()

        # Choose an action (e.g., randomly)
        action = env.action_space.sample()

        # Step the environment
        observation, reward, done, truncated, info = env.step(action)

        # Retrieve the step cost from the info dictionary
        cost = info.get("cost", 0)

        # Print observation, reward, and step cost
        print(f"Observation: {observation}, Reward: {reward}, Step Cost: {cost}")

    env.close()


if __name__ == "__main__":
    main()
