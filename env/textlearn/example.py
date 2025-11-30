import textworld.gym

# Register a text-based game as a new environment.
env_id = textworld.gym.register_game("tw_games/custom_game.z8",
                                     max_episode_steps=50)

env = textworld.gym.make(env_id)  # Start the environment.

obs, infos = env.reset()  # Start new episode.
env.render()

score, moves, done = 0, 0, False
while not done:
    command = input("> ")
    obs, score, done, infos = env.step(command)
    env.render()
    moves += 1

env.close()
print("moves: {}; score: {}".format(moves, score))