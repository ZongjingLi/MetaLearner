import gym
import smartplay
env_name = "Crafter"
env = gym.make("smartplay:{}-v0".format(env_name))
_, info = env.reset()


while True:
    #action = info['action_space'].sample()
    action = env.action_space.sample()
    _, reward, done, info = env.step(action)
    manual, obs, history, score = info['manual'], info['obs'], info['history'], info['score']
    if not done:
        completion=0
    else:
        completion=info['completed']
    print(obs)
"""
['MessengerL1-v0',
 'MessengerL2-v0',
 'MessengerL3-v0',
 'RockPaperScissorBasic-v0',
 'RockPaperScissorDifferentScore-v0',
 'BanditTwoArmedDeterministicFixed-v0',
 'BanditTwoArmedHighHighFixed-v0',
 'BanditTwoArmedHighLowFixed-v0',
 'BanditTwoArmedLowLowFixed-v0',
 'Hanoi3Disk-v0',
 'Hanoi4Disk-v0',
 'MinedojoCreative0-v0',
 'MinedojoCreative1-v0',
 'MinedojoCreative2-v0',
 'MinedojoCreative4-v0',
 'MinedojoCreative5-v0',
 'MinedojoCreative7-v0',
 'MinedojoCreative8-v0',
 'MinedojoCreative9-v0',
 'Crafter-v0']
"""