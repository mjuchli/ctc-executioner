from gym.envs.registration import register

register(
    id='ctc-executioner-v0',
    entry_point='gym_ctc_executioner.envs:ExecutionEnv'
)
