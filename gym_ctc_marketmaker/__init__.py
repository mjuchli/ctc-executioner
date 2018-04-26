from gym.envs.registration import register

register(
    id='ctc-marketmaker-v0',
    entry_point='gym_ctc_marketmaker.envs:MarketMakerEnv'
)
