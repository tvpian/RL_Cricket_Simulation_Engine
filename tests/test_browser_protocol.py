import gymnasium as gym
import numpy as np
from cricket_rl.browser_protocol import Env, Discrete, Box


def test_browser_protocol_seed_matches_desktop_gym():
    browser,desktop=Env(),gym.Env()
    browser.reset(seed=42);desktop.reset(seed=42)
    np.testing.assert_array_equal(browser.np_random.random(20),desktop.np_random.random(20))
    assert Discrete(9).contains(8)
    assert not Discrete(9).contains(9)
    assert not Discrete(9).contains(1.5)
    box=Box(0,1,(2,),np.float32)
    assert box.contains(np.array([0,1],dtype=np.float32))
    assert not box.contains(np.array([0,2],dtype=np.float32))
