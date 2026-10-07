Food Delivery RL v4 - DQN
=========================

Action space: Discrete(9)

0 Left Strong   (-0.8, 0.55)
1 Left Mild     (-0.4, 0.55)
2 Straight      ( 0.0, 0.55)
3 Right Mild    ( 0.4, 0.55)
4 Right Strong  ( 0.8, 0.55)
5 Slow Left     (-0.4, 0.25)
6 Slow Straight ( 0.0, 0.25)
7 Slow Right    ( 0.4, 0.25)
8 Stop          ( 0.0, 0.00)

Why 9 actions?
The circular laboratory track may require sustained turning and slow turning.
This design gives DQN both normal-speed and slow-speed turning options.

Pickup/drop-off logic:
- Entering a target zone alone is NOT sufficient.
- The agent must choose action 8 (STOP) in the correct zone.
- Pickup + STOP -> +10
- Correct drop-off + STOP -> +30
- Wrong drop-off + STOP -> -10
- Unnecessary STOP -> -0.10

Install:
    python -m pip install numpy gymnasium stable-baselines3 torch tensorboard

Run:
    python test_dqn_core.py
    python test_dqn_env.py
    python test_dqn_smoke.py

Train:
    python train_dqn.py

Important:
The fake simulator only validates interfaces and DQN integration.
Real algorithm performance must be evaluated after DonkeySim integration.
