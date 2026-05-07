import numpy as np 
import matplotlib.pyplot as plt


def reward(d_goal, G_h):

    # Goal-oriented Reward Function
    A = np.mod((0.5 * (G_h + np.pi)), 2 * np.pi)
    if d_goal > 0.5 or d_goal <= 0:
        reward = -10
    elif 0 < d_goal <= 0.5:
        reward = 200. * d_goal * (1 - 4 * np.abs(0.5 - np.mod((A/np.pi), 1)))

    return reward 

distance_goals = [0.01, 0.02, 0.03, 0.04, 0.05, 0]
headings = [-2*np.pi, -1.5*np.pi, -1*np.pi, -0.5*np.pi, 0*np.pi,
           0.5*np.pi, 1*np.pi, 1.5*np.pi, 2*np.pi]


plt.figure(figsize=(8, 4))

for d in distance_goals:
    rewards = [reward(d, h) for h in headings]
    plt.plot(headings, rewards, linewidth=2, label = rf'$\Delta d_{{{"Goal"}}} = {d}$')

plt.xlabel('Angles/Heading (rad)', fontsize=8)
plt.ylabel('Angular Reward', fontsize=8)
plt.xticks(
    [-2*np.pi, -1.5*np.pi, -1*np.pi, -0.5*np.pi, 0*np.pi,
     0.5*np.pi, 1*np.pi, 1.5*np.pi, 2*np.pi],
    [r'${-2}{\pi}$', r'${-1.5}{\pi}$', r'${-1}{\pi}$', r'${-0.5}{\pi}$',
     r'${0}{\pi}$', r'${0.5}{\pi}$', r'${1}{\pi}$', r'${1.5}{\pi}$', r'${2}{\pi}$']
)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.title('Goal Angular + Distance Reward', fontsize=8)
plt.legend(loc = "upper right", fontsize=8)
plt.tight_layout()
plt.show()