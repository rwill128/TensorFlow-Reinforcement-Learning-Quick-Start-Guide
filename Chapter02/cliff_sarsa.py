import numpy as np
import sys
import matplotlib.pyplot as plt

nrows = 3
ncols = 12
nact = 4

nepisodes = 100000
epsilon = 0.1
alpha = 0.1
gamma = 0.95

reward_normal = -1
reward_cliff = -100
reward_destination = -1

# ---------------------------------------------------

Q_lookup = np.zeros((nrows, ncols, nact), dtype=np.float)


def go_to_start():
    # start coordinates
    y = nrows
    x = 0
    return x, y


def random_action():
    # a = 0 : top/north
    # a = 1 : right/east
    # a = 2 : bottom/south
    # a = 3 : left/west
    a = np.random.randint(nact)
    return a


def move(x, y, a):
    # state = 0: OK
    # state = 1: reached destination
    # state = 2: fell into cliff
    state = 0

    if x == 0 and y == nrows and a == 0:
        # start location
        next_x = x
        next_y = y - 1
        return next_x, next_y, state
    elif x == ncols - 1 and y == nrows - 1 and a == 2:
        # reached destination
        next_x = x
        next_y = y + 1
        state = 1
        return next_x, next_y, state
    else:
        if a == 0:
            next_x = x
            next_y = y - 1
        elif a == 1:
            next_x = x + 1
            next_y = y
        elif a == 2:
            next_x = x
            next_y = y + 1
        elif a == 3:
            next_x = x - 1
            next_y = y
        if next_x < 0:
            next_x = 0
        if next_x > ncols - 1:
            next_x = ncols - 1
        if next_y < 0:
            next_y = 0
        if next_y > nrows - 1:
            state = 2
        return next_x, next_y, state


def exploit(x, y, Q):
    # start location
    if x == 0 and y == nrows:
        a = 0
        return a
        # destination location
    if x == ncols - 1 and y == nrows - 1:
        a = 2
        return a
    if x == ncols - 1 and y == nrows:
        print("exploit at destination not possible ")
        sys.exit()
    # interior location
    if x < 0 or x > ncols - 1 or y < 0 or y > nrows - 1:
        print("error ", x, y)
        sys.exit()
    a = np.argmax(Q[y, x, :])
    return a


def bellman(x, y, a, reward, Qs1a1, Q):
    if y == nrows and x == 0:
        # at start location; no Bellman update possible
        return Q
    if y == nrows and x == ncols - 1:
        # at destination location; no Bellman update possible
        return Q
    Q[y, x, a] = Q[y, x, a] + alpha * (reward + gamma * Qs1a1 - Q[y, x, a])
    return Q


def max_Q(x, y, Q_table):
    a = np.argmax(Q_table[y, x, :])
    return Q_table[y, x, a]


def explore_exploit(x, y, Q_table):
    # if we end up at the start location, then exploit
    if x == 0 and y == nrows:
        action = 0
        return action

    r = np.random.uniform()
    if r < epsilon:
        # explore
        action = random_action()
    else:
        # exploit
        action = exploit(x, y, Q_table)
    return action


# ---------------------------------------------------

for n in range(nepisodes + 1):
    if n % 1000 == 0:
        print("episode #: ", n)
    x, y = go_to_start()

    chosen_action = explore_exploit(x, y, Q_lookup)

    while True:
        next_x, next_y, state = move(x, y, chosen_action)
        if state == 1:
            reward = reward_destination
            value_for_current_state = 0.0
            Q_lookup = bellman(x, y, chosen_action, reward, value_for_current_state, Q_lookup)
            break
        elif state == 2:
            reward = reward_cliff
            value_for_current_state = 0.0
            Q_lookup = bellman(x, y, chosen_action, reward, value_for_current_state, Q_lookup)
            break
        elif state == 0:
            reward = reward_normal
            # Sarsa
            next_chosen_action = explore_exploit(next_x, next_y, Q_lookup)
            if next_x == 0 and next_y == nrows:
                # start location
                value_for_current_state = 0.0
            else:
                value_for_current_state = Q_lookup[next_y, next_x, next_chosen_action]

            Q_lookup = bellman(x, y, chosen_action, reward, value_for_current_state, Q_lookup)
            x = next_x
            y = next_y
            chosen_action = next_chosen_action

        # ---------------------------------------------------
for i in range(nact):
    plt.subplot(nact, 1, i + 1)
    plt.imshow(Q_lookup[:, :, i])
    plt.axis('off')
    plt.colorbar()
    if i == 0:
        plt.title('Q-north')
    elif i == 1:
        plt.title('Q-east')
    elif i == 2:
        plt.title('Q-south')
    elif i == 3:
        plt.title('Q-west')
plt.savefig('Q_sarsa.png')
plt.clf()
plt.close()
# ---------------------------------------------------

# path planning

path = np.zeros((nrows, ncols, nact), dtype=np.float)

x, y = go_to_start()
while True:
    chosen_action = exploit(x, y, Q_lookup)
    print(x, y, chosen_action)
    next_x, next_y, state = move(x, y, chosen_action)
    if state == 1 or state == 2:
        print("breaking ", state)
        break
    elif state == 0:
        x = next_x
        y = next_y
        if 0 <= x <= ncols - 1 and 0 <= y <= nrows - 1:
            path[y, x] = 100.0

path = np.array(path).astype(np.uint8)

plt.imshow(path)
plt.savefig('path_sarsa.png')

print("done")
# ---------------------------------------------------
