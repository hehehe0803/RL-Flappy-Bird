import matplotlib.pyplot as plt

def plot_results(episodes, all_scores, avr_scores, highest_scores):
    # Plot the data
    _, ax = plt.subplots()

    ax.plot(range(episodes), avr_scores, label='Rolling average', color='red')
    ax.plot(range(episodes), highest_scores, label='High score', color='green')
    ax.scatter(range(episodes), all_scores, label='Episode score', s=5)

    # Add legend and axis labels
    ax.legend(loc='upper left')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')

    print('======================================================================')
    for s in set(all_scores):
        percent = round((all_scores.count(s) / episodes) * 100, 2)
        print('Score {} {}%'.format(s, percent))

    plt.show()

def plot_new_states(episodes, new_states_found):
    _, ax = plt.subplots()

    ax.scatter(range(episodes), new_states_found, color='red', alpha=0.2, s=5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('New States Found')

    plt.show()