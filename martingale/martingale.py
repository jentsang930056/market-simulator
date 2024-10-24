

import numpy as np
import matplotlib.pyplot as plt


def author():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    return "ytsang6"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    return 903949675  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		  		 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def test_code():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    win_prob = 18/38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments
    bankroll = False
    plot_1(win_prob, bankroll)
    plot_2(win_prob, bankroll)
    plot_3(win_prob, bankroll)
    bankroll = True
    plot_4(win_prob, bankroll)
    plot_5(win_prob, bankroll)


# ===================================================================================

# Simple Gambling Simulator
def gambling_simulator(win_prob, bankroll):
    arr = np.zeros(1001)
    arr[1:] = 80
    round = 1
    episode_winnings = 0
    while episode_winnings < 80:  # reach 80 and stop
        won = False
        bet_amount = 1
        while not won:
            if round == 1001:
                return arr # over 1000 round will stop
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                if bankroll == True:
                    if episode_winnings == -256:
                        bet_amount = 0
                    else:
                        if (episode_winnings - (bet_amount * 2)) >= -256:
                            bet_amount *= 2
                        else:
                            bet_amount = episode_winnings - (-256)
                else:
                    bet_amount *= 2

            arr[round] = episode_winnings
            round += 1

    return arr


#10 episodes run with unlimited bankroll
def plot_1(win_prob, bankroll):
    episode = 10
    plt.title("Exp1 - Fig1 show 10 episodes")
    plt.xlabel("bet count")
    plt.ylabel("winning")
    for i in range(episode):
        plt.plot(np.arange(0, 1001), gambling_simulator(win_prob, bankroll))
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.legend(['epi_1', 'epi_2', 'epi_3', 'epi_4', 'epi_5', 'epi_6', 'epi_7', 'epi_8', 'epi_9', 'epi_10'])
    plt.savefig("images/Figure 1.png")
    plt.close()

#mean and standard deviation with unlimited bankroll
def plot_2(win_prob, bankroll):
    i = 0
    twod_array = np.zeros((1000, 1001))
    while i < 1000:
        twod_array[i] = gambling_simulator(win_prob, bankroll)
        i += 1

    mean = np.mean(twod_array, 0)
    std_dev = np.std(twod_array, 0)
    m_plus_std = mean + std_dev
    m_minus_std = mean - std_dev

    plt.title("Exp1 - Fig2 mean and std_dev")
    plt.xlabel("bet count")
    plt.ylabel("winning")
    plt.xlim([0, 300])
    plt.ylim([-256, 100])

    plt.plot(mean)
    plt.plot(m_plus_std)
    plt.plot(m_minus_std)
    plt.legend(['mean', 'mean+std_dev', 'mean-std_dev'])
    plt.savefig("images/Figure 2.png")
    plt.close()


#median and standard deviation with unlimited bankroll
def plot_3(win_prob, bankroll):
    i = 0
    twod_array = np.zeros((1000, 1001))
    while i < 1000:
        twod_array[i] = gambling_simulator(win_prob, bankroll)
        i += 1

    med = np.median(twod_array, 0)
    std_dev = np.std(twod_array, 0)
    m_plus_std = med + std_dev
    m_minus_std = med - std_dev

    plt.title("Exp1 - Fig3 median and std_dev")
    plt.xlabel("bet count")
    plt.ylabel("winning")
    plt.xlim([0, 300])
    plt.ylim([-256, 100])

    plt.plot(med)
    plt.plot(m_plus_std)
    plt.plot(m_minus_std)
    plt.legend(['med', 'med+std_dev', 'med-std_dev'])
    plt.savefig("images/Figure 3.png")
    plt.close()

#mean and standard deviation with limited bankroll
def plot_4(win_prob, bankroll):
    i = 0
    twod_array = np.zeros((1000, 1001))
    while i < 1000:
        twod_array[i] = gambling_simulator(win_prob, bankroll)
        i += 1

    mean = np.mean(twod_array, 0)
    std_dev = np.std(twod_array, 0)
    m_plus_std = mean + std_dev
    m_minus_std = mean - std_dev

    plt.title("Exp2 - Fig4 realistic mean and std_dev")
    plt.xlabel("bet count")
    plt.ylabel("winning")
    plt.xlim([0, 300])
    plt.ylim([-256, 100])

    plt.plot(mean)
    plt.plot(m_plus_std)
    plt.plot(m_minus_std)
    plt.legend(['mean', 'mean+std_dev', 'mean-std_dev'])
    plt.savefig("images/Figure 4.png")
    plt.close()

#median and standard deviation with limited bankroll
def plot_5(win_prob, bankroll):
    i = 0
    twod_array = np.zeros((1000, 1001))
    while i < 1000:
        twod_array[i] = gambling_simulator(win_prob, bankroll)
        i += 1

    med = np.median(twod_array, 0)
    std_dev = np.std(twod_array, 0)
    m_plus_std = med + std_dev
    m_minus_std = med - std_dev

    plt.title("Exp2 - Fig5 realistic median and std_dev ")
    plt.xlabel("bet count")
    plt.ylabel("winning")
    plt.xlim([0, 300])
    plt.ylim([-256, 100])

    plt.plot(med)
    plt.plot(m_plus_std)
    plt.plot(m_minus_std)
    plt.legend(['med', 'med+std_dev', 'med-std_dev'])
    plt.savefig("images/Figure 5.png")
    plt.close()


if __name__ == "__main__":
    test_code()
