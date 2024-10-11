from FourRooms import FourRoomsEnv
from online_mirror_descent import MirrorDescent
import matplotlib.pyplot as plt
import os
import numpy as np


def color_walls_white(env, mu):
    for coor in env.grid.wall_cells:
        x = coor[1] * env.size + coor[0]
        mu[x] = -1000
    mu = mu.reshape((env.size,env.size))
    mu_masked = np.ma.masked_less(mu, 0)

    return mu_masked

def show_final_distribution(env, mu, reward_type, iter, bonus):
    _EPSILON = 10**(-3)
    # Print final distribution
    mu_traj = 0
    for step in range(40):
        mu_traj_step = color_walls_white(env, mu[step,:])
        mu_traj += mu_traj_step
    mu_traj = mu_traj/40
    plt.imshow(np.log(mu_traj+ _EPSILON))
    plt.colorbar()
    plt.title('mu traj.')
    plt.savefig('results/' + reward_type + '_mu_traj_iter_' + str(iter) + '_bonus_' + str(bonus) + '.png')
    plt.close()

    # print last time step
    mu_to_print = color_walls_white(env, mu[40-1,:])
    plt.imshow(mu_to_print)
    plt.colorbar()
    plt.title('mu')
    plt.savefig('results/' + reward_type + '_mu_dist_iter_' + str(iter) + '_bonus_' + str(bonus) + '.png')
    plt.close()

def plot_graphs(samples, iterations, optimal, reward_type):
    # plot regret and log-loss with confidence bounds the curves were created
    loss_bonus_true = np.zeros((samples,iterations))
    loss_bonus_false = np.zeros((samples,iterations))
    regret_bonus_true = np.zeros((samples,iterations))
    regret_bonus_false = np.zeros((samples,iterations))

    for i in range(samples):
        loss_bonus_true[i,:] = np.load('curves/' + reward_type + '_bonus_True_sample_' + str(i) + '.npy')
        loss_bonus_false[i,:] = np.load('curves/' + reward_type +  '_bonus_False_sample_' + str(i) + '.npy')
        regret_bonus_true[i,:] = np.cumsum(np.load('curves/' + reward_type + '_bonus_True_sample_' + str(i) + '.npy') - optimal)
        regret_bonus_false[i,:] = np.cumsum(np.load('curves/' + reward_type + '_bonus_False_sample_' + str(i) + '.npy') - optimal)

    regret_true = np.mean(regret_bonus_true, axis=0)
    regret_false = np.mean(regret_bonus_false, axis=0)
    loss_true = np.mean(loss_bonus_true, axis =0)
    loss_false = np.mean(loss_bonus_false, axis=0)
    # confidence interval
    ci_regret_true = 1.96 * np.std(regret_bonus_true, axis=0)/np.sqrt(samples)
    ci_regret_false = 1.96 * np.std(regret_bonus_false, axis=0)/np.sqrt(samples)
    ci_loss_true = 1.96 * np.std(loss_bonus_true, axis=0)/np.sqrt(samples)
    ci_loss_false = 1.96 * np.std(loss_bonus_false, axis=0)/np.sqrt(samples)

    # plot log losses
    x = np.arange(0, iterations)
    plt.loglog(loss_false, label='Greedy MD-CURL', c='r')
    plt.fill_between(x, (loss_false - ci_loss_false), (loss_false+ci_loss_false), color='r', alpha=.1)
    plt.loglog(loss_true, label='Bonus O-MD-CURL', c='b')
    plt.fill_between(x, (loss_true-ci_loss_true), (loss_true+ci_loss_true), color='b', alpha=.1)
    plt.xlabel('Iteration')
    plt.title('Log-loss per iteration')
    plt.legend()
    plt.savefig('results/log_loss_' + reward_type + '.png')
    plt.close()

    # plot regret
    x = np.arange(0, iterations)
    plt.plot(regret_false, label='Greedy MD-CURL', c='r')
    plt.fill_between(x, (regret_false - ci_regret_false), (regret_false+ci_regret_false), color='r', alpha=.1)
    plt.plot(regret_true, label='Bonus O-MD-CURL', c='b')
    plt.fill_between(x, (regret_true-ci_regret_true), (regret_true+ci_regret_true), color='b', alpha=.1)
    plt.xlabel('Iteration')
    plt.title('Regret')
    plt.legend()
    plt.savefig('results/regret_' + reward_type  + '.png')
    plt.close()

def main(reward_type, n_iterations, samples=1):
    # Create directory to add results if it does not exist
    isExist = os.path.exists('results')
    if not isExist:
        os.makedirs('results')
    
    # Create directory to add curves if it does not exist
    isExist = os.path.exists('curves')
    if not isExist:
        os.makedirs('curves')    

    for i in range(samples):
        for bonus in [True, False]:
            # define environment 
            env = FourRoomsEnv(max_steps=40)
            obs = env.reset()
            P_model = env.P(env.p) # true probability transition kernel
            model = MirrorDescent(env, P_model, reward_type=reward_type)

            if n_iterations >= 100:
                # save state-action distribution after 50 iterations
                model.iteration(n_iterations=40, bonus=bonus)
                mu = model.mu_induced(model.policy, P_model)
                # save state-action distribution figure for only one repetition
                if i == 0:
                    show_final_distribution(env, mu, reward_type, model.count_step, bonus)

                # after all iterations
                model.iteration(n_iterations=n_iterations-50, bonus=bonus)
                mu = model.mu_induced(model.policy, P_model)
                # save state-action distribution figure for only one repetition
                if i == 0:
                    show_final_distribution(env, mu, reward_type, model.count_step, bonus)
            else:
                # save only after all iterations
                model.iteration(n_iterations=n_iterations, bonus=bonus)
                mu = model.mu_induced(model.policy, P_model)
                # save state-action distribution figure for only one repetition
                if i == 0:
                    show_final_distribution(env, mu, reward_type, model.count_step, bonus)

            # save the losses at each iteration
            np.save('curves/' + reward_type + '_bonus_' + str(bonus) + '_sample_' + str(i) + '.npy', model.error_steplast)
        
    # plot the graphs
    optimal = np.load('optimal/optimal_' + reward_type + '.npy')
    plot_graphs(samples, n_iterations, optimal, reward_type)

    


if __name__ == '__main__':
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_type', type=str, required=True, help='type of reward: constrained or multi_objectives')
    parser.add_argument('--n_iterations', type=int, required=True, help='number of iterations')
    parser.add_argument('--samples', type=int, required=False, help='number of repetitions per experiment, default is 1')
    args = parser.parse_args()

    main(reward_type=args.reward_type, n_iterations=args.n_iterations, samples=args.samples)
