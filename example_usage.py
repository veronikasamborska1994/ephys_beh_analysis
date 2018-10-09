import data_import as di
import align_activity as aa

session = di.Session('m479-2018-08-12-150904.txt')

spikes = np.load('m479_2018-08-12_15-08-53.npy')

spikes = spikes[:,~np.isnan(spikes[1,:])] # Array [2, n_spikes] of neuron IDs and spike times.

init_times = session.times['choice_state']

inits_and_choices = [ev for ev in session.events if ev.name in 
                    ['choice_state', 'sound_a_reward', 'sound_b_reward',
                     'sound_a_no_reward','sound_b_no_reward']]

choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                         i>0 and inits_and_choices[i-1].name == 'choice_state'])

# Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
# Here we are using [init-1000, init, choice, choice+1000]
trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T

# target_times is the reference times to warp all trials to. Shape: [n_ref_points]
# Here we are using the median timings from this session.
target_times = np.hstack(([0], np.cumsum(np.median(np.diff(trial_times,1),0))))

# Calculate trial aligned firing rates.
aligned_rates, t_out, min_max_stretch = aa.align_activity(trial_times, target_times, spikes, plot=True)