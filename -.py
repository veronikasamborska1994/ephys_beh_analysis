import Esync as es

# Instantiate aligner by passing it the times of the pulses.
aligner = es.Rsync_aligner(pulse_times_A, pulse_times_B, plot=True)

# Convert times from system A to B.
times_B = aligner.A_to_B(times_B)

# Convert times from system B to A.
times_A = aligner.B_to_A(times_A)