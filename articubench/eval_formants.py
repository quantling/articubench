import librosa
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
from paule.util import RMSELoss

def get_formants_at_times(sound, times):
    pitch = sound.to_pitch()    
    formant = sound.to_formant_burg()
    f0, f1, f2, f3 = [], [], [], []
    for t in times:
        f0.append(pitch.get_value_at_time(t) or 0)
        f1.append(formant.get_value_at_time(1, t) or 0)  # Default to 0 if no value, otherwise we get NaNs
        f2.append(formant.get_value_at_time(2, t) or 0)
        f3.append(formant.get_value_at_time(3, t) or 0)
    return np.array(f0), np.array(f1), np.array(f2), np.array(f3)


def eval_formants(data):
    # Load target and synthesized signals
    y_target = data['target_sig']
    sr_target = data['target_sr']
    y_synth = data['synthe_sig']
    sr_synth = data['target_sr']  # Assuming the sample rate is the same for both

    # Create parselmouth sound objects
    sound_target = parselmouth.Sound(y_target, sampling_frequency=sr_target)
    sound_synth = parselmouth.Sound(y_synth, sampling_frequency=sr_synth)

    # Define time points for formant extraction
    duration = min(sound_target.get_total_duration(), sound_synth.get_total_duration())
    times = np.linspace(0, duration, num=100)

    # Extract formants
    f0_target, f1_target, f2_target, f3_target = get_formants_at_times(sound_target, times)
    f0_synth, f1_synth, f2_synth, f3_synth = get_formants_at_times(sound_synth, times)

    # Compute RMSE for each formant
    compute_rmse = RMSELoss()
    rmse_f0 = compute_rmse(f0_target, f0_synth)
    rmse_f1 = compute_rmse(f1_target, f1_synth)
    rmse_f2 = compute_rmse(f2_target, f2_synth)
    rmse_f3 = compute_rmse(f3_target, f3_synth)



    # Print RMSE values for debugging
    #print(f"RMSE F0: {rmse_f0:.2f}")
    #print(f"RMSE F1: {rmse_f1:.2f}")
    #print(f"RMSE F2: {rmse_f2:.2f}")
    #print(f"RMSE F3: {rmse_f3:.2f}")


    return {
        'rmse_f0': rmse_f0,
        'rmse_f1': rmse_f1,
        'rmse_f2': rmse_f2,
        'rmse_f3': rmse_f3
    }

