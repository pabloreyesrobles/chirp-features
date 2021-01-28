from scipy.signal import hann, find_peaks, hilbert
from scipy.optimize import curve_fit, OptimizeWarning
from spikelib import spiketools as spkt
from functools import reduce
from tqdm import tqdm

import numpy as np
import pandas as pd
import warnings

def matlab_hanning(n):
    return hann(n + 2)[1:-1]

def get_freq_spikes(spks, ts, t0, duration, win_size):
    """
    Windowed histogram. JG code adapted to python

    Parameters
    ----------
    spks: Spiketimes array
    ts: sample time
    t0: start time
    duration: duration time
    win_size: windows size array

    """
    time_units = int(round(duration / ts))
    time = np.linspace(0, time_units * ts, time_units)
    
    result = np.zeros(time_units)
    spks_by_win = np.zeros([len(win_size), time_units])
    
    for key, win in enumerate(win_size):
        win_time = (win + 1) * ts
        for i in range(round(win / 2), time_units - round(win / 2)):
            loop = spks[abs(spks - t0 - time[i]) <= (win / 2) * ts]
            
            if len(loop) >= 2:
                #TODO: check with other windows. May be setted
                signal = (np.cos(np.pi * (loop - t0 - time[i]) / win_time)**2 / 
                          (np.sum(matlab_hanning(win + 1)) * ts))
                spks_by_win[key, i] = np.sum(signal)
            
            arr = spks_by_win[:, i]
            arr = arr[~np.isnan(arr)]
            if arr.size > 0:
                result[i] = np.max(arr)
                
    return result

def chirp_generator(sample_rate, args, splitted=False):
    adap_units = args['base_amp'] * np.ones(int(args['t_adap'] / sample_rate))
    
    on_units = args['amp_on'] * np.ones(int(args['t_on'] / sample_rate))
    off_units = np.zeros(int(args['t_off'] / sample_rate))
    
    freq_acc = args['freq_mod_final_freq'] / args['freq_mod_time']
    freq_time = np.linspace(0, args['freq_mod_time'] - sample_rate, int(args['freq_mod_time'] / sample_rate))
    freq_units = args['freq_mod_amp'] * np.sin(args['freq_mod_init_phase'] + np.pi * freq_acc * np.multiply(freq_time, freq_time))
    freq_units = freq_units + args['base_amp']
    
    amp_time = np.linspace(0, args['amp_mod_time'] - sample_rate, int(args['amp_mod_time'] / sample_rate))
    amp_units = np.sin(2 * np.pi * args['amp_mod_freq'] * amp_time)
    amp_time = np.linspace(0, args['amp_mod_max'], int(args['amp_mod_time'] / sample_rate))
    amp_units = np.multiply(amp_units, amp_time)
    amp_units = amp_units + args['base_amp']

    # TODO: make it adaptable to the chirp parameters
    line_limit = int(np.floor((args['freq_mod_time']**2 - 0.5) * 0.5))
    freq_mod_max_lines = [np.sqrt(0.5 + 2 * i) for i in range(line_limit)]
    freq_mod_min_lines = [np.sqrt(-0.5 + 2 * i) for i in range(1, line_limit)]
    freq_mod_zero_lines = [np.sqrt(2 * i) for i in range(line_limit)]

    line_limit = int(np.round(args['amp_mod_time'] * args['amp_mod_freq'] - 0.25))
    amp_mod_max_lines = [((0.25 + i) / args['amp_mod_freq']) for i in range(line_limit + 1)]
    amp_mod_min_lines = [((-0.25 + i + 1) / args['amp_mod_freq']) for i in range(line_limit)] # i + 1 need 

    chirp_signal = {}

    on_off_units = np.concatenate((on_units, off_units))
    chirp_signal['on_off'] = (np.linspace(0, args['t_on'] + args['t_off'] - sample_rate, len(on_off_units)), on_off_units)
    chirp_signal['adap_0'] = (np.linspace(0, args['t_adap'] - sample_rate, len(adap_units)), adap_units)
    chirp_signal['freq'] = (np.linspace(0, args['freq_mod_time'] - sample_rate, len(freq_units)), freq_units)
    chirp_signal['adap_1'] = (np.linspace(0, args['t_adap'] - sample_rate, len(adap_units)), adap_units)
    chirp_signal['amp'] = (np.linspace(0, args['amp_mod_time'] - sample_rate, len(amp_units)), amp_units)
    chirp_signal['adap_2'] = (np.linspace(0, args['t_adap'] - sample_rate, len(adap_units)), adap_units)

    chirp_signal['freq_mod_max_lines'] = freq_mod_max_lines
    chirp_signal['freq_mod_min_lines'] = freq_mod_min_lines
    chirp_signal['freq_mod_zero_lines'] = freq_mod_zero_lines

    chirp_signal['amp_mod_max_lines'] = amp_mod_max_lines
    chirp_signal['amp_mod_min_lines'] = amp_mod_min_lines

    chirp_signal['full_signal'] = np.concatenate((on_units, off_units, adap_units, freq_units, adap_units, amp_units, adap_units))
    chirp_signal['full_time'] = np.linspace(0, len(chirp_signal) * sample_rate - sample_rate, len(chirp_signal))

    return chirp_signal

def get_chirp_subevents(sync_path, start_end_path, repeated_path, output_file, names, times):
    sync = pd.read_csv(sync_path)
    start_end = np.loadtxt(start_end_path, dtype=int)
    start_frame, end_frame = start_end[:, 0], start_end[:, 1]
    repetead = np.loadtxt(repeated_path, dtype=int)

    protocol_name = 'chirp'  # name of protocol in event_list file

    chirp_filter = sync['protocol_name'] == protocol_name
    chirp_times = sync[chirp_filter]

    if chirp_times.shape[0] == 0:
        return None

    sr = 20000.0
    fps = 60

    start_trans = np.array([[0] + times[:-1]]).cumsum() * fps
    end_trans = np.array(times).cumsum() * fps - 1

    events_chirp = []
    for _, kchirp in enumerate(chirp_times.itertuples()):
        rep_trans = np.zeros_like(times)
        sub_set = np.logical_and(start_frame >= kchirp.start_event, 
                                 end_frame <= kchirp.end_event)
        
        sub_start_frame = start_frame[sub_set]
        sub_end_frame = end_frame[sub_set]
        
        for ktrans, (start, end) in enumerate(zip(start_trans, end_trans)):
            rep_trans[ktrans] = np.logical_and(
                repetead >= sub_start_frame[start + rep_trans[:ktrans].sum()],
                repetead <= sub_end_frame[end + rep_trans[:ktrans].sum()]
            ).sum()
                
        idx = np.where(start_frame == kchirp.start_event)[0][0]
        start_event = start_frame[idx + start_trans + rep_trans]
        end_event = end_frame[idx + end_trans + rep_trans]
        
        df = pd.DataFrame(
            {
                'n_frames': end_trans - start_trans + rep_trans + 1,
                'start_event': start_event,
                'end_event': end_event,
                'start_next_event': end_event,
                'event_duration': end_event - start_event,
                'event_duration_seg': (end_event - start_event) / sr,
                'inter_event_duration': 0,
                'inter_event_duration_seg': 0.0,
                'protocol_name': protocol_name,
                'extra_description': names,
                'rep_name': kchirp.repetition_name,
                'repetead_frames': '',
                '#repetead_frames': rep_trans,
            })
        events_chirp.append(df)
    events = reduce(lambda x, y: pd.concat([x, y]), events_chirp)
    events.to_csv(output_file, sep=';', index=False)
    return events

def get_freq_peaks(resp):
    fmin = np.mean(resp) * 0.2
    freq_peaks, _ = find_peaks(resp, height=fmin, prominence=4) # heuristica

    return freq_peaks

def freq_on_peak_sel(resp, bins_fit, min_lines, max_lines, peaks):
    dt_0 = 0.2
    dt_1 = max_lines[0]

    # Para graficar
    dt = []

    t_spike = np.empty(len(max_lines))
    t_spike[:] = np.nan

    val_spike = np.empty(len(max_lines))
    val_spike[:] = np.nan

    delay = np.empty(len(max_lines))
    delay[:] = np.nan

    # def dt_0_factor(idx):
    #    return np.exp(-idx) * 0.8 + 0.05
    # def dt_1_factor(idx):
    #    return np.exp(-idx / 10) * 0.6
    def dt_0_factor(idx):
        return -0.3 * np.exp(-idx * 0.05) + 0.9
    def dt_1_factor(idx):
        return 0.1 * np.exp(-idx * 0.05) + 1.05
    
    for idx, line in enumerate([0] + min_lines):
        dt.append((dt_0, dt_1))
            
        idx_search = np.logical_and(bins_fit[peaks] > line + dt_0, bins_fit[peaks] < line + dt_1)
        if len(resp[peaks][idx_search]) != 0:
            val_spike[idx] = np.max(resp[peaks][idx_search])
            max_idx = np.argmax(resp[peaks][idx_search])
            t_spike[idx] = bins_fit[peaks][idx_search][max_idx]

            # Compensando el delay de los primeros peaks max_lines[idx] por line
            # Lento rising time(?)
            delay[idx] = t_spike[idx] - line
            
            # dt_0 = delay[idx] * 0.6 + dt_0 * dt_0_factor(idx)
            # dt_1 = np.abs(delay[idx]) + dt_1 * dt_1_factor(idx)
            dt_0 = delay[idx] * dt_0_factor(idx)
            dt_1 = delay[idx] * dt_1_factor(idx)
                
        else:
            if idx == 0:
                delay[idx] = dt_0
            else:
                delay[idx] = delay[idx - 1]

    return (t_spike, val_spike, delay, dt)

def freq_off_peak_sel(resp, bins_fit, min_lines, max_lines, peaks):
    #t,val,pos,STD,indON,fON,pos2
    dt_0 = 0.3
    dt_1 = 0.6

    # Para graficar
    dt = []

    t_spike = np.empty(len(max_lines))
    t_spike[:] = np.nan

    val_spike = np.empty(len(max_lines))
    val_spike[:] = np.nan

    delay = np.empty(len(max_lines))
    delay[:] = np.nan

    def dt_0_factor(idx):
        return -0.3 * np.exp(-idx * 0.05) + 0.9
    def dt_1_factor(idx):
        return 0.1 * np.exp(-idx * 0.05) + 1.0
        
    for idx, line in enumerate(max_lines):
        dt.append((dt_0, dt_1))
        
        idx_search = np.logical_and(bins_fit[peaks] > line + dt_0, bins_fit[peaks] < line + dt_1)
        if len(resp[peaks][idx_search]) != 0:
            val_spike[idx] = np.max(resp[peaks][idx_search])
            max_idx = np.argmax(resp[peaks][idx_search])
            t_spike[idx] = bins_fit[peaks][idx_search][max_idx]

            # Compensando el delay de los primeros peaks min_lines[idx] por line
            # Lento rising time(?)
            delay[idx] = t_spike[idx] - max_lines[idx]

            dt_0 = delay[idx] * dt_0_factor(idx)
            dt_1 = delay[idx] * dt_1_factor(idx)
                
        else:
            if idx == 0:
                delay[idx] = dt_0
            else:
                delay[idx] = delay[idx - 1]

    return (t_spike, val_spike, delay, dt)

def freq_gauss_fit(t_spike, v_spike, bins_fit):
    # Índice máximo para hacer fitting, previo a filtrar NaN
    top_filt = np.argmin(np.logical_not(np.isnan(t_spike))) #-1 
    filt = np.logical_not(np.isnan(t_spike))[:top_filt]

    t_filter = t_spike[:top_filt][filt]
    v_filter = v_spike[:top_filt][filt]

    fmax_end = np.nan
    f0_end = np.nan
    sigma_end = np.nan 
    x_new = np.nan
    y_new = np.nan
    error = np.nan

    def Gaussian(x, a, b, c):
        return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))

    # Normalizar est_resp para el fitting. pcov se puede usar para ver la varianza en los componentes
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            # Se definen límites para el fitting
            bounds = ((0, 0, -np.inf), (1, 8.0, np.inf))

            popt, pcov = curve_fit(Gaussian, t_filter, v_filter / np.max(v_filter), bounds=bounds)

            # Denormalizar fmax_end
            fmax_end = popt[0] * np.max(v_filter)
            f0_end = popt[1]
            sigma_end = popt[2]

            x_new = bins_fit[bins_fit < t_filter[-1]]
            y_new = Gaussian(x_new, fmax_end, f0_end, sigma_end)

            error = np.nanmean(np.abs(v_filter - Gaussian(t_filter, fmax_end, f0_end, sigma_end)) ** 2) ** 0.5
        except OptimizeWarning:
            pass
        except ValueError:
            pass
        except RuntimeError:
            pass

    """
    popt, pcov = curve_fit(Gaussian, t_filter, v_filter / np.max(v_filter))

    # Denormalizar fmax_end
    fmax_end = popt[0] * np.max(v_filter)
    f0_end = popt[1]
    sigma_end = popt[2]

    x_new = bins_fit[bins_fit < t_filter[-1]]
    y_new = Gaussian(x_new, fmax_end, f0_end, sigma_end)

    error = np.nanmean(np.abs(v_filter - Gaussian(t_filter, fmax_end, f0_end, sigma_end)) ** 2) ** 0.5
    """
    return (fmax_end, f0_end, sigma_end, x_new, y_new, error)

def freq_analysis(spikes, bounds, parameters, psth_bin, fit_resolution, cell_type, min_lines, max_lines):
    time_start, time_end = bounds[:, 0], bounds[:, 1]
    freq_time_dur = np.mean(np.diff(bounds, axis=1))

    bins_fit = np.linspace(0, freq_time_dur, int(np.ceil(freq_time_dur / fit_resolution)), endpoint=False)
    bins = np.linspace(0, freq_time_dur, int(np.ceil(freq_time_dur / psth_bin)), endpoint=False)

    trials_chirp = spkt.get_trials(spikes, time_start, time_end)
    spikes_chirp = spkt.flatten_trials(trials_chirp)

    (psth, _) = np.histogram(spikes_chirp, bins=bins)
    resp = spkt.est_pdf(trials_chirp, bins_fit, bandwidth=psth_bin, norm_factor=psth.max())
    
    peaks = get_freq_peaks(resp)

    on_fmax = np.nan
    on_f0 = np.nan
    on_sigma = np.nan
    on_error = np.nan
    on_fcut = np.nan
    on_delay = np.nan
    off_fmax = np.nan
    off_f0 = np.nan
    off_sigma = np.nan
    off_error = np.nan
    off_fcut = np.nan
    off_delay = np.nan

    on_tim_spike, on_val_spike, on_delays = ([], [], [])
    on_time_fit, on_resp_fit = ([], [])
    on_fresp = []

    off_tim_spike, off_val_spike, off_delays = ([], [], [])
    off_time_fit, off_resp_fit = ([], [])
    off_fresp = []

    if cell_type == 1 or cell_type == 3: # ON u ON/OFF
        on_tim_spike, on_val_spike, on_delays, _ = freq_on_peak_sel(resp, bins_fit, min_lines, max_lines, peaks)
        on_fmax, on_f0, on_sigma, on_time_fit, on_resp_fit, on_error = freq_gauss_fit(on_tim_spike, on_val_spike, bins_fit)

        on_fresp = 1 / np.diff(on_tim_spike)

        filt = np.logical_not(np.isnan(on_tim_spike))
        id_max = np.argmin(filt) - 1
        if id_max < 0:
            id_max = 0
        
        on_fcut = on_tim_spike[id_max] # o agregar -1 acá

        id_min = id_max - 5
        if id_min < 0:
            id_min = 0
        
        if len(on_delays[id_min:id_max]) > 0 or np.count_nonzero(~np.isnan(on_delays[id_min:id_max])) > 0:
            on_delay = np.nanmean(on_delays[id_min:id_max])
    
    if cell_type == 2 or cell_type == 3: # OFF u ON/OFF
        off_tim_spike, off_val_spike, off_delays, _ = freq_off_peak_sel(resp, bins_fit, min_lines, max_lines, peaks)
        off_fmax, off_f0, off_sigma, off_time_fit, off_resp_fit, off_error = freq_gauss_fit(off_tim_spike, off_val_spike, bins_fit)

        off_fresp = 1 / np.diff(off_tim_spike)

        filt = np.logical_not(np.isnan(off_tim_spike))
        id_max = np.argmin(filt) - 1
        if id_max < 0:
            id_max = 0    

        off_fcut = off_tim_spike[id_max] # o agregar -1 acá

        id_min = id_max - 5
        if id_min < 0:
            id_min = 0
        
        if len(off_delays[id_min:id_max]) > 0 or np.count_nonzero(~np.isnan(off_delays[id_min:id_max])) > 0:
            off_delay = np.nanmean(off_delays[id_min:id_max])

    #SNR = 20 * np.log10(np.max(resp) / (2 * np.mean(resp)))
    
    char = np.asarray([on_fmax,
                       on_f0,
                       on_sigma,
                       on_error,
                       on_fcut,
                       on_delay,
                       off_fmax,
                       off_f0,
                       off_sigma,
                       off_error,
                       off_fcut,
                       off_delay])
                            #SNR])
    
    # Retorna respuesta de modulación en frecuencia, el análisis de peaks, fitting y features
    processed = {}
    processed['response'] = (bins_fit, resp)
    processed['peaks'] = peaks
    processed['on_peaks'] = (on_tim_spike, on_val_spike)
    processed['on_delays'] = on_delays
    processed['on_fresp'] = on_fresp
    processed['on_fitting'] = (on_time_fit, on_resp_fit)
    processed['off_peaks'] = (off_tim_spike, off_val_spike)
    processed['off_delays'] = off_delays
    processed['off_fresp'] = off_fresp
    processed['off_fitting'] = (off_time_fit, off_resp_fit)
    processed['char'] = char
    return processed

def amp_on_peak_sel(resp, bins_fit, min_lines, max_lines, peaks):
    dt_0 = 0.3
    dt_1 = 0.7

    # Para graficar
    dt = []

    t_spike = np.empty(len(min_lines))
    t_spike[:] = np.nan

    val_spike = np.empty(len(min_lines))
    val_spike[:] = np.nan

    delay = np.empty(len(min_lines))
    delay[:] = np.nan
    
    for idx, line in enumerate(min_lines):
        dt.append((dt_0, dt_1))
            
        idx_search = np.logical_and(bins_fit[peaks] > line + dt_0, bins_fit[peaks] < line + dt_1)
        if len(resp[peaks][idx_search]) != 0:
            val_spike[idx] = np.max(resp[peaks][idx_search])
            max_idx = np.argmax(resp[peaks][idx_search])
            t_spike[idx] = bins_fit[peaks][idx_search][max_idx]

            # Compensando el delay de los primeros peaks max_lines[idx] por line
            # Lento rising time(?)
            delay[idx] = t_spike[idx] - line
                
        else:
            if idx == 0:
                delay[idx] = dt_0
            else:
                delay[idx] = delay[idx - 1]

    return (t_spike, val_spike, delay, dt)

def amp_off_peak_sel(resp, bins_fit, min_lines, max_lines, peaks):
    #t,val,pos,STD,indON,fON,pos2
    dt_0 = 0.3
    dt_1 = 0.85

    # Para graficar
    dt = []

    t_spike = np.empty(len(max_lines))
    t_spike[:] = np.nan

    val_spike = np.empty(len(max_lines))
    val_spike[:] = np.nan

    delay = np.empty(len(max_lines))
    delay[:] = np.nan
        
    for idx, line in enumerate(max_lines):
        dt.append((dt_0, dt_1))
        
        idx_search = np.logical_and(bins_fit[peaks] > line + dt_0, bins_fit[peaks] < line + dt_1)
        if len(resp[peaks][idx_search]) != 0:
            val_spike[idx] = np.max(resp[peaks][idx_search])
            max_idx = np.argmax(resp[peaks][idx_search])
            t_spike[idx] = bins_fit[peaks][idx_search][max_idx]

            # Compensando el delay de los primeros peaks min_lines[idx] por line
            # Lento rising time(?)
            delay[idx] = t_spike[idx] - max_lines[idx]
                
        else:
            if idx == 0:
                delay[idx] = dt_0
            else:
                delay[idx] = delay[idx - 1]

    return (t_spike, val_spike, delay, dt)

def amp_sigmoid_fit(t_spike, v_spike, bins_fit):
    # Índice máximo para hacer fitting, previo a filtrar NaN
    filt = np.logical_not(np.isnan(t_spike))

    t_filter = t_spike[filt]
    v_filter = v_spike[filt]

    fmax_end = np.nan
    slope_end = np.nan
    shift_end = np.nan 
    x_new = np.nan
    y_new = np.nan
    error = np.nan

    def Sigmoid(x, a, b, c):
        return a / (1.0 + np.exp(-b * x + c))

    # Normalizar est_resp para el fitting. pcov se puede usar para ver la varianza en los componentes
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            # Se definen límites para el fitting
            bounds = ((0, -np.inf, 0), (1, np.inf, 10.0))

            popt, pcov = curve_fit(Sigmoid, t_filter, v_filter / np.max(v_filter), bounds=bounds)

            # Denormalizar fmax_end
            fmax_end = popt[0] * np.max(v_filter)
            slope_end = popt[1]
            shift_end = popt[2]

            x_new = bins_fit[bins_fit < t_filter[-1]]
            y_new = Sigmoid(x_new, fmax_end, slope_end, shift_end)

            error = np.nanmean(np.abs(v_filter - Sigmoid(t_filter, fmax_end, slope_end, shift_end)) ** 2) ** 0.5
        except OptimizeWarning:
            pass
        except ValueError:
            pass
        except RuntimeError:
            pass

    """
    popt, pcov = curve_fit(Gaussian, t_filter, v_filter / np.max(v_filter))

    # Denormalizar fmax_end
    fmax_end = popt[0] * np.max(v_filter)
    f0_end = popt[1]
    sigma_end = popt[2]

    x_new = bins_fit[bins_fit < t_filter[-1]]
    y_new = Gaussian(x_new, fmax_end, f0_end, sigma_end)

    error = np.nanmean(np.abs(v_filter - Gaussian(t_filter, fmax_end, f0_end, sigma_end)) ** 2) ** 0.5
    """
    return (fmax_end, slope_end, shift_end, x_new, y_new, error)

def amp_analysis(spikes, bounds, parameters, psth_bin, fit_resolution, cell_type, min_lines, max_lines):
    time_start, time_end = bounds[:, 0], bounds[:, 1]
    amp_time_dur = np.mean(np.diff(bounds, axis=1))

    bins_fit = np.linspace(0, amp_time_dur, int(np.ceil(amp_time_dur / fit_resolution)), endpoint=False)
    bins = np.linspace(0, amp_time_dur, int(np.ceil(amp_time_dur / psth_bin)), endpoint=False)

    trials_chirp = spkt.get_trials(spikes, time_start, time_end)
    spikes_chirp = spkt.flatten_trials(trials_chirp)

    (psth, _) = np.histogram(spikes_chirp, bins=bins)
    resp = spkt.est_pdf(trials_chirp, bins_fit, bandwidth=psth_bin, norm_factor=psth.max())

    peaks, _ = find_peaks(resp, height=np.mean(resp)*0.2, prominence=4)

    on_fmax = np.nan
    on_slope = np.nan
    on_shift = np.nan
    on_error = np.nan
    on_fcut = np.nan
    on_delay = np.nan
    off_fmax = np.nan
    off_slope = np.nan
    off_shift = np.nan
    off_error = np.nan
    off_fcut = np.nan
    off_delay = np.nan

    on_tim_spike, on_val_spike, on_delays = ([], [], [])
    on_time_fit, on_resp_fit = ([], [])
    on_fresp = []

    off_tim_spike, off_val_spike, off_delays = ([], [], [])
    off_time_fit, off_resp_fit = ([], [])
    off_fresp = []

    # Amplitude modulation vs time. Se asume: max_val * (ramp[time] - ramp[time - dur])
    def amp_mod(time, max_val, dur):
        if time < dur:
            return time * max_val / dur
        else:
            return max_val

    if cell_type == 1 or cell_type == 3: # ON u ON/OFF
        on_tim_spike, on_val_spike, on_delays, _ = amp_on_peak_sel(resp, bins_fit, min_lines, max_lines, peaks)
        on_fmax, on_slope, on_shift, on_time_fit, on_resp_fit, on_error = amp_sigmoid_fit(on_tim_spike, on_val_spike, bins_fit)

        on_fresp = 1 / np.diff(on_tim_spike)

        if len(on_fresp) > 0 and np.count_nonzero(~np.isnan(on_fresp)) > 0:
            on_fcut = np.nanmean(on_fresp)
        if len(on_delays) > 0 and np.count_nonzero(~np.isnan(on_delays)) > 0:
            on_delay = np.nanmean(on_delays)
    
    if cell_type == 2 or cell_type == 3: # OFF u ON/OFF
        off_tim_spike, off_val_spike, off_delays, _ = amp_off_peak_sel(resp, bins_fit, min_lines, max_lines, peaks)
        off_fmax, off_slope, off_shift, off_time_fit, off_resp_fit, off_error = amp_sigmoid_fit(off_tim_spike, off_val_spike, bins_fit)
        
        off_fresp = 1 / np.diff(off_tim_spike)
        
        if len(off_fresp) > 0 and np.count_nonzero(~np.isnan(off_fresp)) > 0:
            off_fcut = np.nanmean(off_fresp)
        if len(off_delays) > 0 and np.count_nonzero(~np.isnan(off_delays)) > 0:
            off_delay = np.nanmean(off_delays)

    #SNR = 20 * np.log10(np.max(resp) / (2 * np.mean(resp)))
    
    char = np.asarray([on_fmax,
                       on_slope,
                       on_shift,
                       on_error,
                       on_fcut,
                       on_delay,
                       off_fmax,
                       off_slope,
                       off_shift,
                       off_error,
                       off_fcut,
                       off_delay])
                            #SNR])
    
    # Retorna respuesta de modulación en frecuencia, el análisis de peaks, fitting y features
    processed = {}
    processed['response'] = (bins_fit, resp)
    processed['peaks'] = peaks
    processed['on_peaks'] = (on_tim_spike, on_val_spike)
    processed['on_delays'] = on_delays
    processed['on_fresp'] = on_fresp
    processed['on_fitting'] = (on_time_fit, on_resp_fit)
    processed['off_peaks'] = (off_tim_spike, off_val_spike)
    processed['off_delays'] = off_delays
    processed['off_fresp'] = off_fresp
    processed['off_fitting'] = (off_time_fit, off_resp_fit)
    processed['char'] = char
    return processed

# PROTO-VERSION
#def write_csv_features(freq_features):

def get_chirp_features(spks, events, parameters, panalysis, 
                       psth_bin, fit_resolution):
    """
    Compute chirp response and psth from spiketimes

    """

    sample_rate = 1 / 60 # should be a parameter
    chirp_signal = chirp_generator(sample_rate, parameters, splitted=True)

    columns = ['on_freq_fmax',
               'on_freq_f0',
               'on_freq_sigma',
               'on_freq_error',
               'on_freq_fcut',
               'on_freq_delay',
               'off_freq_fmax',
               'off_freq_f0',
               'off_freq_sigma',
               'off_freq_error',
               'off_freq_fcut',
               'off_freq_delay',
               'on_amp_fmax',
               'on_amp_slope',
               'on_amp_shift',
               'on_amp_error',
               'on_amp_fcut',
               'on_amp_delay',
               'off_amp_fmax',
               'off_amp_slope',
               'off_amp_shift',
               'off_amp_error',
               'off_amp_fcut',
               'off_amp_delay',
               'SNR',
               'QI']

    index = []
    cells_feat = np.empty((len(spks.keys()), len(columns)))
    cells_feat[:] = np.nan

    for idx, unit in enumerate(tqdm(spks.keys())):
        spikes = spks[unit][:].flatten() / 20000.0

        fields_df = ['start_event', 'end_event']
        mask_on = events['extra_description'] == 'ON'
        mask_off = events['extra_description'] == 'OFF'
        mask_freq = events['extra_description'] == 'FREQ'
        mask_adap = events['extra_description'] == 'adap_2' # Using as final chirp times and for amplitude mod
        mask_amp = events['extra_description'] == 'AMP'

        times_on = np.array(events[mask_on][fields_df]) / 20000.0
        times_off = np.array(events[mask_off][fields_df]) / 20000.0
        times_freq = np.array(events[mask_freq][fields_df]) / 20000.0
        times_adap = np.array(events[mask_adap][fields_df]) / 20000.0
        times_amp = np.array(events[mask_amp][fields_df]) / 20000.0

        # Change end_event of amp mod by finish times of adap
        times_amp[:, 1] = times_adap[:, 1]

        # Get SNR from complete experiment
        chirp_bounds = np.array([times_on[:, 0], times_adap[:, 1]]).transpose()
        chirp_dur = np.mean(np.diff(chirp_bounds, axis=1))

        trials_chirp = spkt.get_trials(spikes, times_on[:, 0], times_adap[:, 1])
        spikes_chirp = spkt.flatten_trials(trials_chirp)

        exp_time = np.linspace(0, chirp_dur, int(chirp_dur / fit_resolution))
        exp_bins = np.linspace(0, chirp_dur, int(chirp_dur / psth_bin))        

        (psth, _) = np.histogram(spikes_chirp, bins=exp_bins)
        exp_resp = spkt.est_pdf(trials_chirp, exp_time, bandwidth=psth_bin, norm_factor=psth.max())

        if np.max(exp_resp) != 0:
            SNR = 20 * np.log10(np.max(exp_resp) / (2 * np.mean(exp_resp)))
        else:
            SNR = np.nan
        
        # Quality index
        repetitions = []

        for trial in trials_chirp:
            (rep_psth, _) = np.histogram(trial, bins=exp_bins)
            est_resp_rep = spkt.est_pdf([trial], exp_time, bandwidth=psth_bin, norm_factor=rep_psth.max())
            repetitions.append(est_resp_rep)

        arr = np.array(repetitions)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                QI = np.var(np.mean(arr, axis=0)) / np.mean(np.var(arr, axis=1))
            except RuntimeWarning:
                QI = 0.0
                pass

        #QI = np.var(np.mean(arr, axis=0)) / np.mean(np.var(arr, axis=1))
        repetitions = len(trials_chirp)

        # ON-OFF-ON/OFF classifier
        flash_bounds = np.array([times_on[:, 0], times_off[:, 1]]).transpose()
        flash_dur = np.mean(np.diff(flash_bounds, axis=1))

        flash_time_start = flash_bounds[:, 0]
        flash_time_end = flash_bounds[:, 1]
        flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))
        flash_bins = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / psth_bin)))

        on_dur, off_dur = np.mean(np.diff(times_on, axis=1)), np.mean(np.diff(times_off, axis=1))

        trials_chirp = spkt.get_trials(spikes, flash_time_start, flash_time_end)
        spikes_chirp = spkt.flatten_trials(trials_chirp)

        (psth, _) = np.histogram(spikes_chirp, bins=flash_bins)
        flash_resp = spkt.est_pdf(trials_chirp, flash_time, bandwidth=psth_bin, norm_factor=psth.max())

        flash_bounds = (0, on_dur, on_dur, on_dur + off_dur)
        # Repetitions added due to spiketools conditions
        cell_type, _ = spkt.get_features_flash(flash_resp / repetitions, flash_time, flash_bounds, resp_thr=0.2, bias_thr=0.65)

        # Frequency analysis
        max_lines, min_lines = chirp_signal['freq_mod_max_lines'], chirp_signal['freq_mod_min_lines']
        freq_data = freq_analysis(spikes, times_freq, parameters, psth_bin, fit_resolution, cell_type, min_lines, max_lines)

        # Amplitude analysis
        max_lines, min_lines = chirp_signal['amp_mod_max_lines'], chirp_signal['amp_mod_min_lines']
        amp_data = amp_analysis(spikes, times_amp, parameters, psth_bin, fit_resolution, cell_type, min_lines, max_lines)
        
        index.append(unit)
        cells_feat[idx, :] = np.concatenate((freq_data['char'], amp_data['char'], [SNR, QI]))
       
    df = pd.DataFrame(cells_feat, columns=columns, index=index)
    return df


def get_chirp_response(spks, events, parameters, panalysis, 
                       psth_bin, fit_resolution):
    """
    Compute chirp response and psth from spiketimes

    """
    warnings.filterwarnings(action='once')

    sample_rate = 1 / 60 # should be a parameter
    chirp_signal = chirp_generator(sample_rate, parameters, splitted=True)

    output_units = {}
    freq_columns = ['on_freq_fmax',
                    'on_freq_f0',
                    'on_freq_sigma',
                    'on_freq_error',
                    'on_freq_fcut',
                    'on_freq_delay',
                    'off_freq_fmax',
                    'off_freq_f0',
                    'off_freq_sigma',
                    'off_freq_error',
                    'off_freq_fcut',
                    'off_freq_delay']
    amp_columns = ['on_amp_fmax',
                   'on_amp_slope',
                   'on_amp_shift',
                   'on_amp_error',
                   'on_amp_fcut',
                   'on_amp_delay',
                   'off_amp_fmax',
                   'off_amp_slope',
                   'off_amp_shift',
                   'off_amp_error',
                   'off_amp_fcut',
                   'off_amp_delay']

    index = []
    #cells_feat = np.empty((len(spks), len(columns)))
    #cells_feat[:] = np.nan

    for idx, unit in enumerate(spks.keys()):

        output_units[unit] = {}
        spikes = spks[unit][:].flatten() / 20000.0
        repetitions = 21

        fields_df = ['start_event', 'end_event']
        mask_on = events['extra_description'] == 'ON'
        mask_off = events['extra_description'] == 'OFF'
        mask_freq = events['extra_description'] == 'FREQ'
        mask_adap = events['extra_description'] == 'adap_2' # Using as final chirp times and for amplitude mod
        mask_amp = events['extra_description'] == 'AMP'

        times_on = np.array(events[mask_on][fields_df]) / 20000.0
        times_off = np.array(events[mask_off][fields_df]) / 20000.0
        times_freq = np.array(events[mask_freq][fields_df]) / 20000.0
        times_adap = np.array(events[mask_adap][fields_df]) / 20000.0
        times_amp = np.array(events[mask_amp][fields_df]) / 20000.0

        # Change end_event of amp mod by finish times of adap
        times_amp[:, 1] = times_adap[:, 1]

        # ON-OFF-ON/OFF classifier
        flash_bounds = np.array([times_on[:, 0], times_off[:, 1]]).transpose()
        flash_dur = np.mean(np.diff(flash_bounds, axis=1))

        flash_time_start = flash_bounds[:, 0]
        flash_time_end = flash_bounds[:, 1]
        flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))
        flash_bins = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / psth_bin)))

        on_dur, off_dur = np.mean(np.diff(times_on, axis=1)), np.mean(np.diff(times_off, axis=1))

        trials_chirp = spkt.get_trials(spikes, flash_time_start, flash_time_end)
        spikes_chirp = spkt.flatten_trials(trials_chirp)

        (psth, _) = np.histogram(spikes_chirp, bins=flash_bins)
        flash_resp = spkt.est_pdf(trials_chirp, flash_time, bandwidth=psth_bin, norm_factor=psth.max())

        flash_bounds = (0, on_dur, on_dur, on_dur + off_dur)
        cell_type, flash_char = spkt.get_features_flash(flash_resp / repetitions, flash_time, flash_bounds, resp_thr=0.2, bias_thr=0.65)

        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                cell_type, flash_char = spkt.get_features_flash(flash_resp / repetitions, flash_time, flash_bounds, resp_thr=0.2, bias_thr=0.65)
            except RuntimeWarning:
                pass
        """

        # Frequency analysis
        max_lines, min_lines = chirp_signal['freq_mod_max_lines'], chirp_signal['freq_mod_min_lines']
        freq_data = freq_analysis(spikes, times_freq, parameters, psth_bin, fit_resolution, cell_type, min_lines, max_lines)

        # Amplitude analysis
        max_lines, min_lines = chirp_signal['amp_mod_max_lines'], chirp_signal['amp_mod_min_lines']
        amp_data = amp_analysis(spikes, times_amp, parameters, psth_bin, fit_resolution, cell_type, min_lines, max_lines)

        #                  'cell_type'
        output_units[unit]['flash_type'] = cell_type
        output_units[unit]['flash_char'] = flash_char
        output_units[unit]['flash_time'] = flash_time
        output_units[unit]['flash_resp'] = flash_resp
        
        output_units[unit]['freq_time'] = freq_data['response'][0]
        output_units[unit]['freq_response'] = freq_data['response'][1]
        output_units[unit]['freq_peaks'] = freq_data['peaks']
        output_units[unit]['freq_on_peaks'] = freq_data['on_peaks']
        output_units[unit]['freq_on_delays'] = freq_data['on_delays']
        output_units[unit]['freq_on_fresp'] = freq_data['on_fresp']
        output_units[unit]['freq_on_fitting'] = freq_data['on_fitting']
        output_units[unit]['freq_off_peaks'] = freq_data['off_peaks']
        output_units[unit]['freq_off_delays'] = freq_data['off_delays']
        output_units[unit]['freq_off_fresp'] = freq_data['off_fresp']
        output_units[unit]['freq_off_fitting'] = freq_data['off_fitting']
        output_units[unit]['freq_char'] = dict(zip(freq_columns, freq_data['char']))

        output_units[unit]['amp_time'] = amp_data['response'][0]
        output_units[unit]['amp_response'] = amp_data['response'][1]
        output_units[unit]['amp_peaks'] = amp_data['peaks']
        output_units[unit]['amp_on_peaks'] = amp_data['on_peaks']
        output_units[unit]['amp_on_delays'] = amp_data['on_delays']
        output_units[unit]['amp_on_fresp'] = amp_data['on_fresp']
        output_units[unit]['amp_on_fitting'] = amp_data['on_fitting']
        output_units[unit]['amp_off_peaks'] = amp_data['off_peaks']
        output_units[unit]['amp_off_delays'] = amp_data['off_delays']
        output_units[unit]['amp_off_fresp'] = amp_data['off_fresp']
        output_units[unit]['amp_off_fitting'] = amp_data['off_fitting']
        output_units[unit]['amp_char'] = dict(zip(amp_columns, amp_data['char']))

    return (output_units, chirp_signal)
