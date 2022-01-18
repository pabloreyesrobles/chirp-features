from scipy.signal import hann, find_peaks, hilbert
from scipy.optimize import curve_fit, OptimizeWarning
from spikelib import spiketools as spkt
from functools import reduce
from tqdm import tqdm

import numpy as np
import pandas as pd
import warnings
import h5py
import os
# import sys
import istarmap
from multiprocessing import Pool

class PeakDecoding:
  
  def __init__(self):
    # Time, intensity and delay respect to ON/OFF stimulus
    self.tim_spike = np.array([])
    self.val_spike = np.array([])
    self.delays = np.array([])

    # Time and values arrays of response fitting
    self.time_fit = np.array([])
    self.resp_fit = np.array([])

    # Time and values arrays of delay fitting
    self.dtime_fit = np.array([])
    self.dresp_fit = np.array([])

    # Frequency response of data
    self.fresp = np.array([])

def chirp_def_args():
  # Chirp ECOVIS default parameters
  chirp_args = {}
  chirp_args['chirp_duration'] = 35
  chirp_args['base_amp'] = 0.5

  chirp_args['t_adap'] = 2

  chirp_args['amp_on'] = 1
  chirp_args['t_on'] = 3
  chirp_args['t_off'] = 3

  chirp_args['freq_mod_final_freq'] = 15
  chirp_args['freq_mod_time'] = 15
  chirp_args['freq_mod_amp'] = 0.5
  chirp_args['freq_mod_init_phase'] = 2 * np.pi

  chirp_args['amp_mod_freq'] = 1
  chirp_args['amp_mod_time'] = 8
  chirp_args['amp_mod_max'] = 0.5

  return chirp_args

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
  freq_mod_min_lines = [0] + [np.sqrt(-0.5 + 2 * i) for i in range(1, line_limit)]
  freq_mod_zero_lines = [np.sqrt(2 * i) for i in range(line_limit)]

  line_limit = int(np.round(args['amp_mod_time'] * args['amp_mod_freq'] - 0.25))
  amp_mod_max_lines = [((0.25 + i) / args['amp_mod_freq']) for i in range(line_limit)]
  amp_mod_min_lines = [0] + [((-0.25 + i + 1) / args['amp_mod_freq']) for i in range(line_limit)] # i + 1 need 

  chirp_signal = {}

  on_off_units = np.concatenate((on_units, off_units))
  chirp_signal['on_off'] = np.array([np.linspace(0, args['t_on'] + args['t_off'] - sample_rate, len(on_off_units)), on_off_units])
  chirp_signal['adap_0'] = np.array([np.linspace(0, args['t_adap'] - sample_rate, len(adap_units)), adap_units])
  chirp_signal['freq'] = np.array([np.linspace(0, args['freq_mod_time'] - sample_rate, len(freq_units)), freq_units])
  chirp_signal['adap_1'] = np.array([np.linspace(0, args['t_adap'] - sample_rate, len(adap_units)), adap_units])
  chirp_signal['amp'] = np.array([np.linspace(0, args['amp_mod_time'] - sample_rate, len(amp_units)), amp_units])
  chirp_signal['adap_2'] = np.array([np.linspace(0, args['t_adap'] - sample_rate, len(adap_units)), adap_units])

  chirp_signal['freq_mod_max_lines'] = np.array(freq_mod_max_lines)
  chirp_signal['freq_mod_min_lines'] = np.array(freq_mod_min_lines)
  chirp_signal['freq_mod_zero_lines'] = np.array(freq_mod_zero_lines)

  chirp_signal['amp_mod_max_lines'] = np.array(amp_mod_max_lines)
  chirp_signal['amp_mod_min_lines'] = np.array(amp_mod_min_lines)

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

  # extra_description filter

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
        'protocol_spec': kchirp.extra_description,
        'rep_name': kchirp.repetition_name,
        'repetead_frames': '',
        '#repetead_frames': rep_trans,
      })
    events_chirp.append(df)
  events = reduce(lambda x, y: pd.concat([x, y]), events_chirp)
  events.to_csv(output_file, sep=';', index=False)
  return events

def get_freq_peaks(resp):
  fmin = np.mean(resp) * 0.1
  freq_peaks, _ = find_peaks(resp, height=fmin, prominence=fmin*0.5) # heuristica

  return freq_peaks

def freq_on_peak_sel(resp, bins_fit, min_lines, max_lines, peaks):
  dt_0 = (max_lines[0] - min_lines[0]) * 0.08
  dt_1 = (max_lines[0] - min_lines[0]) * 1.08

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
    return -0.35 * np.exp(-idx * 0.45) + 0.7
  def dt_1_factor(idx):
    return 0.05 * np.exp(-idx * 0.45) + 1.15
  def Exp(x, a, b, c):
    return a * np.exp(-x * b) + c
  
  a1, b1, c1 = 0.3, 0.6, 0.165
  a2, b2, c2 = 0.6, 0.6, 0.195
  a3, b3, c3 = 0.15, 0.6, 0.125
  
  for idx, line in enumerate(min_lines):
    dt_0 = Exp(line, a3, b3, c3)
    dt_1 = Exp(line, a2, b2, c2)		
    if idx != 0:
        if delay[idx - 1] < Exp(line, a1, b1, c1):
          dt_0 -= delay[idx - 1] * 0.05
          dt_1 -= delay[idx - 1] * 0.1
        else:
          dt_0 += delay[idx - 1] * 0.08
          dt_1 += delay[idx - 1] * 0.2

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
  dt_0 = (min_lines[1] - max_lines[0]) * 0.08
  dt_1 = (min_lines[1] - max_lines[0]) * 1.08

  # Para graficar
  dt = []

  # Peak conflict identifier

  t_spike = np.empty(len(max_lines))
  t_spike[:] = np.nan

  val_spike = np.empty(len(max_lines))
  val_spike[:] = np.nan

  delay = np.empty(len(max_lines))
  delay[:] = np.nan

  def dt_0_factor(idx):
    return -0.35 * np.exp(-idx * 0.35) + 0.7
  def dt_1_factor(idx):
    return 0.05 * np.exp(-idx * 0.45) + 1.15
  def Exp(x, a, b, c):
    return a * np.exp(-x * b) + c
  
  a1, b1, c1 = 0.3, 0.6, 0.165
  a4, b4, c4 = 0.6, 0.6, 0.195
  a5, b5, c5 = 0.05, 0.6, 0.13
    
  for idx, line in enumerate(max_lines):
    dt_0 = Exp(line, a5, b5, c5)
    dt_1 = Exp(line, a4, b4, c4)
    if idx != 0:
      if delay[idx - 1] < Exp(line, a1, b1, c1):
        dt_0 -= delay[idx - 1] * 0.1
        dt_1 -= delay[idx - 1] * 0.1
      else:
        dt_0 += delay[idx - 1] * 0.1
        dt_1 += delay[idx - 1] * 0.3

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
  #top_filt = np.argmin(np.logical_not(np.isnan(t_spike))) #-1 
  #filt = np.logical_not(np.isnan(t_spike))[:top_filt]
  on_id_max = np.argmax(np.isnan(t_spike[1:])) + 1

  t_filter = t_spike[:on_id_max]
  v_filter = v_spike[:on_id_max]

  nan_filt = np.logical_not(np.isnan(t_filter))
  t_filter = t_filter[nan_filt]
  v_filter = v_filter[nan_filt]

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

  return (fmax_end, f0_end, sigma_end, x_new, y_new, error)

def freq_exp_fit(t_spike, v_spike, bins_fit):
  # Índice máximo para hacer fitting, previo a filtrar NaN
  filt = np.argmax(np.isnan(t_spike[1:])) + 1

  t_filter = t_spike[:filt]
  v_filter = v_spike[:filt]

  nan_filt = np.logical_not(np.isnan(t_filter))
  t_filter = t_filter[nan_filt]
  v_filter = v_filter[nan_filt]

  delay_start = np.nan
  scale = np.nan
  delay_end = np.nan 
  x_new = np.nan
  y_new = np.nan
  error = np.nan

  def Exp(x, a, b, c):
    return a * np.exp(-x * b) + c

  # Normalizar est_resp para el fitting. pcov se puede usar para ver la varianza en los componentes
  with warnings.catch_warnings():
    warnings.simplefilter("error", OptimizeWarning)
    try:
      bounds = ((-np.inf, 0, 0), (0, np.inf, np.inf))
      popt, pcov = curve_fit(Exp, t_filter, v_filter)

      delay_start = popt[0]
      scale = popt[1]
      delay_end = popt[2]

      x_new = bins_fit[bins_fit < t_filter[-1]]
      y_new = Exp(x_new, delay_start, scale, delay_end)

      error = np.nanmean(np.abs(v_filter - y_new) ** 2) ** 0.5

    except OptimizeWarning:
      pass
    except (ValueError, RuntimeError):
      pass
    except:
      pass
  
  return (delay_start, scale, delay_end, x_new, y_new, error)

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

  # TODO: will give order to script
  #on_decode = PeakDecoding()
  #off_decode = PeakDecoding()

  on_tim_spike, on_val_spike, on_delays = ([], [], [])
  on_time_fit, on_resp_fit = ([], [])
  on_dtime_fit, on_dresp_fit = ([], [])
  on_fresp = []

  off_tim_spike, off_val_spike, off_delays = ([], [], [])
  off_time_fit, off_resp_fit = ([], [])
  off_dtime_fit, off_dresp_fit = ([], [])
  off_fresp = []

  if cell_type == 1 or cell_type == 3: # ON u ON/OFF
    on_tim_spike, on_val_spike, on_delays, _ = freq_on_peak_sel(resp, bins_fit, min_lines, max_lines, peaks)
    on_fmax, on_f0, on_sigma, on_time_fit, on_resp_fit, on_error = freq_gauss_fit(on_tim_spike, on_val_spike, bins_fit)
    on_dstart, on_dscale, on_dend, on_dtime_fit, on_dresp_fit, on_derror = freq_exp_fit(on_tim_spike, on_delays, bins_fit)

    on_diff = np.diff(on_tim_spike)
    on_fresp = np.divide(np.ones_like(on_diff), on_diff, out=np.full_like(on_diff, np.nan, dtype=np.double), where=on_diff!=0)
    
    on_id_max = np.argmax(np.isnan(on_tim_spike[1:]))
    on_fcut = on_tim_spike[on_id_max] # o agregar -1 acá

    on_id_min = on_id_max - 5
    if on_id_min < 0:
      on_id_min = 0
    
    if len(on_delays[on_id_min:on_id_max]) > 0 or np.count_nonzero(~np.isnan(on_delays[on_id_min:on_id_max])) > 0:
      on_delay = np.nanmean(on_delays[on_id_min:on_id_max])
  
  if cell_type == 2 or cell_type == 3: # OFF u ON/OFF
    off_tim_spike, off_val_spike, off_delays, _ = freq_off_peak_sel(resp, bins_fit, min_lines, max_lines, peaks)
    off_fmax, off_f0, off_sigma, off_time_fit, off_resp_fit, off_error = freq_gauss_fit(off_tim_spike, off_val_spike, bins_fit)
    off_dstart, off_dscale, off_dend, off_dtime_fit, off_dresp_fit, off_derror = freq_exp_fit(off_tim_spike, off_delays, bins_fit)

    off_diff = np.diff(off_tim_spike)
    off_fresp = np.divide(np.ones_like(off_diff), off_diff, out=np.full_like(off_diff, np.nan, dtype=np.double), where=off_diff!=0)

    off_id_max = np.argmax(np.isnan(off_tim_spike[1:]))
    off_fcut = off_tim_spike[off_id_max] # o agregar -1 acá

    off_id_min = off_id_max - 5
    if off_id_min < 0:
      off_id_min = 0
    
    if len(off_delays[off_id_min:off_id_max]) > 0 or np.count_nonzero(~np.isnan(off_delays[off_id_min:off_id_max])) > 0:
      off_delay = np.nanmean(off_delays[off_id_min:off_id_max])
  
  try:
    peak_conflict = any(i in on_tim_spike[:on_id_max] for i in off_tim_spike[:off_id_max])
  except:
    peak_conflict = 0
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
             off_delay,
             peak_conflict])
  
  # Retorna respuesta de modulación en frecuencia, el análisis de peaks, fitting y features
  processed = {}
  processed['response'] = np.array([bins_fit, resp])
  processed['peaks'] = np.array(peaks)
  processed['on_peaks'] = np.array([on_tim_spike, on_val_spike])
  processed['on_delays'] = np.array(on_delays)
  processed['on_fresp'] = np.array(on_fresp)
  processed['on_fitting'] = np.array([on_time_fit, on_resp_fit])
  processed['on_delay_fit'] = np.array([on_dtime_fit, on_dresp_fit])
  processed['off_peaks'] = np.array([off_tim_spike, off_val_spike])
  processed['off_delays'] = np.array(off_delays)
  processed['off_fresp'] = np.array(off_fresp)
  processed['off_fitting'] = np.array([off_time_fit, off_resp_fit])
  processed['off_delay_fit'] = np.array([off_dtime_fit, off_dresp_fit])
  processed['char'] = char
  return processed

def amp_on_peak_sel(resp, bins_fit, min_lines, max_lines, peaks):
  dt_0 = 0.2
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
  dt_0 = 0.2
  dt_1 = 0.7

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

  peaks, _ = find_peaks(resp, height=np.mean(resp)*0.2, prominence=np.mean(resp)*0.05)

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
    return time * max_val / dur

  if cell_type == 1 or cell_type == 3: # ON u ON/OFF
    on_tim_spike, on_val_spike, on_delays, _ = amp_on_peak_sel(resp, bins_fit, min_lines, max_lines, peaks)
    on_fmax, on_slope, on_shift, on_time_fit, on_resp_fit, on_error = amp_sigmoid_fit(on_tim_spike, on_val_spike, bins_fit)

    on_diff = np.diff(on_tim_spike)
    on_fresp = np.divide(np.ones_like(on_diff), on_diff, out=np.full_like(on_diff, np.nan, dtype=np.double), where=on_diff!=0)

    if len(on_fresp) > 0 and np.count_nonzero(~np.isnan(on_fresp)) > 0:
      on_fcut = np.nanmean(on_fresp)
    if len(on_delays) > 0 and np.count_nonzero(~np.isnan(on_delays)) > 0:
      on_delay = np.nanmean(on_delays)
  
  if cell_type == 2 or cell_type == 3: # OFF u ON/OFF
    off_tim_spike, off_val_spike, off_delays, _ = amp_off_peak_sel(resp, bins_fit, min_lines, max_lines, peaks)
    off_fmax, off_slope, off_shift, off_time_fit, off_resp_fit, off_error = amp_sigmoid_fit(off_tim_spike, off_val_spike, bins_fit)
    
    off_diff = np.diff(off_tim_spike)
    off_fresp = np.divide(np.ones_like(off_diff), off_diff, out=np.full_like(off_diff, np.nan, dtype=np.double), where=off_diff!=0)
    
    if len(off_fresp) > 0 and np.count_nonzero(~np.isnan(off_fresp)) > 0:
      off_fcut = np.nanmean(off_fresp)
    if len(off_delays) > 0 and np.count_nonzero(~np.isnan(off_delays)) > 0:
      off_delay = np.nanmean(off_delays)

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
  
  # Retorna respuesta de modulación en frecuencia, el análisis de peaks, fitting y features
  processed = {}
  processed['response'] = (bins_fit, resp)
  processed['peaks'] = np.array(peaks)
  processed['on_peaks'] = np.array([on_tim_spike, on_val_spike])
  processed['on_delays'] = np.array(on_delays)
  processed['on_fresp'] = np.array(on_fresp)
  processed['on_fitting'] = np.array([on_time_fit, on_resp_fit])
  processed['off_peaks'] = np.array([off_tim_spike, off_val_spike])
  processed['off_delays'] = np.array(off_delays)
  processed['off_fresp'] = np.array(off_fresp)
  processed['off_fitting'] = np.array([off_time_fit, off_resp_fit])
  processed['char'] = char
  return processed

def get_chirp_response(spks, events, parameters, cell_key, 
                        psth_bin, fit_resolution):                     
  sample_rate = 1 / 60 # should be a parameter
  chirp_signal = chirp_generator(sample_rate, parameters, splitted=True)
  spikes = spks[cell_key][:].flatten() / 20000.0

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
    SNR = 10 * np.log10(np.max(exp_resp) / (2 * np.mean(exp_resp)))
  else:
    SNR = np.nan
  
  # Quality index
  est_resp_list = []

  for trial in trials_chirp:
    (rep_psth, _) = np.histogram(trial, bins=exp_bins)
    est_resp = spkt.est_pdf([trial], exp_time, bandwidth=psth_bin, norm_factor=rep_psth.max())
    est_resp_list.append(est_resp)

  arr = np.array(est_resp_list)

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
  cell_type, flash_char = spkt.get_features_flash(flash_resp / repetitions, flash_time, flash_bounds, resp_thr=0.2, bias_thr=0.65, sust_time=1.0)
  flash_char[-2] = flash_char[-2] * repetitions
  flash_char[-1] = flash_char[-1] * repetitions

  # Frequency analysis
  min_lines = chirp_signal['freq_mod_min_lines']
  max_lines = chirp_signal['freq_mod_max_lines']
  freq_data = freq_analysis(spikes, times_freq, parameters, psth_bin, fit_resolution, cell_type, min_lines, max_lines)

  # Amplitude analysis
  min_lines = chirp_signal['amp_mod_min_lines']
  max_lines = chirp_signal['amp_mod_max_lines']
  amp_data = amp_analysis(spikes, times_amp, parameters, psth_bin, fit_resolution, cell_type, min_lines, max_lines)

  output = {}
  char = {}

  char['flash_type'] = cell_type
  #char['flash_char'] = flash_char
  char['SNR'] = SNR
  char['QI'] = QI
  char['flash_char'] = flash_char
  char['amp_char'] = amp_data['char']
  char['freq_char'] = freq_data['char']

  output['flash_time'] = flash_time
  output['flash_resp'] = flash_resp
  
  output['freq_time'] = freq_data['response'][0]
  output['freq_response'] = freq_data['response'][1]
  output['freq_peaks'] = freq_data['peaks']
  output['freq_on_peaks'] = freq_data['on_peaks']
  output['freq_on_delays'] = freq_data['on_delays']
  output['freq_on_fresp'] = freq_data['on_fresp']
  output['freq_on_fitting'] = freq_data['on_fitting']
  output['freq_on_delay_fit'] = freq_data['on_delay_fit']
  output['freq_off_peaks'] = freq_data['off_peaks']
  output['freq_off_delays'] = freq_data['off_delays']
  output['freq_off_fresp'] = freq_data['off_fresp']
  output['freq_off_fitting'] = freq_data['off_fitting']
  output['freq_off_delay_fit'] = freq_data['off_delay_fit']

  output['amp_time'] = amp_data['response'][0]
  output['amp_response'] = amp_data['response'][1]
  output['amp_peaks'] = amp_data['peaks']
  output['amp_on_peaks'] = amp_data['on_peaks']
  output['amp_on_delays'] = amp_data['on_delays']
  output['amp_on_fresp'] = amp_data['on_fresp']
  output['amp_on_fitting'] = amp_data['on_fitting']
  output['amp_off_peaks'] = amp_data['off_peaks']
  output['amp_off_delays'] = amp_data['off_delays']
  output['amp_off_fresp'] = amp_data['off_fresp']
  output['amp_off_fitting'] = amp_data['off_fitting']

  return output, char

def get_pop_response(spks, events, parameters, 
                     psth_bin, fit_resolution,
                     panalysis=None, feat_file=None):
  """
  Compute chirp response and psth from spiketimes

  """
  warnings.filterwarnings(action='once')

  sample_rate = 1 / 60 # should be a parameter
  chirp_signal = chirp_generator(sample_rate, parameters, splitted=True)

  exp_columns = ['flash_type',
                 'SNR',
                 'QI']
  flash_columns = ['on_latency',
                   'off_latency',
                   'bias_idx',
                   'on_decay',
                   'off_decay',
                   'on_resp_index',
                   'off_resp_index',
                   'on_sust_index',
                   'off_sust_index',
                   'on_flash_fmax',
                   'off_flash_fmax']
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
                  'off_freq_delay',
                  'peak_conflict']
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

  columns = []
  columns.extend(exp_columns)
  columns.extend(flash_columns)
  columns.extend(freq_columns)
  columns.extend(amp_columns)

  index = []
  cells_feat = np.empty((len(spks), len(columns)))
  cells_feat[:] = np.nan
  
  get_resp_args = [(spks, events, parameters, unit, psth_bin, fit_resolution) for unit in spks.keys()]
  num_threads = os.cpu_count()
  with Pool(num_threads) as pool:
    cells_proc = list(tqdm(pool.istarmap(get_chirp_response, get_resp_args), desc='Cells processing', ncols=100, total=len(get_resp_args)))

  for idx, unit in enumerate(tqdm(spks.keys(), desc='Saving info', ncols=100)):
    output, char = cells_proc[idx][0], cells_proc[idx][1]

    # Mapping temp_0 -> Unit_0001
    def temp_to_unit(key):
      unit = int(key.split('_')[-1]) + 1
      return 'Unit_{:04d}'.format(unit)
    index.append(temp_to_unit(unit))

    cells_feat[idx, :] = np.concatenate(([char['flash_type'], char['SNR'], char['QI']], char['flash_char'], char['freq_char'], char['amp_char']))

    if panalysis is not None:
      with h5py.File(panalysis, 'a') as hdf5_resp:
        cell_group = hdf5_resp.require_group(temp_to_unit(unit))
        chirp_group = cell_group.require_group('chirp_signal')
        for key, data in chirp_signal.items():
          if key in chirp_group:
            del chirp_group[key]
          chirp_group.create_dataset(key, data.shape, data=data)

        resp_group = cell_group.require_group('cell_resp')
        for key, data in output.items():
          if key in resp_group:
            del resp_group[key]
          resp_group.create_dataset(key, data.shape, data=data)

  df = pd.DataFrame(cells_feat, columns=columns, index=index)
  df['flash_type'] = df['flash_type'].replace(to_replace=[0, 1, 2, 3], value=['Null', 'ON', 'OFF', 'ON/OFF'])

  if feat_file is not None:
    with open(feat_file, 'w') as feat_csv:
      df.to_csv(feat_csv)

  return df
