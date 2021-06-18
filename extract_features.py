import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import sys
import os
import json
import faulthandler

import h5py
from tqdm import tqdm

from scipy.signal import hann, find_peaks, hilbert
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from chirp import chirp_generator, get_pop_response, get_chirp_subevents
from spikelib import visualizations as vis
from spikelib import spiketools as spkt

from configparser import ConfigParser, ExtendedInterpolation

def plot_resp(key, panalysis, csv_feat, save_dir):
	# Stimulus time and signal			
	df_feat = pd.read_csv(csv_feat, index_col=0)
	
	flash_type = df_feat.loc[key]['flash_type']
	data = resp['{}/cell_resp/'.format(key)]

	flash_time = data['flash_time'][:]
	flash_response = data['flash_resp'][:]
	flash_on_peak = df_feat.loc[key]['on_latency']
	flash_off_peak = df_feat.loc[key]['off_latency'] + 3 # off time start. Should be param
	flash_on_max = df_feat.loc[key]['on_flash_fmax']
	flash_off_max = df_feat.loc[key]['off_flash_fmax']

	freq_time = data['freq_time'][:]
	freq_response = data['freq_response'][:]

	amp_time = data['amp_time'][:]
	amp_response = data['amp_response'][:]

	def time_to_amp(arr):
		# Time axis changed to amplitude modulation
		# Max aplitude is 0.5. 0.125 added because of time_adap stim at the end
		return np.multiply(np.ones_like(arr) * 0.625 / 10, arr, out=np.full_like(arr, np.nan, dtype=np.double), where=arr!=np.nan)

	fig_fl, ax_fl = plt.subplots(figsize=(8, 4), constrained_layout=True) # Flash plot
	fig_fm, ax_fm = plt.subplots(figsize=(8, 4), constrained_layout=True) # Freq plot
	fig_am, ax_am = plt.subplots(figsize=(8, 4), constrained_layout=True) # Amp plot

	ax_fl.plot(flash_time, flash_response, color='tab:gray')
	ax_fl.set_title('{} response - Flash response'.format(key))
	ax_fl.set_xlabel('Time [s]')
	ax_fl.set_ylabel('Cell spike rate [Hz]')
	
	ax_fm.plot(freq_time, freq_response, color='tab:gray')
	ax_fm.set_title('{} response - Frequency modulation'.format(key))
	ax_fm.set_xlabel('Stimulus freq [Hz]')
	ax_fm.set_ylabel('Cell spike rate [Hz]')

	ax_am.plot(time_to_amp(amp_time), amp_response, color='tab:gray')
	ax_am.set_title('{} response - Amplitude modulation'.format(key))
	ax_am.set_xlabel('Stimulus amp modulation [-]')
	ax_am.set_ylabel('Cell spike rate [Hz]')

	if flash_type == 'ON' or flash_type == 'ON/OFF':
		# Flash peak
		ax_fl.plot(flash_on_peak, flash_on_max, 'o', color='tab:blue')
		ax_fl.axvline(flash_on_peak, linestyle='--', alpha=0.5)

		# Freq peaks and fitting
		t_on_fit = data['freq_on_fitting'][:][0]
		v_on_fit = data['freq_on_fitting'][:][1]
		t_on_peaks = data['freq_on_peaks'][:][0]
		v_on_peaks = data['freq_on_peaks'][:][1]

		# In order to plot on peaks in case the first is missing
		on_id_max = np.argmax(np.isnan(t_on_peaks[1:])) + 1

		ax_fm.plot(t_on_fit, v_on_fit, color='tab:cyan')
		ax_fm.plot(t_on_peaks[:on_id_max], v_on_peaks[:on_id_max], 'o', color='tab:blue', label='ON')
		
		# Amp peaks and fitting
		t_on_fit = time_to_amp(data['amp_on_fitting'][:][0])
		v_on_fit = data['amp_on_fitting'][:][1]
		t_on_peaks = time_to_amp(data['amp_on_peaks'][:][0])
		v_on_peaks = data['amp_on_peaks'][:][1]

		ax_am.plot(t_on_fit, v_on_fit, color='tab:cyan')
		ax_am.plot(t_on_peaks, v_on_peaks, 'o', color='tab:blue', label='ON')
		
		ax_fm.legend()
		ax_am.legend()

	if flash_type == 'OFF' or flash_type == 'ON/OFF':
		# Flash peak
		ax_fl.plot(flash_off_peak, flash_off_max, 'o', color='tab:green')
		ax_fl.axvline(flash_off_peak, linestyle='--', alpha=0.5)

		# Freq peaks and fitting
		t_off_fit = data['freq_off_fitting'][:][0]
		v_off_fit = data['freq_off_fitting'][:][1]
		t_off_peaks = data['freq_off_peaks'][:][0]
		v_off_peaks = data['freq_off_peaks'][:][1]

		off_id_max = np.argmax(np.isnan(t_off_peaks))

		ax_fm.plot(t_off_fit, v_off_fit, color='tab:olive')
		ax_fm.plot(t_off_peaks[:off_id_max], v_off_peaks[:off_id_max], 'o', color='tab:green', label='OFF')
		
		# Amp peaks and fitting
		t_off_fit = time_to_amp(data['amp_off_fitting'][:][0])
		v_off_fit = data['amp_off_fitting'][:][1]
		t_off_peaks = time_to_amp(data['amp_off_peaks'][:][0])
		v_off_peaks = data['amp_off_peaks'][:][1]

		ax_am.plot(t_off_fit, v_off_fit, color='tab:olive')
		ax_am.plot(t_off_peaks, v_off_peaks, 'o', color='tab:green', label='OFF')
		
		ax_fm.legend()
		ax_am.legend()

	fig_fl.savefig(os.path.join(save_dir[0], '{}.png'.format(key)))
	fig_fm.savefig(os.path.join(save_dir[1], '{}.png'.format(key)))
	fig_am.savefig(os.path.join(save_dir[2], '{}.png'.format(key)))

	# Clear fig and axes
	ax_fl.clear()
	ax_fm.clear()
	ax_am.clear()
	fig_fl.clf()
	fig_fl.clear()
	fig_fm.clf()
	fig_fm.clear()
	fig_am.clf()
	fig_am.clear()

	plt.close('all')

def plot_features(key, panalysis, csv_feat, save_dir):
	# Stimulus time and signal			
	df_feat = pd.read_csv(csv_feat, index_col=0)
	
	flash_type = df_feat.loc[key]['flash_type']
	data = resp['{}/cell_resp/'.format(key)]

	fig, ax = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
	fig.suptitle('{} response'.format(key), fontsize=16)
	
	pt = 100
	lin_freq = np.linspace(0, 15, pt)
	ax[0][2].plot(lin_freq, lin_freq, '--', color='gray')
	ax[1][2].plot(lin_freq, np.ones(100), '--', color='gray')
	
	ax[0][0].set_xlim([-0.75, 15.75])
	ax[0][1].set_xlim([-0.75, 15.75])
	ax[0][2].set_xlim([-0.75, 15.75])
	ax[1][0].set_xlim([-0.05, 0.55])
	ax[1][1].set_xlim([-0.05, 0.55])
	ax[1][2].set_xlim([-0.05, 0.55])
	
	ax[0][1].set_ylim([-0.05, 1.0])
	ax[1][1].set_ylim([-0.05, 1.0])
	
	ax[0][2].set_ylim([-0.75, 15.75])
	ax[1][2].set_ylim([0.5, 1.5])
	
	ax[0][0].set_title('Freq fitting')
	ax[0][1].set_title('Freq delay')
	ax[0][2].set_title('Freq response')
	
	ax[1][0].set_title('Amp fitting')
	ax[1][1].set_title('Amp delay')
	ax[1][2].set_title('Amp response')
	
	ax[0][0].set_xlabel('fstimulus [Hz]')
	ax[0][0].set_ylabel('fspike [Hz]')
	ax[0][1].set_xlabel('fstimulus [Hz]')
	ax[0][1].set_ylabel('Delay [s]')
	ax[0][2].set_xlabel('fstimulus [Hz]')
	ax[0][2].set_ylabel('fresponse [Hz]')
	
	ax[1][0].set_xlabel('Amplitude')
	ax[1][0].set_ylabel('fspike [Hz]')
	ax[1][1].set_xlabel('Amplitude')
	ax[1][1].set_ylabel('Delay [s]')
	ax[1][2].set_xlabel('Amplitude')
	ax[1][2].set_ylabel('fresponse [Hz]')

	def time_to_amp(arr):
		# Time axis changed to amplitude modulation
		# Max aplitude is 0.5. 0.125 added because of time_adap stim at the end
		return np.multiply(np.ones_like(arr) * 0.625 / 10, arr, out=np.full_like(arr, np.nan, dtype=np.double), where=arr!=np.nan)

	if flash_type == 'ON' or flash_type == 'ON/OFF':            
			t_on_fit = data['freq_on_fitting'][:][0]
			v_on_fit = data['freq_on_fitting'][:][1]
			t_on_peaks = data['freq_on_peaks'][:][0]
			v_on_peaks = data['freq_on_peaks'][:][1]
			freq_on_delays = data['freq_on_delays'][:]
			freq_on_resp = data['freq_on_fresp'][:]
			t_on_delay_fit = data['freq_on_delay_fit'][:][0]
			v_on_delay_fit = data['freq_on_delay_fit'][:][1]
			
			on_id_max = np.argmax(np.isnan(t_on_peaks))
			
			ax[0][0].plot(t_on_fit, v_on_fit, color='tab:blue', label='ON')
			
			if on_id_max != 0:
					ax[0][0].plot(t_on_peaks[:on_id_max], v_on_peaks[:on_id_max], '.-', color='tab:cyan', label='ON')
					ax[0][1].plot(t_on_peaks[:on_id_max], freq_on_delays[:on_id_max], '.-', color='tab:cyan')            
					ax[0][1].plot(t_on_delay_fit, v_on_delay_fit, color='tab:blue')
					ax[0][2].plot(t_on_peaks[:on_id_max - 1], freq_on_resp[:on_id_max - 1], '.-', color='tab:cyan')   
			
			t_on_fit = time_to_amp(data['amp_on_fitting'][:][0])
			v_on_fit = data['amp_on_fitting'][:][1]
			t_on_peaks = time_to_amp(data['amp_on_peaks'][:][0])
			v_on_peaks = data['amp_on_peaks'][:][1]
			amp_on_delays = data['amp_on_delays'][:]
			amp_on_resp = data['amp_on_fresp'][:]

			ax[1][0].plot(t_on_fit, v_on_fit, color='tab:blue')
			ax[1][0].plot(t_on_peaks, v_on_peaks, '.-', color='tab:cyan')

			ax[1][1].plot(t_on_peaks, amp_on_delays, '.-', color='tab:cyan')

			ax[1][2].plot(t_on_peaks[:-1], amp_on_resp, '.-', color='tab:cyan')        

	if flash_type == 'OFF' or flash_type == 'ON/OFF':
			t_off_fit = data['freq_off_fitting'][:][0]
			v_off_fit = data['freq_off_fitting'][:][1]
			t_off_peaks = data['freq_off_peaks'][:][0]
			v_off_peaks = data['freq_off_peaks'][:][1]
			freq_off_delays = data['freq_off_delays'][:]
			freq_off_resp = data['freq_off_fresp'][:]
			t_off_delay_fit = data['freq_off_delay_fit'][:][0]
			v_off_delay_fit = data['freq_off_delay_fit'][:][1]
			
			off_id_max = np.argmax(np.isnan(t_off_peaks))
			
			ax[0][0].plot(t_off_fit, v_off_fit, color='tab:green', label='OFF')
			
			if off_id_max != 0:
					ax[0][0].plot(t_off_peaks[:off_id_max], v_off_peaks[:off_id_max], '.-', color='tab:olive', label='OFF')
					ax[0][1].plot(t_off_peaks[:off_id_max], freq_off_delays[:off_id_max], '.-', color='tab:olive')            
					ax[0][1].plot(t_off_delay_fit, v_off_delay_fit, color='tab:green')
					ax[0][2].plot(t_off_peaks[:off_id_max - 1], freq_off_resp[:off_id_max - 1], '.-', color='tab:olive')
			
			t_off_fit = time_to_amp(data['amp_off_fitting'][:][0])
			v_off_fit = data['amp_off_fitting'][:][1]
			t_off_peaks = time_to_amp(data['amp_off_peaks'][:][0])
			v_off_peaks = data['amp_off_peaks'][:][1]
			amp_off_delays = data['amp_off_delays'][:]
			amp_off_resp = data['amp_off_fresp'][:]

			ax[1][0].plot(t_off_fit, v_off_fit, color='tab:green')
			ax[1][0].plot(t_off_peaks, v_off_peaks, '.-', color='tab:olive')

			ax[1][1].plot(t_off_peaks, amp_off_delays, '.-', color='tab:olive')

			ax[1][2].plot(t_off_peaks[:-1], amp_off_resp, '.-', color='tab:olive')    
					
			handles, labels = ax[0][0].get_legend_handles_labels()
			fig.legend(handles, labels, loc='center right')

	plt.savefig(os.path.join(save_dir, '{}.png'.format(key)))
	
	[_ax.clear() for _ax in ax[0]]
	[_ax.clear() for _ax in ax[1]]
	fig.clf()
	fig.clear()
	plt.close('all')

if __name__ == "__main__":
	config = ConfigParser(interpolation=ExtendedInterpolation())
	# Chirp parameters
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

	sample_rate = 1 / 60

	chirp_signal = chirp_generator(sample_rate, chirp_args, splitted=True)
	psth_bin = 0.06
	fit_resolution = 0.001

	rootdir = os.getcwd()
	with open("params.json") as p:
		params = json.load(p)
	print(params)

	# Create output folder if it does not exist
	if os.path.isdir(params['Output']) == False:
		os.mkdir(params['Output'])

	group_features = None
	faulthandler.enable()

	fig_only = False
	no_fig = False

	if len(sys.argv) > 1:
		if '-fig' in sys.argv: # Fig only
			fig_only = True
		if '-no-fig' in sys.argv:
			no_fig = True
			print('Skipping figures')

	for exp in params['Experiments']:
		os.chdir(exp)	
		cfg_file = 'config.ini'
		config.read(cfg_file)

		config.set('PROJECT', 'path', '{}/'.format(os.getcwd()))
		with open(cfg_file, 'w+') as configfile:
			config.write(configfile)
		config.read(cfg_file)

		os.chdir(rootdir)

		exp_name = config['EXP']['name']
		sorting_file = config['FILES']['sorting']
		sync_file = config['SYNC']['events']
		start_end_path = config['SYNC']['frametime']
		repeated_path = config['SYNC']['repeated']
		output_file = os.path.join(config['SYNC']['folder'],
									'sub_event_list_{}_chirp.csv'.format(exp_name))
		output_features = os.path.join(config['SYNC']['folder'],
										'chirp_features_{}.csv'.format(exp_name))

		# General parameters
		samplerate = 20000.0

		# Temporal resolution of flash response
		psth_bin = 0.06  # In sec
		bandwidth_fit = psth_bin
		fit_resolution = 0.001  # In sec

		names = ['ON', 'OFF', 'adap_0', 'FREQ', 'adap_1', 'AMP', 'adap_2']
		times = [3, 3, 2, 15, 2, 8, 2]

		print('Processing: {} experiment...'.format(exp_name))

		events = get_chirp_subevents(sync_file, start_end_path, repeated_path, output_file, names, times)
		"""
		try:
			events = get_chirp_subevents(sync_file, start_end_path, repeated_path, output_file, names, times)
		except Exception as e:
			print(e)
			print('Error getting chirp sub events')
			continue
		"""
		if isinstance(events, pd.DataFrame) is not True:
			print('Error computing chirp sub events')
			continue
		
		exp_output = os.path.join(params['Output'], exp_name)
		if os.path.isdir(exp_output) == False:
			os.mkdir(exp_output)
		
		resp_file = os.path.join(exp_output, 'response.hdf5')
		feat_file = os.path.join(exp_output, 'features.csv')

		if fig_only is False:
			print('Response and features:')
			try:
				with h5py.File(sorting_file, 'r') as spks:
					get_pop_response(spks['/spiketimes/'], events, chirp_args, psth_bin, fit_resolution,
														panalysis=resp_file, feat_file=feat_file)
			except Warning as e:
				print(e)
				print('Error getting response and features')
				continue

		if no_fig is False:
			print('Figures:')
			fig_dir = os.path.join(exp_output, 'fig')
			if os.path.isdir(fig_dir) == False:
				os.mkdir(fig_dir)

			flash_dir = os.path.join(fig_dir, 'flashes')
			if os.path.isdir(flash_dir) == False:
				os.mkdir(flash_dir)
			freq_dir = os.path.join(fig_dir, 'freq_mod')
			if os.path.isdir(freq_dir) == False:
				os.mkdir(freq_dir)
			amp_dir = os.path.join(fig_dir, 'amp_mod')
			if os.path.isdir(amp_dir) == False:
				os.mkdir(amp_dir)

			# Now resp_dir is vector folder
			resp_dir = [flash_dir, freq_dir, amp_dir]
			
			feat_dir = os.path.join(fig_dir, 'feat')
			if os.path.isdir(feat_dir) == False:
				os.mkdir(feat_dir)

			with h5py.File(resp_file, 'r') as resp:
				for key in tqdm(resp.keys()):
					plot_resp(key, resp, feat_file, resp_dir)
					plot_features(key, resp, feat_file, feat_dir)
					plt.close('all')

		print('Done!\n')
		