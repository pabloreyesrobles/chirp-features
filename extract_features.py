from configparser import ConfigParser, ExtendedInterpolation

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
import sys
import os
import json
import faulthandler
from tqdm import tqdm

from scipy.signal import hann, find_peaks, hilbert
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from chirp import chirp_generator, get_pop_response, get_chirp_subevents
from spikelib import visualizations as vis
from ipywidgets import IntSlider, interact, Dropdown, fixed
from spikelib import spiketools as spkt

def plot_resp(panalysis, csv_feat, save_dir):
	# Stimulus time and signal			
	df_feat = pd.read_csv(csv_feat, index_col=0)
	
	with h5py.File(panalysis, 'r') as resp:
		for key in tqdm(resp.keys()):
			flash_type = df_feat.loc[key]['flash_type']
			data = resp['{}/cell_resp/'.format(key)]

			freq_time = data['freq_time'][:]
			freq_response = data['freq_response'][:]

			amp_time = data['amp_time'][:]
			amp_response = data['amp_response'][:]

			fig, ax = plt.subplots(2, figsize=(10, 10), constrained_layout=True)
			unit = int(key.split('_')[-1]) + 1
			fig.suptitle('Unit_{:04d} response'.format(unit), fontsize=16)
			
			ax[0].plot(freq_time, freq_response, color='tab:gray')
			ax[1].plot(amp_time, amp_response, color='tab:gray')
			
			ax[0].set_title('Frequency modulation')
			ax[1].set_title('Amplitude modulation')

			if flash_type == 1 or flash_type == 3:
				# Freq peaks and fitting
				t_on_fit = data['freq_on_fitting'][:][0]
				v_on_fit = data['freq_on_fitting'][:][1]
				t_on_peaks = data['freq_on_peaks'][:][0]
				v_on_peaks = data['freq_on_peaks'][:][1]

				on_id_max = np.argmax(np.isnan(t_on_peaks)) 

				ax[0].plot(t_on_fit, v_on_fit, color='tab:cyan')
				ax[0].plot(t_on_peaks[:on_id_max], v_on_peaks[:on_id_max], 'o', color='tab:blue', label='ON')
				
				# Amp peaks and fitting
				t_on_fit = data['amp_on_fitting'][:][0]
				v_on_fit = data['amp_on_fitting'][:][1]
				t_on_peaks = data['amp_on_peaks'][:][0]
				v_on_peaks = data['amp_on_peaks'][:][1]

				ax[1].plot(t_on_fit, v_on_fit, color='tab:cyan')
				ax[1].plot(t_on_peaks, v_on_peaks, 'o', color='tab:blue', label='ON')
				
				ax[0].legend()
				ax[1].legend()

			if flash_type == 2 or flash_type == 3:
				# Freq peaks and fitting
				t_off_fit = data['freq_off_fitting'][:][0]
				v_off_fit = data['freq_off_fitting'][:][1]
				t_off_peaks = data['freq_off_peaks'][:][0]
				v_off_peaks = data['freq_off_peaks'][:][1]

				off_id_max = np.argmax(np.isnan(t_off_peaks))

				ax[0].plot(t_off_fit, v_off_fit, color='tab:olive')
				ax[0].plot(t_off_peaks[:off_id_max], v_off_peaks[:off_id_max], 'o', color='tab:green', label='OFF')
				
				# Amp peaks and fitting
				t_off_fit = data['amp_off_fitting'][:][0]
				v_off_fit = data['amp_off_fitting'][:][1]
				t_off_peaks = data['amp_off_peaks'][:][0]
				v_off_peaks = data['amp_off_peaks'][:][1]

				ax[1].plot(t_off_fit, v_off_fit, color='tab:olive')
				ax[1].plot(t_off_peaks, v_off_peaks, 'o', color='tab:green', label='OFF')
				
				ax[0].legend()
				ax[1].legend()

			plt.savefig(os.path.join(save_dir, '{}.png'.format(key)))
			plt.cla()
			plt.clf() 
			plt.close(fig)

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
									'sub_event_list_{}_protocol_name.csv'.format(exp_name))
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
		if isinstance(events, pd.DataFrame) is not True:
			print('Error while processing')
			continue
		
		exp_output = os.path.join(params['Output'], exp_name)
		if os.path.isdir(exp_output) == False:
			os.mkdir(exp_output)
		
		resp_file = os.path.join(exp_output, 'response.hdf5')
		feat_file = os.path.join(exp_output, 'features.csv')
		
		print('Response and features:')
		"""
		with h5py.File(sorting_file, 'r') as spks:
			get_pop_response(spks['/spiketimes/'], events, chirp_args, psth_bin, fit_resolution,
														panalysis=resp_file, feat_file=feat_file)
		"""

		print('Figures:')
		fig_dir = os.path.join(exp_output, 'fig')
		if os.path.isdir(fig_dir) == False:
			os.mkdir(fig_dir)

		resp_dir = os.path.join(fig_dir, 'resp')
		if os.path.isdir(resp_dir) == False:
			os.mkdir(resp_dir)

		plot_resp(resp_file, feat_file, resp_dir)
		plt.close('all')

		print('Done!\n')