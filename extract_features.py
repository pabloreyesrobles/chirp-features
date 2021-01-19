from configparser import ConfigParser, ExtendedInterpolation

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
import sys
import os
import faulthandler

from scipy.signal import hann, find_peaks, hilbert
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from chirp import matlab_hanning, get_freq_spikes, chirp_generator, get_chirp_response, get_chirp_features, get_chirp_subevents
from spikelib import visualizations as vis
from ipywidgets import IntSlider, interact, Dropdown, fixed
from spikelib import spiketools as spkt

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

folders = next(os.walk('.'))[1]
exp_dirs = []
for f in folders:
	if len(f.split('-')) > 1:
		exp_dirs.append(f)
print(exp_dirs)
count = 0

group_features = None
faulthandler.enable()

for exp in exp_dirs:	
	config.read(os.path.join(rootdir, exp, 'config.ini'))
	config.set('PROJECT', 'path', '{}/'.format(os.path.join(rootdir, exp)))
	with open(os.path.join(rootdir, exp, 'config.ini'), 'w+') as configfile:
		config.write(configfile)
	config.read(os.path.join(rootdir, exp, 'config.ini'))

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
	
	with h5py.File(sorting_file, 'r') as spks:
		spikes = spks['/spiketimes/']
		df = get_chirp_features(spikes, events, chirp_args, None, psth_bin, fit_resolution)
		df.to_csv(output_features, mode='w')

		# Extend big csv
		df.reset_index(inplace=True)
		df['experiment'] = [exp_name for _ in range(df.shape[0])]
		cols = df.columns.tolist()
		df = df[cols[-1:] + cols[:-1]]

		if isinstance(group_features, pd.DataFrame):
			group_features = group_features.append(df)
		else:
			group_features = df

	print('Done!\n')

if isinstance(group_features, pd.DataFrame):
	group_features.to_csv('chirp.csv', mode='w')