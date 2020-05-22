import mne
import os
import glob
from moabb.datasets.base import BaseDataset
from moabb.datasets import download as dl
import zipfile
import re

SPOT_PILOT_P300_URL = 'https://freidok.uni-freiburg.de/fedora/objects/freidok:154576/datastreams'


class SpotPilotData(BaseDataset):
    """P300 dataset from initial spot study.

    Dataset [1].

    **Dataset Description**

    This dataset contains multiple small trials of an auditory oddball paradigm. The paradigm presented two different
    sinusoidal tones. A low-pitched (500 Hz, 40 ms duration) non-target tone and a high-pitched (1000 Hz,
    40 ms duration) target tone. Subjects were instructed to attend to the high-pitched target tones and ignore the
    low-pitched tones.

    One trial (= one file) consisted of 90 tones, 15 targets and 75 non-targets. The order was pseudo-randomized in a
    way that at least two non-target tones occur between two target tones. Additionally, if you split the 90 tones of
    one trial into consecutive sets of six tones, there will always be exactly one target and five non-target tones
    in each set.

    In the first part of the experiment (run 1), each subject performed 50-70 trials with various different stimulus
    onset asynchronies (SOAs) -- i.e. the time between the onset of successive tones -- for each trial. In the second
    part (run 2), 4-5 SOAs were played, with blocks of 5 trials having the same SOA. All SOAs were in the range of 60
    ms to 600 ms. Regardless of the experiment part, after a set of five trials, subjects were given the opportunity
    to take a short break to e.g. drink etc.

    Finally, before and after each run, resting data was recorded. One minute with eyes open and one minute with eyes
    closed, i.e. in total four minutes of resting data are available for each subject.

    Data was recorded using a BrainAmp DC (BrainVision) amplifier and a 32 passive electrode EasyCap. The cap was
    placed according to the extended 10-20 electrode layout. The reference electrode was placed on the nose. Before
    recording, the cap was prepared such that impedances on all electrodes were around 20 kOhm. The EEG signal was
    recorded at 1000 Hz.

    The data contains 31 scalp channels, one EOG channel and five miscellaneous non-EEG signal channels. However,
    only scalp EEG and the EOG channel is available in all subjects. The markers in the marker file indicate the
    onset of target tones (21) and non-target tones (1).

    WARNING: Note that this wrapper currently only loads the second part of the experiment and uses pseudo-sessions
    to achieve the functionality to handle different conditions in MOABB. As a result, the statistical testing
    features of MOABB cannot be used for this dataset.

    References
    ----------

    .. [1] Sosulski, J., Tangermann, M.: Electroencephalogram signals recorded from 13 healthy subjects during an
    auditory oddball paradigm under different stimulus onset asynchrony conditions. Dataset. DOI: 10.6094/UNIFR/154576
    """

    def __init__(self, load_soa_60=False, reject_non_iid=False, load_single_trials=False, interval=None):
        self.load_soa_60 = load_soa_60
        self.reject_non_iid = reject_non_iid
        self.load_single_trials = load_single_trials
        self.stimulus_modality = 'tone_oddball'
        self.N_channels = 31
        code = 'Spot Pilot P300 dataset'
        code += ' single' if load_single_trials else ' pooled'
        interval = [0, 1] if interval is None else interval
        super().__init__(
            subjects=list(range(1, 13 + 1)),
            sessions_per_subject='varying',  # note that these are fake sessions corresponding to conditions
            events=dict(Target=21, NonTarget=1),
            code=code,
            interval=interval,
            paradigm='p300',
            doi='N/A')

    @staticmethod
    def _map_subject_to_filenumber(subject_number):
        # The ordering of the uploaded files makes no sense, this function maps subject_numbers to corresponding files
        mapping = [5, 2, 4, 6, 3, 1, 10, 7, 12, 9, 8, 11, 13]
        return mapping[subject_number - 1]

    @staticmethod
    def _filename_trial_info_extraction(filepath):
        info_pattern = 'Oddball_Run_([0-9]+)_Trial_([0-9]+)_SOA_[0-9]\\.([0-9]+)\\.vhdr'
        filename = filepath.split(os.path.sep)[-1]
        trial_info = dict()
        re_matches = re.match(info_pattern, filename)
        trial_info['run'] = int(re_matches.group(1))
        trial_info['trial'] = int(re_matches.group(2))
        trial_info['soa'] = int(re_matches.group(3))
        return trial_info

    def _get_single_run_data(self, file_path):
        # trial_info = self.filename_trial_info_extraction(file_path)
        non_scalp_channels = ['EOGvu', 'x_EMGl', 'x_GSR', 'x_Respi', 'x_Pulse', 'x_Optic']
        raw = mne.io.read_raw_brainvision(file_path, misc=non_scalp_channels, preload=True)
        raw.set_montage('standard_1020')
        if self.reject_non_iid:
            raw.set_annotations(raw.annotations[7:85])  # non-iid rejection
        return raw

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        sessions = {}

        for p_i, file_path in enumerate(file_path_list):
            file_exp_info = SpotPilotData._filename_trial_info_extraction(file_path)
            soa = file_exp_info['soa']
            trial = file_exp_info['trial']
            if soa == 60 and not self.load_soa_60:
                continue
            if self.load_single_trials:
                session_name = f'soa_{soa}_trial_{trial}'
            else:
                session_name = f'soa_{soa}'

            if session_name not in sessions.keys():
                sessions[session_name] = {}

            run_name = f'run_{p_i + 1}_soa_{p_i}'
            sessions[session_name][run_name] = self._get_single_run_data(
                file_path)

        return sessions

    def data_path(
            self,
            subject,
            path=None,
            force_update=False,
            update_path=None,
            verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        file_number = SpotPilotData._map_subject_to_filenumber(subject)
        url = f'{SPOT_PILOT_P300_URL}/FILE{file_number}/content'
        path_zip = dl.data_path(url, 'spot')
        path_folder = path_zip[:-8] + f'/subject{subject}'

        # check if has to unzip
        if not(os.path.isdir(path_folder)):
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_zip[:-7])

        # get the path to all files
        pattern = f'/*Run_2*.vhdr'
        subject_paths = glob.glob(
            path_folder + pattern)
        return sorted(subject_paths)
