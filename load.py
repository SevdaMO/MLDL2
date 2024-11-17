import os
from mlcroissant import Dataset
import wfdb
import numpy as np
from utils import read_event_file_as_list, yml_import, get_question_mapping
import logging
from tqdm import tqdm
import argparse

ANALYSIS_DATA_FILES = ['AF.txt', 'Akt.txt', 'Alpha+Beta FFT.txt', 'Autonome Arousal.txt', 'Average Frequency Value.txt', 'CAP.txt', 'Cheyne Stokes.txt', 'Delta FFT.txt', 'Diastol PTT.txt', 'Effort Ext1 Anstieg.txt', 'Effort Ext1 Events.txt', 'Effort Ext2 Anstieg.txt', 'Effort Ext2 Events.txt', 'Flow Events.txt', 'HRV HF.txt', 'HRV LF.txt', 'Herzfrequenz Kurve.txt', 'Integral EMG.txt', 'Kardio Events.txt', 'Klassifizierte Arousal.txt', 'Klassifizierte PTT.txt', 'Körperlage.txt', 'Licht.txt', 'Marker.txt', 'Obstruktion.txt', 'PLM Events.txt', 'PTT Raw.txt', 'Phasenw. Events.txt', 'Phasenwinkel.txt', 'REM Prüfung.txt', 'REM.txt', 'RR-Intervall.txt', 'SVB.txt', 'Schlaf Profil Sicherheit.txt', 'Schlafprofil.txt', 'Schnarchen Events.txt', 'Sigma FFT.txt', 'SpO2 Events.txt', 'SpO2.txt', 'Spindel  K.txt', 'Spindelfrequenz.txt', 'Systol. PTT.txt']

YAML_DATA_FILES = ['allgemeiner_schlaffragebogen_1.yml', 'allgemeiner_schlaffragebogen_1_2.yml', 'arztbrief_1.yml', 'epworth_sleepiness_scale.yml', 'psqi_fragebogen_1.yml', 'psqi_fragebogen_2.yml', 'psqi_fragebogen_3.yml', 'psqi_fragebogen_4.yml', 'restless_legs_fragebogen.yml', 'scorer.yml']

def load_signals(base_path, sample_ids):
    psg_files = [os.path.join(base_path, sample_id, 'PSG', sample_id) for sample_id in sample_ids]

    signal_data = {}

    for psg_file in tqdm(psg_files, desc='Loading signals'):
        record = wfdb.rdrecord(psg_file)
        signal = np.transpose(record.p_signal)

        sort_index = np.argsort(record.sig_name)
        sorted_sig_name = np.array(record.sig_name)[sort_index]
        sorted_signal = signal[sort_index]

        signal_data[record.record_name] = {signal_name: signal_data for signal_name, signal_data in zip(sorted_sig_name, sorted_signal)}

    return signal_data


def load_analysis_and_yaml_files(base_path, sample_ids, analysis_data_files=ANALYSIS_DATA_FILES, yaml_data_files=YAML_DATA_FILES):

    analysis_data = {}
    yaml_data = {}
    logging.info(f'Loading {len(yaml_data_files)} yaml files and {len(analysis_data_files)} analysis files.')
    for sample_id in tqdm(sample_ids, desc='Loading analysis and yaml files'):
        analysis_data[sample_id] = {}
        yaml_data[sample_id] = {}

        for analysis_data_file in analysis_data_files:
            path = os.path.join(base_path, sample_id, 'PSG', 'Analysedaten', analysis_data_file)
            try:
                file_as_list, _, _ = read_event_file_as_list(path)
                analysis_data[sample_id][analysis_data_file] = file_as_list
            except FileNotFoundError:
                print(f"Warning: File {analysis_data_file} from sample {sample_id} not found. Leaving it empty.")
                analysis_data[sample_id][analysis_data_file] = None

        for yaml_data_file in yaml_data_files:
            path = os.path.join(base_path, sample_id, 'YAML', yaml_data_file)
            try:
                data = yml_import(path)
                yaml_data[sample_id][yaml_data_file] = data
            except FileNotFoundError:
                print(f"Warning: File {yaml_data_file} from sample {sample_id} not found. Leaving it empty.")
                yaml_data[sample_id][yaml_data_file] = None

    return analysis_data, yaml_data

def setup():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Load data samples.')
    parser.add_argument('--sample_ids', nargs='*', help='List of sample IDs to load.')
    parser.add_argument('--analysis_data_files', nargs='*', help='List of analysis data files to load.')
    parser.add_argument('--yaml_data_files', nargs='*', help='List of YAML data files to load.')
    args = parser.parse_args()
    return args


def main():

    args = setup()    

    dataset = Dataset(jsonld="croissant.json")

    base_path = dataset.metadata.url
    directory = os.path.dirname(__file__)
    base_path = os.path.join(directory, base_path)
    
    sample_ids = args.sample_ids if args.sample_ids else os.listdir(base_path)
    analysis_data_files = args.analysis_data_files if args.analysis_data_files else ANALYSIS_DATA_FILES
    yaml_data_files = args.yaml_data_files if args.yaml_data_files else YAML_DATA_FILES

    signals = load_signals(base_path, sample_ids)
    logging.info(f'Loaded {len(signals)} signals.')

    analysis_data, yaml_data = load_analysis_and_yaml_files(base_path, sample_ids, analysis_data_files, yaml_data_files)
    logging.info(f'Loaded analysis data for {len(analysis_data)} samples.')
    logging.info(f'Loaded YAML data for {len(yaml_data)} samples.')

    # Optional: Get the question mapping
    yaml_data_mapped = get_question_mapping(yaml_data)


if __name__ == '__main__':
    main()