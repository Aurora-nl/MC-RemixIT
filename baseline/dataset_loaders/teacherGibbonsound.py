import torch
import os
import numpy as np
import glob2
import torchaudio
from __config__ import TEACHER_GIBBONSOUND_ROOT_PATH
import baseline.dataset_loaders.abstract_dataset as abstract_dataset
class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """
    Dataset class for TeacherModelDataset, which contains gibbonsound and noise audio files.

    Example of kwargs:
        root_dirpath='/path/to/teacherModelDataset', split='train', sample_rate=8000, timelength=4.0, normalize_audio=False
    """

    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool, default_value=True)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['train', 'val', 'test'])

        self.min_or_max = self.get_arg_and_check_validness(
            'min_or_max', known_type=str, choices=['min', 'max'])

        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[16000])

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.normalize_audio = self.get_arg_and_check_validness(
            'normalize_audio', known_type=bool)

        self.dataset_dirpath = self.get_path()

        self.available_filenames = [
            os.path.basename(f) for f in
            glob2.glob(os.path.join(self.dataset_dirpath, 'gibbonsound') + '/*.wav')]

        # Check that all files are available
        for fname in self.available_filenames:
            for s_type in [ 'noise','gibbonsound']:
                this_path = os.path.join(self.dataset_dirpath, s_type, fname)
                if not os.path.lexists(this_path):
                    raise IOError(f"File not found in: {this_path}")

        self.n_samples = len(self.available_filenames)

        self.time_samples = int(self.sample_rate * self.timelength)

    def get_path(self):
        path = os.path.join(TEACHER_GIBBONSOUND_ROOT_PATH, self.split)
        if os.path.lexists(path):
            return path
        else:
            raise IOError('Dataset path: {} not found!'.format(path))

    def get_arg_and_check_validness(self, arg_name, known_type=None, choices=None, extra_lambda_checks=None,
                                    default_value=None):
        """ Helper function to fetch arguments and validate them. """
        value = self.kwargs.get(arg_name, default_value)

        if value is None:
            raise ValueError(f"Argument '{arg_name}' is required but not provided!")

        if not isinstance(value, known_type):
            raise TypeError(f"Argument '{arg_name}' should be of type {known_type}.")

        if choices and value not in choices:
            raise ValueError(f"Argument '{arg_name}' should be one of {choices}.")

        if extra_lambda_checks:
            for check in extra_lambda_checks:
                if not check(value):
                    raise ValueError(f"Argument '{arg_name}' failed additional check.")

        return value
    def wavread(self, path):
        waveform, _ = torchaudio.load(path)

        # Convert to single-channel if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        return waveform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fname = self.available_filenames[idx]
        gibbonsound_path = os.path.join(self.dataset_dirpath, 'gibbonsound', fname)
        noise_path = os.path.join(self.dataset_dirpath, 'noise', fname)

        gibbonsound_w = self.wavread(gibbonsound_path)
        noise_w = self.wavread(noise_path)

        max_len = max(gibbonsound_w.shape[-1], noise_w.shape[-1])
        start_index = 0
        if self.augment and max_len > self.time_samples > 0:
            start_index = np.random.randint(0, max_len - self.time_samples)

        gibbonsound_tensor = self.get_padded_tensor(gibbonsound_w, start_index=start_index)
        noise_tensor = self.get_padded_tensor(noise_w, start_index=start_index)
        return gibbonsound_tensor, noise_tensor

def test_teacher_model_dataset():
    import time
    batch_size = 1
    sample_rate = 16000
    timelength = 10.0
    fixed_n_sources =-1
    split = 'train'
    dataset = Dataset(
        min_or_max='min',
        zero_pad=True,
        split='train',
        sample_rate=sample_rate,
        timelength=timelength,
        augment='train' in split,
        normalize_audio=True )

    generator = dataset.get_generator(
        batch_size=batch_size, num_workers=batch_size)
    print(f"Obtained: {len(generator)} files with fixed n_sources: {fixed_n_sources}")
    before = time.time()
    for gibbonsound, noise in generator:
        print(gibbonsound.shape)
        print(noise.shape)
        assert gibbonsound.shape == (batch_size, 1, int(sample_rate * timelength))
        assert noise.shape == (batch_size, 1, int(sample_rate * timelength))
        break
    this_time = time.time() - before
    print(f"It took me: {this_time} secs to fetch the batch")

if __name__ == "__main__":
    test_teacher_model_dataset()
