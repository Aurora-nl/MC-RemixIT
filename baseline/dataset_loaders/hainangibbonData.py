"""
@brief Pytorch dataloader for Hainan Gibbon dataset.
@author
"""
import os
import glob
import numpy as np
import torch
import torchaudio

import baseline.dataset_loaders.abstract_dataset as abstract_dataset


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for Hainan Gibbon dataset
        Handles train / val / test with synthetic & real subsets
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.zero_pad = self.kwargs.get("zero_pad", True)  # 默认启用零填充

        self.normalize_audio = self.get_arg_and_check_validness(
            'normalize_audio', known_type=bool)
        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['train', 'val', 'test'])
        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[lambda x: x >= -1])
        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)
        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)
        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[8000, 16000])

        # root_dir should be passed in hparams
        self.root_dir = self.kwargs.get('root_dir')
        if self.root_dir is None:
            # 默认路径
            self.root_dir = "../datasets/HainanGibbon/"
        self.dataset_dirpath = os.path.join(self.root_dir, self.split)

        # Collect file lists depending on split
        if self.split == "train":
            self.available_filenames = sorted(
                glob.glob(os.path.join(self.dataset_dirpath, "*.wav"))
            )
        else:
            # validation & test have synthetic + real
            synthetic_mix = glob.glob(os.path.join(self.dataset_dirpath, "synthetic/mixtures/*.wav"))
            synthetic_clean = glob.glob(os.path.join(self.dataset_dirpath, "synthetic/clean/*.wav"))
            synthetic_noise = glob.glob(os.path.join(self.dataset_dirpath, "synthetic/noise/*.wav"))
            real_files = glob.glob(os.path.join(self.dataset_dirpath, "real/*.wav"))

            # Store as dict for easier access
            self.available_filenames = {
                "mixtures": sorted(synthetic_mix),
                "clean": sorted(synthetic_clean),
                "noise": sorted(synthetic_noise),
                "real": sorted(real_files)
            }

        # If using fixed n_samples
        if isinstance(self.available_filenames, list):
            if self.n_samples > 0:
                self.available_filenames = self.available_filenames[:self.n_samples]
            self.n_samples = len(self.available_filenames)
        else:
            # for dict type (val/test)
            if self.n_samples > 0:
                for k in self.available_filenames:
                    self.available_filenames[k] = self.available_filenames[k][:self.n_samples]
            self.n_samples = len(self.available_filenames.get("real", []))

        self.time_samples = int(self.sample_rate * self.timelength)

    def wavread(self, path):
        waveform, fs = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)  # [time]
        else:
            waveform = waveform.squeeze(0)  # [time]

        if self.sample_rate < fs:
            waveform = torchaudio.functional.resample(
                waveform, fs, self.sample_rate, resampling_method="kaiser_window")
        elif self.sample_rate > fs:
            raise ValueError("Cannot upsample.")


        waveform = waveform - waveform.mean()
        if self.normalize_audio:
            waveform = waveform / (waveform.std() + 1e-8)

        desired_len = 160000
        if waveform.shape[0] < desired_len:
            pad_len = desired_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif waveform.shape[0] > desired_len:
            waveform = waveform[:desired_len]

        return waveform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Train only has mixtures
        if self.split == "train":
            wav_path = self.available_filenames[idx]
            mixture = self.wavread(wav_path)
            return self.get_padded_tensor(mixture, start_index=self._get_start_index(mixture))

        elif self.split == "val":
            val_real_path = self.available_filenames["real"][idx]
            val_real_mixture = self.wavread(val_real_path)
            return self.get_padded_tensor(val_real_mixture, start_index=0)
        elif self.split == "test":
            test_real_path = self.available_filenames["real"][idx]
            test_real_mixture = self.wavread(test_real_path)
            return self.get_padded_tensor(test_real_mixture, start_index=0)
        else:
            # For val/test, provide dict
            item = {}
            if idx < len(self.available_filenames["mixtures"]):
                mix_path = self.available_filenames["mixtures"][idx]
                clean_path = self.available_filenames["clean"][idx]
                noise_path = self.available_filenames["noise"][idx]
                item["mixture"] = self.get_padded_tensor(self.wavread(mix_path), start_index=0)
                item["clean"] = self.get_padded_tensor(self.wavread(clean_path), start_index=0)
                item["noise"] = self.get_padded_tensor(self.wavread(noise_path), start_index=0)
            else:
                real_idx = idx - len(self.available_filenames["mixtures"])
                real_path = self.available_filenames["real"][real_idx]
                item["real"] = self.get_padded_tensor(self.wavread(real_path), start_index=0)

            return item

    def _get_start_index(self, waveform):
        max_len = waveform.shape[-1]
        if self.augment and max_len > self.time_samples > 0:
            return np.random.randint(0, max_len - self.time_samples)
        return 0


def test_generator():
    root_dir = "../datasets/HainanGibbon/"
    data_loader = Dataset(
        root_dir=root_dir,
        sample_rate=16000,
        timelength=10.0,
        augment=True,
        zero_pad=True,
        split='train',
        normalize_audio=False,
        n_samples=10
    )
    generator = data_loader.get_generator(batch_size=8, num_workers=1)
    for batch in generator:
        print(batch.shape)
        break


if __name__ == "__main__":
    test_generator()
