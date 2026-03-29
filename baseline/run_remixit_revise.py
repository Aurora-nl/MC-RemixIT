import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from __config__ import API_KEY
from comet_ml import Experiment

import copy
import torch
import numpy as np
import argparse
import pandas as pd
import os
from tqdm import tqdm
from pprint import pprint
import baseline.utils.cmd_parser as parser
import baseline.utils.cometml_logger as cometml_logger
import baseline.utils.dataset_setup as dataset_setup
import baseline.utils.mixture_consistency as mixture_consistency
import baseline.utils.CombineLoss_EnSp as combineLossnew
import baseline.models.improved_sudormrf_revise as improved_sudormrf
import baseline.metrics.dnnmos_metric as dnnmos_metric
from multiprocessing import Pool, Semaphore
from asteroid.losses import pairwise_neg_sisdr
from asteroid.losses import pairwise_neg_snr

torch.backends.cudnn.benchmark = True  # cuDNN

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

combineLoss_sse = combineLossnew.CombinedLoss(
    use_energy=True,
    use_flatness=True,
    use_Xcorr=False,
    w_energy=0.1,
    w_flat=0.05,
    w_xcorr=0.2
)

args = parser.get_args()
hparams = vars(args)
generators = dataset_setup.unsupervised_setup(hparams)

def get_parameter_groups(model, base_lr):
    main_params = []
    bak_params = []
    for name, param in model.named_parameters():
        if "bak_suppress" in name:
            bak_params.append(param)
        else:
            main_params.append(param)
    return [
        {'params': main_params, 'lr': base_lr},
        {'params': bak_params, 'lr': base_lr * 0.1}
    ]

def get_new_student(hparams, depth_growth):
    student = improved_sudormrf.SuDORMRF(
        out_channels=hparams["out_channels"],
        in_channels=hparams["in_channels"],
        num_blocks=int(depth_growth * hparams["num_blocks"]),
        upsampling_depth=hparams["upsampling_depth"],
        enc_kernel_size=hparams["enc_kernel_size"],
        enc_num_basis=hparams["enc_num_basis"],
        num_sources=2,
    )
    return student

def freeze_model(model):
    for f in model.parameters():
        if f.requires_grad:
            f.requires_grad = False


def correlatelossFunction(teacher_est_active_speakers, teacher_est_noises):
    eps = 1e-8
    batch_size = teacher_est_active_speakers.shape[0]
    signal_length = teacher_est_active_speakers.shape[2]

    sig1 = (teacher_est_active_speakers - torch.mean(teacher_est_active_speakers, dim=2, keepdim=True)).squeeze(1)
    sig2 = (teacher_est_noises - torch.mean(teacher_est_noises, dim=2, keepdim=True)).squeeze(1)

    energy_sig1 = torch.sum(sig1 ** 2, dim=1)  # [16]
    energy_sig2 = torch.sum(sig2 ** 2, dim=1)  # [16]
    energy_norm = torch.sqrt(energy_sig1 * energy_sig2) + eps  # [16]

    batch_corr_norm = []

    for i in range(batch_size):
        sig1_cpu = sig1[i].cpu()
        sig2_cpu = sig2[i].cpu()

        sig1_fft = torch.fft.fft(sig1_cpu, n=2 * signal_length)
        sig2_fft = torch.fft.fft(sig2_cpu, n=2 * signal_length)

        # cross-correlation: IFFT(FFT(sig1) * conj(FFT(sig2)))
        corr_fft = sig1_fft * torch.conj(sig2_fft)
        corr = torch.fft.ifft(corr_fft).real

        corr_max = torch.max(torch.abs(corr))

        corr_norm = corr_max / energy_norm[i].cpu()
        batch_corr_norm.append(corr_norm)

    batch_corr_norm = torch.stack(batch_corr_norm).to(teacher_est_active_speakers.device)  # [16]
    loss1 = torch.mean(batch_corr_norm) * 10

    return loss1


def apply_output_transform(rec_sources_wavs, input_mix_std,
                           input_mix_mean, input_mom, hparams):
    if hparams["rescale_to_input_mixture"]:
        rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
    if hparams["apply_mixture_consistency"]:
        rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs, input_mom)
    return rec_sources_wavs


def normalize_waveform(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-9)


def compute_dnsmos_process(x):
    """DNSMOS"""
    try:
        return dnnmos_metric.compute_dnsmos(x, fs=16000)
    except Exception as e:
        print(f"[Warning] DNSMOS failed：{e}")
        return None


# load audio
audio_logger = cometml_logger.AudioLogger(fs=hparams["fs"], n_sources=2)

experiment = Experiment(API_KEY, project_name=hparams["project_name"])
experiment.log_parameters(hparams)
experiment_name = '_'.join(hparams['cometml_tags'])

for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

checkpoint_storage_path = os.path.join(hparams["checkpoint_storage_path"],
                                       experiment_name)


if checkpoint_storage_path is not None:
    if hparams["save_models_every"] <= 0:
        raise ValueError("Expected a value greater than 0 for checkpoint storing.")
    if not os.path.exists(checkpoint_storage_path):
        os.makedirs(checkpoint_storage_path)
        print(f"Created directory: {checkpoint_storage_path}")

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

train_loss_name, train_loss = "train_neg_sisdr", pairwise_neg_sisdr

val_losses = {
    "train_speaker": {"sisdr": pairwise_neg_sisdr},
    "train_noise": {"sisdr": pairwise_neg_sisdr},
    "train_total": {"sisdr": pairwise_neg_sisdr},
}

for val_set in [x for x in generators if not x == 'train']:
    if generators[val_set] is None:
        continue
    if val_set in ['val_chime_1sp', 'test_chime_1sp', 'val_hainangibbon', 'test_hainangibbon']:
        val_losses[val_set] = {
            "sig_mos": None,
            "bak_mos": None,
            "ovr_mos": None,
        }
    else:
        val_losses[val_set] = {
            "sisdr": pairwise_neg_sisdr, "sisdri": pairwise_neg_sisdr
        }

# Get initial teacher and student models
student = get_new_student(hparams, depth_growth=1)
teacher = get_new_student(hparams, depth_growth=1)
if not 0. <= hparams["teacher_momentum"] <= 1.:
    raise ValueError("Teacher momentum should be in the range of [0, 1] but got: "
                     f"{hparams['teacher_momentum']}")
if hparams["initialize_student_from_checkpoint"]:
    # Initialize the student with the same checkpoint as the teacher.
    student.load_state_dict(torch.load(hparams["warmup_checkpoint"]))


teacher.load_state_dict(torch.load(hparams["warmup_checkpoint"]))
student = torch.nn.DataParallel(student).cuda()
teacher = torch.nn.DataParallel(teacher).cuda()

freeze_model(teacher)

param_groups = get_parameter_groups(student, hparams['learning_rate'])
opt = torch.optim.Adam(param_groups)

initial_seed = 17

tr_step = 0
val_step = 0
sum_loss = 0.
student_step = 1
student_order = 1
train_loss_save_csv = []
val_loss_save_csv = []
momentum_degrade1 = 0.005
momentum_degrade2 = 0.01
t_momentum = hparams["teacher_momentum"]
gama_momentum = (1.0 - t_momentum)
update_momentum = 10
current_momentum = t_momentum

num_of_workers = max(os.cpu_count() // 2, 1)

poolval = Pool(num_of_workers)

try:
    for i in range(hparams['n_epochs']):
        # Set seeds for reproducability
        torch.manual_seed(initial_seed + i)
        np.random.seed(initial_seed + i)

        res_dic = {}
        for d_name in val_losses:
            res_dic[d_name] = {}
            for loss_name in val_losses[d_name]:
                res_dic[d_name][loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        print("RemixIT w Sudo-RM-RF: {} - {} | Epoch: {}/{} | St step: {}".format(
            experiment.get_key(), experiment.get_tags(), i + 1, hparams['n_epochs'],
            student_step))

        # Figure out which student order is and replace teacher if needed
        # Figure out which student order is and replace teacher if needed
        if hparams["n_epochs_teacher_update"] is not None:
            update_needed = i // hparams["n_epochs_teacher_update"] + 1 > student_order
            if update_needed and hparams["student_depth_growth"] > 1.:
                # Sequential teacher update protocol.
                # Replace old teacher with the newest student and update order
                del teacher
                teacher = student.module.cpu()
                del student
                old_student_depth = hparams["student_depth_growth"] ** (student_order - 1)
                new_student_growth = hparams["student_depth_growth"] ** student_order
                student = get_new_student(hparams, depth_growth=new_student_growth)
                student = torch.nn.DataParallel(student).cuda()
                teacher = torch.nn.DataParallel(teacher).cuda()
                opt = torch.optim.Adam(student.parameters(), lr=hparams["learning_rate"])
                print(f"Replaced old teacher with latest student: {old_student_depth} -> {new_student_growth}")
                student_step = 1
                student_order = i // hparams["n_epochs_teacher_update"]

            elif update_needed and hparams["teacher_momentum"] > 0.:
                # # Exponential moving average protocol.
                # t_momentum = hparams["teacher_momentum"]

                new_teacher_w = copy.deepcopy(teacher.state_dict())
                student_w = student.state_dict()
                for key in new_teacher_w.keys():
                    new_teacher_w[key] = (
                            current_momentum * new_teacher_w[key] + gama_momentum * student_w[key])

                teacher.load_state_dict(new_teacher_w)
                del new_teacher_w
                freeze_model(teacher)
                print(f"Updated the teacher with EMA in the {student_order}-th student order.")
                student_step = 1
                student_order = i // hparams["n_epochs_teacher_update"]

        student.train()
        teacher.eval()
        train_tqdm_gen = tqdm(generators['train'], desc='Training')

        sum_loss = 0.0
        for cnt, input_mix in enumerate(train_tqdm_gen):
            opt.zero_grad()

            input_mix = input_mix.unsqueeze(1).cuda()
            input_mix_std = input_mix.std(-1, keepdim=True)
            input_mix_mean = input_mix.mean(-1, keepdim=True)
            input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

            with torch.no_grad():
                # Teacher's estimates
                teacher_estimates = teacher(input_mix).detach()
                teacher_estimates = apply_output_transform(
                    teacher_estimates, input_mix_std, input_mix_mean, input_mix, hparams)
                t_est_speech, t_est_noise = teacher_estimates[:, 0:1], teacher_estimates[:, 1:]
                batch_size, n_noises, _ = t_est_noise.shape
                # Bootstrapped remixing
                permuted_t_est_noise = t_est_noise[torch.randperm(batch_size)]
                # permuted_t_est_noise -= permuted_t_est_noise.mean(-1, keepdim=True)
                # t_est_speech -= t_est_speech.mean(-1, keepdim=True)
                bootstrapped_mix = t_est_speech + permuted_t_est_noise

                bootstrapped_mix_std = bootstrapped_mix.std(-1, keepdim=True)
                bootstrapped_mix_mean = bootstrapped_mix.mean(-1, keepdim=True)
                bootstrapped_mix = (bootstrapped_mix - bootstrapped_mix_mean) / (
                        bootstrapped_mix_std + 1e-9)

            # Apply the student model and regress over teacher's estimates
            student_estimates = student(bootstrapped_mix)
            student_estimates = apply_output_transform(
                student_estimates, bootstrapped_mix_std, bootstrapped_mix_mean,
                bootstrapped_mix, hparams)
            s_est_speech, s_est_noise = student_estimates[:, 0:1], student_estimates[:, 1:]

            # Regress over the teacher estimated speech and the permuted noise estiamtes
            speaker_l = torch.mean(
                torch.clamp(train_loss(s_est_speech, t_est_speech.detach()),
                            min=-30., max=+30.))

            noise_l = torch.mean(
                torch.clamp(train_loss(s_est_noise, permuted_t_est_noise.detach()),
                            min=-30., max=+30.))

            seperation_list = [s_est_speech, s_est_noise]
            l2 = combineLoss_sse(s_est_speech, s_est_noise, bootstrapped_mix, separated_list = seperation_list)
            crossCorrelation_l = correlatelossFunction(s_est_speech, s_est_noise)

            l =0.6 * speaker_l + 0.4 * noise_l + 0.2 * crossCorrelation_l + l2

            l.backward()
            if hparams['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), hparams['clip_grad_norm'])

            opt.step()

            np_loss_value = l.detach().item()
            sum_loss += np_loss_value

            train_tqdm_gen.set_description(
                f"Training - Avg Loss: {round(sum_loss / (cnt + 1), 2)} ")
            res_dic['train_total']['sisdr']['acc'] += [- l.detach().cpu()]
            res_dic['train_speaker']['sisdr']['acc'] += [- speaker_l.detach().cpu()]
            res_dic['train_noise']['sisdr']['acc'] += [- noise_l.detach().cpu()]

            if cnt + 1 == train_tqdm_gen.total:
                train_loss_total = sum_loss / (cnt + 1)
                experiment.log_metric("train_loss_epoch", train_loss_total, step=i)
                train_loss_save_csv.append(train_loss_total)

        if hparams['patience'] > 0:
            if student_step % hparams['patience'] == 0:
                new_lr = (hparams['learning_rate'] / (hparams['divide_lr_by'] ** (
                        student_step // hparams['patience'])))
                print('Reducing Learning rate to: {}'.format(new_lr))
                for param_group in opt.param_groups:
                    param_group['lr'] = new_lr
        tr_step += 1
        student_step += 1

        run_validation = ((i + 1) % 2 == 0)
        run_test = ((i + 1) % 2 == 0)

        if run_validation or run_test:
            for val_d_name in [x for x in generators if not x == 'train']:
                if not run_validation and "val" in val_d_name:
                    continue

                if not run_test and "test" in val_d_name:
                    continue

                if generators[val_d_name] is None:
                    continue

                if val_d_name in ['val_chime_1sp', 'test_chime_1sp', 'val_hainangibbon', 'test_hainangibbon']:
                    student.eval()
                    all_dnsmos_results, batch_estimates = [], []
                    with torch.inference_mode():
                        for mixture in tqdm(generators[val_d_name], desc='Validation on {}'.format(val_d_name)):
                            input_mix = mixture.unsqueeze(1).cuda()
                            input_mix_std = input_mix.std(-1, keepdim=True)
                            input_mix_mean = input_mix.mean(-1, keepdim=True)
                            input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

                            student_estimates = student(input_mix)
                            student_estimates = apply_output_transform(
                                student_estimates, input_mix_std, input_mix_mean, input_mix, hparams)

                            s_est_speech = student_estimates[:, 0].detach().cpu().numpy()
                            s_est_speech -= s_est_speech.mean(-1, keepdims=True)
                            s_est_speech /= np.abs(s_est_speech).max(-1, keepdims=True) + 1e-9
                            # batch_estimates.append(s_est_speech)
                            batch_estimates.extend([s_est_speech[b_ind] for b_ind in range(s_est_speech.shape[0])])

                            if len(batch_estimates) >= 8:
                                # batch_data = [xj for xj in batch_estimates]
                                batch_results = poolval.map(compute_dnsmos_process, batch_estimates)
                                all_dnsmos_results.extend([r for r in batch_results if r is not None])
                                batch_estimates = []

                        if batch_estimates:
                            batch_results = poolval.map(compute_dnsmos_process, batch_estimates)
                            all_dnsmos_results.extend([r for r in batch_results if r is not None])

                    for dnsmos_values in all_dnsmos_results:
                        if dnsmos_values:
                            for k1, v1 in dnsmos_values.items():
                                res_dic[val_d_name][k1]['acc'].append(v1)

                    if hparams["log_audio"]:
                        audio_logger.log_sp_enh_no_gt_batch(
                            student_estimates[:, 0:1].detach(),
                            student_estimates[:, 1:2].detach(),
                            input_mix.detach(),
                            experiment, step=val_step, tag=f"{val_d_name}_stud_{student_order}",
                            max_batch_items=4)

                else:
                    student.eval()
                    with torch.inference_mode():
                        for speakers, noise in tqdm(generators[val_d_name], desc='Validation on {}'.format(val_d_name)):
                            gt_speaker_mix = speakers.sum(1, keepdims=True).cuda()
                            noise = noise.cuda()

                            input_mix = noise + gt_speaker_mix
                            input_mix_std = input_mix.std(-1, keepdim=True)
                            input_mix_mean = input_mix.mean(-1, keepdim=True)
                            input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

                            rec_sources_wavs = student(input_mix)
                            rec_sources_wavs = apply_output_transform(
                                rec_sources_wavs, input_mix_std, input_mix_mean, input_mix, hparams)
                            teacher_est_active_speakers = rec_sources_wavs[:, 0:1]
                            teacher_est_noises = rec_sources_wavs[:, 1:]

                            sisdr = - pairwise_neg_sisdr(
                                teacher_est_active_speakers, gt_speaker_mix).detach().cpu()
                            mix_sisdr = sisdr + pairwise_neg_sisdr(
                                input_mix, gt_speaker_mix).detach().cpu()
                            res_dic[val_d_name]['sisdr']['acc'] += sisdr.tolist()
                            res_dic[val_d_name]['sisdri']['acc'] += mix_sisdr.tolist()

            val_step += 1
        res_dic = cometml_logger.report_losses_mean_and_std(
            res_dic, experiment, tr_step, val_step)

        for d_name in res_dic:
            for loss_name in res_dic[d_name]:
                res_dic[d_name][loss_name]['acc'] = []
        pprint(res_dic)
        checkpoint_storage_path = '../save_checkpoints'
        if hparams["save_models_every"] > 0:
            if tr_step % hparams["save_models_every"] == 0:
                torch.save(
                    student.module.cpu().state_dict(),
                    os.path.join(
                        checkpoint_storage_path,
                        f"hainangibbonData_mcremixit_student_{student_order}_epoch_{student_step}_global_{tr_step}_suodteacher.pt"),
                )
                # Restore the model in the proper device.
                student = student.cuda()

finally:
    poolval.close()
    poolval.join()
    torch.cuda.empty_cache()