1. Prepare the hainangibbon dataset

2. get MC-RemixIT code

git clone https://github.com/etzinis/unsup_speech_enh_adaptation.git

cd unsup_speech_enh_adaptation

3. Repo and paths configurations

conda create -n callseparation

conda activate callseparation

pip -r install --user -r requirements.txt

vim __config__.py

GIBBON_ROOT_PATH = '{inset_path_to_hainangibboncall}'

TEACHER_GIBBONSOUND_ROOT_PATH = '{inset_path_to_teachergibboncall}'

API_KEY = 'your_comet_ml_key'

4. train the supervised teacher:

python -Wignore baseline/run_sup_ood_pretrain.py \
--train teacherGibbonsound \
--val teacherGibbonsound \
--test teacherGibbonsound \
-fs 16000 \
--enc_kernel_size 81 \
--num_blocks 8 \
--out_channels 256 \
--divide_lr_by 3. \
--upsampling_depth 7 \
--patience 15  -tags supervised_ood_teacher \
--n_epochs 81 \
--project_name teacherGibbonsound_baseline_v3 \
--clip_grad_norm 5.0 \
--save_models_every 20 \
--audio_timelength 10.0 \
--p_single_speaker 0.5 \
--min_or_max min --max_num_sources 2 \
--checkpoint_storage_path ../pretrained_checkpoints/teacherGibbonsound \
--log_audio \
--apply_mixture_consistency \
--n_jobs 8 -cad 0 -bs 16 \

5. train the student:

python -Wignore baseline/run_remixit_revise.py \
--train hainangibbon \
--val hainangibbon \
--test hainangibbon \
-fs 16000 \
--audio_timelength 10.0 \
--enc_kernel_size 81 \
--num_blocks 8 \
--out_channels 256 \
--divide_lr_by 3. \
--student_depth_growth 1 \
--n_epochs_teacher_update 1 \
--teacher_momentum 0.99 \
--upsampling_depth 7 \
--patience 10 \
--learning_rate 0.0003 -tags remixit student allData \
--n_epochs 60 --project_name uchime_baseline_v3 \
--clip_grad_norm 5.0 \
--min_or_max min \
--max_num_sources 2 \
--save_models_every 20 \
--initialize_student_from_checkpoint \
--checkpoint_storage_path ../unsup_speech_enh_adaptation-main/save_checkpoints \
--warmup_checkpoint ../pretrained_checkpoints/teacherGibbonsound/supervised_ood_teacher/sup_teacher_epoch_80.pt \
--log_audio --apply_mixture_consistency \
--n_jobs 8 -cad 0 -bs 16 \

