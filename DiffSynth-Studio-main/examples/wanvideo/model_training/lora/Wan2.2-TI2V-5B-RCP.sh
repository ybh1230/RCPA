export DIFFSYNTH_SKIP_DOWNLOAD=true

accelerate launch examples/wanvideo/model_training/train.py \
  --task "mvla_sft" \
  --dataset_base_path "./data/InternImgs" \
  --dataset_metadata_path "./data/InternImgs/hr_data_final_clean.csv" \
  --height 1536 \
  --width 2752 \
  --num_frames 1 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_rcp_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --save_steps 1000 \
  --extra_inputs "input_image" \
  --mvla_num_views 3 \
  --mvla_down_factors "2,4" \
  --mvla_reconstruction_weight 1.0 \
  --mvla_agreement_weight 0.15 \
  --mvla_high_freq_weight 0.25
