# Borrowed and modified from https://github.com/ymcui/Chinese-LLaMA-Alpaca-3

lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,k_proj"

modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model="path/to/original-hf-model"
tokenizer_name_or_path=${pretrained_model}
dataset_dir="../training_set/sequential_planning"
per_device_train_batch_size=5
per_device_eval_batch_size=5
gradient_accumulation_steps=5
max_seq_length=1600
output_dir="../output_sequential_planning"
validation_file="../eval_set/sequential_planning.json"

torchrun --nnodes 1 --nproc_per_node 6 step4_sft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --low_cpu_mem_usage \
    --do_eval \
    --seed 5 \
    --bf16 \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype bfloat16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False
