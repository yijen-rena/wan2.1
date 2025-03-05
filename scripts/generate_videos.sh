#!/bin/bash

# Script to generate videos from prompts in prompts.json

OUTPUT_DIR="./outputs"

mkdir -p "$OUTPUT_DIR"

process_prompt() {
    local prompt="$1"
    local index="$2"

    echo "Generating video for prompt #${index}:"
    echo "$prompt"
    
    for seed in 42 469 709 309 174 221; do # seed chosen according to https://arxiv.org/pdf/2405.14828
        start_time=$(date +%s)
        
        python generate.py \
            --task "t2v-1.3B" \
            --size "832*480" \
            --ckpt_dir "Wan2.1-T2V-1.3B" \
            --sample_shift 8 \
            --sample_guide_scale 6 \
            --prompt "$prompt" \
            --base_seed $seed

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "Completed video #${index} in ${duration} seconds"
        echo "------------------------"
    
    done
}

echo "Extracting prompts from prompts.json..."
PROMPTS=$(jq -r '.prompts[]' prompts.json)

INDEX=1
echo "$PROMPTS" | while IFS= read -r prompt; do
    process_prompt "$prompt" "$INDEX"
    ((INDEX++))
done

echo "All videos generated successfully!"
echo "Videos saved to: $OUTPUT_DIR"
