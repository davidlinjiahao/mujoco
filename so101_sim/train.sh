#!/bin/bash
# Unified training script with sleep prevention and live progress monitoring
# Usage: ./train.sh [--monitor-only]

set -e

DATASET_REPO="davidlinjiahao/lerobot_batch_001"
OUTPUT_DIR="outputs/train/so101_overnight"
JOB_NAME="so101_overnight_200k"
POLICY_REPO="davidlinjiahao/so101_overnight"
LOG_FILE="training_overnight.log"
TOTAL_STEPS=200000

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're just monitoring an existing training run
MONITOR_ONLY=false
if [ "$1" = "--monitor-only" ]; then
    MONITOR_ONLY=true
fi

# Function to display progress
show_progress() {
    while true; do
        if [ ! -f "$LOG_FILE" ]; then
            clear
            echo "🤖 SO-101 Training Monitor"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "⏳ Waiting for training to start..."
            echo "   Log file not found: $LOG_FILE"
            sleep 10
            continue
        fi
        
        # Get latest training step
        LATEST_LINE=$(grep "step:" "$LOG_FILE" 2>/dev/null | tail -1)
        
        if [ -z "$LATEST_LINE" ]; then
            clear
            echo "🤖 SO-101 Training Monitor"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "⏳ Training initializing... (no steps yet)"
            echo ""
            echo "Recent log:"
            tail -5 "$LOG_FILE" 2>/dev/null || echo "  (empty)"
            sleep 10
            continue
        fi
        
        # Extract info
        STEP_RAW=$(echo "$LATEST_LINE" | grep -o 'step:[0-9K]*' | cut -d: -f2)
        LOSS=$(echo "$LATEST_LINE" | grep -o 'loss:[0-9.]*' | cut -d: -f2)
        UPDT=$(echo "$LATEST_LINE" | grep -o 'updt_s:[0-9.]*' | cut -d: -f2)
        
        # Convert K notation to actual number
        if [[ "$STEP_RAW" == *K ]]; then
            STEP_NUM=$(echo "$STEP_RAW" | sed 's/K//g')
            STEP_NUM=$((STEP_NUM * 1000))
        else
            STEP_NUM=$STEP_RAW
        fi
        
        # Calculate progress percentage
        PROGRESS=$(awk "BEGIN {printf \"%.2f\", ($STEP_NUM / $TOTAL_STEPS) * 100}")
        
        # Build progress bar (50 characters wide)
        FILLED=$(awk "BEGIN {printf \"%.0f\", ($STEP_NUM / $TOTAL_STEPS) * 50}")
        if [ "$FILLED" -gt 0 ]; then
            BAR=$(printf '█%.0s' $(seq 1 $FILLED))
        else
            BAR=""
        fi
        EMPTY_COUNT=$((50 - FILLED))
        if [ "$EMPTY_COUNT" -gt 0 ]; then
            EMPTY=$(printf '░%.0s' $(seq 1 $EMPTY_COUNT))
        else
            EMPTY=""
        fi
        
        # Calculate ETA
        if [ "$STEP_NUM" -gt 0 ] && [ ! -z "$UPDT" ]; then
            STEPS_PER_SEC=$(awk "BEGIN {printf \"%.3f\", 1/$UPDT}")
            STEPS_PER_MIN=$(awk "BEGIN {printf \"%.1f\", $STEPS_PER_SEC * 60}")
            
            # ETA to 100K
            TO_100K=$((100000 - STEP_NUM))
            if [ "$TO_100K" -gt 0 ]; then
                ETA_100K_SEC=$(awk "BEGIN {printf \"%.0f\", $TO_100K / $STEPS_PER_SEC}")
                ETA_100K_HOURS=$(awk "BEGIN {printf \"%.1f\", $ETA_100K_SEC / 3600}")
            else
                ETA_100K_HOURS="✅"
            fi
            
            # ETA to completion
            REMAINING=$((TOTAL_STEPS - STEP_NUM))
            ETA_SEC=$(awk "BEGIN {printf \"%.0f\", $REMAINING / $STEPS_PER_SEC}")
            ETA_HOURS=$(awk "BEGIN {printf \"%.1f\", $ETA_SEC / 3600}")
        else
            STEPS_PER_SEC="--"
            STEPS_PER_MIN="--"
            ETA_100K_HOURS="--"
            ETA_HOURS="--"
        fi
        
        # Mark milestones
        MARK_10K=" "
        MARK_50K=" "
        MARK_100K=" "
        MARK_150K=" "
        MARK_200K=" "
        
        [ "$STEP_NUM" -ge 10000 ] && MARK_10K="✓"
        [ "$STEP_NUM" -ge 50000 ] && MARK_50K="✓"
        [ "$STEP_NUM" -ge 100000 ] && MARK_100K="✓"
        [ "$STEP_NUM" -ge 150000 ] && MARK_150K="✓"
        [ "$STEP_NUM" -ge 200000 ] && MARK_200K="✓"
        
        # Clear screen and display
        clear
        echo "🤖 SO-101 Live Training Monitor"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Progress: $STEP_RAW / 200K ($PROGRESS%)"
        echo "[$BAR$EMPTY]"
        echo ""
        printf "Loss: %s | Speed: %s steps/min (%s steps/s)\n" "$LOSS" "$STEPS_PER_MIN" "$STEPS_PER_SEC"
        echo ""
        echo "⏱️  Time Estimates:"
        echo "  🎯 To 100K (evaluation): ~$ETA_100K_HOURS hours"
        echo "  ⏰ To 200K (complete):   ~$ETA_HOURS hours"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Milestones:"
        echo "  [$MARK_10K]  10K   - Basic learning"
        echo "  [$MARK_50K]  50K   - Decent performance"
        echo "  [$MARK_100K]  100K  - Ready to evaluate!"
        echo "  [$MARK_150K]  150K  - Refined behavior"
        echo "  [$MARK_200K]  200K  - Complete ✨"
        echo ""
        
        # Show latest checkpoints
        echo "Latest checkpoints:"
        if ls -1 "$OUTPUT_DIR/checkpoints/" 2>/dev/null | grep -E "^[0-9]" | tail -5 >/dev/null 2>&1; then
            ls -1 "$OUTPUT_DIR/checkpoints/" 2>/dev/null | grep -E "^[0-9]" | tail -5 | while read ckpt; do
                echo "  ✓ $ckpt"
            done
        else
            echo "  (none yet)"
        fi
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "💡 Evaluate: mjpython scripts/deploy.py --policy-path $OUTPUT_DIR/checkpoints/100000/pretrained_model --episodes 3 --render --hz 10"
        echo ""
        echo "Press Ctrl+C to stop monitoring (training continues in background)"
        printf "Last update: %s\n" "$(date '+%H:%M:%S')"
        
        sleep 10
    done
}

# Main execution
if [ "$MONITOR_ONLY" = true ]; then
    echo "📊 Monitoring existing training run..."
    echo ""
    show_progress
else
    echo "🚀 Starting SO-101 Training"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Configuration:"
    echo "  📂 Dataset: $DATASET_REPO"
    echo "  🎯 Total Steps: $TOTAL_STEPS"
    echo "  💾 Output: $OUTPUT_DIR"
    echo "  🔄 Checkpoints: Every 1,000 steps"
    echo "  💤 Sleep Prevention: Enabled (caffeinate)"
    echo ""
    echo "Starting training in background..."
    echo ""
    
    # Check if resuming from existing checkpoint
    CONFIG_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
    
    if [ -f "$CONFIG_PATH" ]; then
        echo "🔄 Resuming from existing checkpoint..."
        echo "   Config: $CONFIG_PATH"
        echo ""
        
        # Start training with caffeinate in background (resume mode)
        caffeinate -disu lerobot-train \
            --config_path="$CONFIG_PATH" \
            --resume=true \
            2>&1 | tee "$LOG_FILE" &
    else
        echo "🆕 Starting fresh training..."
        echo ""
        
        # Start training with caffeinate in background (new training)
        caffeinate -disu lerobot-train \
            --dataset.repo_id="$DATASET_REPO" \
            --dataset.video_backend=pyav \
            --policy.type=act \
            --output_dir="$OUTPUT_DIR" \
            --job_name="$JOB_NAME" \
            --policy.repo_id="$POLICY_REPO" \
            --policy.device=mps \
            --steps=$TOTAL_STEPS \
            --eval_freq=50000 \
            --log_freq=500 \
            --save_checkpoint=true \
            --save_freq=1000 \
            --wandb.enable=false \
            2>&1 | tee "$LOG_FILE" &
    fi
    
    TRAIN_PID=$!
    
    echo "✅ Training started (PID: $TRAIN_PID)"
    echo "   Log: $LOG_FILE"
    echo ""
    echo "Waiting 3 seconds for training to initialize..."
    sleep 3
    
    # Start monitoring
    show_progress
fi

