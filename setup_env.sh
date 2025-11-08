#!/bin/bash
# SO-101 MuJoCo + LeRobot Environment Setup Script
# Run: bash setup_env.sh

set -e  # Exit on error

echo ""
echo "======================================================================"
echo "SO-101 MUJOCO + LEROBOT ENVIRONMENT SETUP"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Check if venv exists
echo "📦 Step 1: Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment already exists"
fi

# Step 2: Activate venv
echo ""
echo "🔌 Step 2: Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"

# Step 3: Check for .env file
echo ""
echo "🔐 Step 3: Checking environment file..."
if [ ! -f ".env" ]; then
    echo "   Creating .env file..."
    cat > .env << 'EOF'
# HuggingFace Credentials
HF_TOKEN=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
HF_USER=davidlinjiahao

# Dataset configuration
DATASET_NAME=davidlinjiahao/so101_mujoco_pick_place

# MuJoCo rendering (macOS)
MUJOCO_GL=glfw
EOF
    echo "   ✅ .env file created"
else
    echo "   ✅ .env file already exists"
fi

# Step 4: Load environment variables
echo ""
echo "📋 Step 4: Loading environment variables..."
export $(grep -v '^#' .env | xargs)
echo "   ✅ Environment variables loaded"
echo "      HF_USER: $HF_USER"
echo "      DATASET_NAME: $DATASET_NAME"

# Step 5: Upgrade pip
echo ""
echo "⬆️  Step 5: Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✅ Pip upgraded"

# Step 6: Install requirements
echo ""
echo "📦 Step 6: Installing project dependencies..."
cd so101_sim
pip install -r requirements.txt --quiet
echo "   ✅ Project dependencies installed"

# Step 7: Install LeRobot
echo ""
echo "🤖 Step 7: Installing LeRobot..."
pip install lerobot --quiet
echo "   ✅ LeRobot installed"

# Step 8: Verify installations
echo ""
echo "🔍 Step 8: Verifying installations..."

# Check LeRobot
if python3 -c "import lerobot" 2>/dev/null; then
    LEROBOT_VERSION=$(python3 -c "import lerobot; print(lerobot.__version__)")
    echo "   ✅ LeRobot: $LEROBOT_VERSION"
else
    echo "   ❌ LeRobot: Not installed"
    exit 1
fi

# Check MuJoCo
if python3 -c "import mujoco" 2>/dev/null; then
    MUJOCO_VERSION=$(python3 -c "import mujoco; print(mujoco.__version__)")
    echo "   ✅ MuJoCo: $MUJOCO_VERSION"
else
    echo "   ❌ MuJoCo: Not installed"
    exit 1
fi

# Check PyTorch
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "   ✅ PyTorch: $TORCH_VERSION"
else
    echo "   ❌ PyTorch: Not installed"
    exit 1
fi

# Check OpenCV
if python3 -c "import cv2" 2>/dev/null; then
    CV2_VERSION=$(python3 -c "import cv2; print(cv2.__version__)")
    echo "   ✅ OpenCV: $CV2_VERSION"
else
    echo "   ⚠️  OpenCV: Not installed (optional for video recording)"
fi

# Step 9: Login to HuggingFace
echo ""
echo "🔑 Step 9: Logging into HuggingFace..."
if huggingface-cli whoami &>/dev/null; then
    HF_USERNAME=$(huggingface-cli whoami | head -n 1)
    echo "   ✅ Already logged in as: $HF_USERNAME"
else
    echo "   Logging in with token..."
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
    echo "   ✅ Logged in as: $HF_USER"
fi

# Step 10: Check leader arm port
echo ""
echo "🦾 Step 10: Checking for leader arm..."
if ls /dev/tty.usb* 1> /dev/null 2>&1; then
    echo "   ✅ USB devices found:"
    ls -1 /dev/tty.usb* | while read port; do
        echo "      - $port"
    done
else
    echo "   ⚠️  No USB devices found. Connect your leader arm."
fi

# Summary
echo ""
echo "======================================================================"
echo "SETUP COMPLETE! 🎉"
echo "======================================================================"
echo ""
echo "📊 Installation Summary:"
echo "   Virtual environment: $(which python3)"
echo "   HF Username: $HF_USER"
echo "   Dataset: $DATASET_NAME"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. To start recording:"
echo "   python3 scripts/record_lerobot.py \\"
echo "       --leader-port /dev/tty.usbmodem58760431551 \\"
echo "       --leader-id my_leader_arm \\"
echo "       --repo-id \$DATASET_NAME \\"
echo "       --episodes 50 \\"
echo "       --render"
echo ""
echo "2. To train a policy:"
echo "   lerobot-train \\"
echo "       --dataset.repo_id=\$DATASET_NAME \\"
echo "       --policy.type=act \\"
echo "       --policy.device=mps"
echo ""
echo "3. To evaluate:"
echo "   python3 scripts/deploy.py \\"
echo "       --policy-path outputs/train/act_sim/checkpoints/last/pretrained_model \\"
echo "       --episodes 20 --render"
echo ""
echo "💡 Pro Tip: Add this alias to your ~/.zshrc:"
echo "   alias mujoco='cd $SCRIPT_DIR/so101_sim && source ../venv/bin/activate && export \$(grep -v \"^#\" ../.env | xargs)'"
echo ""
echo "======================================================================"

