#!/bin/bash

# LHM Standard Installation Script for Ubuntu 22.04 with CUDA 12.1
# Based on the official LHM repository installation guide

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Running as root is acceptable for temporary instances
if [[ $EUID -eq 0 ]]; then
   print_warning "Running as root - acceptable for temporary instances"
fi

# Check Ubuntu version
if ! grep -q "22.04" /etc/os-release; then
    print_warning "This script is designed for Ubuntu 22.04. Your version might work but is untested."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA installation
print_status "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null || ! nvcc --version | grep -q "release 12.1"; then
    print_error "CUDA 12.1 not found. Please install NVIDIA CUDA 12.1 first."
    print_status "You can install it from: https://developer.nvidia.com/cuda-12-1-0-download-archive"
    exit 1
fi

print_success "CUDA 12.1 detected"

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        print_status "You can install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
        print_status "Quick install command:"
        print_status "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        print_status "bash Miniconda3-latest-Linux-x86_64.sh"
        exit 1
    fi
    print_success "Conda found: $(conda --version)"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    apt update
    apt install -y \
        git \
        wget \
        curl \
        unzip \
        build-essential \
        cmake \
        pkg-config \
        libgl1-mesa-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgcc-s1 \
        ninja-build \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev
    print_success "System dependencies installed"
}

# Create and setup conda environment
setup_conda_env() {
    print_status "Setting up conda environment..."
    
    # Create conda environment with Python 3.10
    ENV_NAME="lhm"
    
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Conda environment '$ENV_NAME' already exists. Removing it..."
        conda env remove -n $ENV_NAME -y
    fi
    
    conda create -n $ENV_NAME python=3.10 -y
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    # Install conda-forge packages
    conda install -c conda-forge -y \
        pip \
        cmake \
        ninja \
        pkg-config \
        ffmpeg
        
    print_success "Conda environment '$ENV_NAME' created and activated"
}

# Clone repository
clone_repo() {
    print_status "Cloning LHM repository..."
    
    if [ ! -d "LHM" ]; then
        git clone https://github.com/aigc3d/LHM.git
        print_success "LHM repository cloned"
    else
        print_warning "LHM directory already exists, skipping clone"
    fi
    
    cd LHM
}

# Install PyTorch and dependencies
install_pytorch() {
    print_status "Installing PyTorch with CUDA 12.1 support via conda..."
    
    # Install PyTorch from conda-forge and pytorch channels
    conda install -c pytorch -c nvidia -y \
        pytorch==2.3.0 \
        torchvision==0.18.0 \
        torchaudio==2.3.0 \
        pytorch-cuda=12.1
    
    print_success "PyTorch installed"
}

# Install basic requirements
install_requirements() {
    print_status "Installing requirements from requirements.txt..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Requirements installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Install additional dependencies
install_additional_deps() {
    print_status "Installing additional dependencies..."
    
    # Install rembg for background removal
    pip install rembg
    
    # Install xformers
    pip install xformers
    
    # Install additional packages commonly needed
    pip install opencv-python
    pip install pillow
    pip install numpy
    pip install scipy
    pip install trimesh
    
    # Install specific versions to align with project requirements
    print_status "Installing specific package versions for compatibility..."
    pip install pydantic==2.5.0
    pip install gradio==4.36.0
    
    print_success "Additional dependencies installed with specified versions"
}

# Download pretrained models
download_models() {
    print_status "Downloading pretrained models..."
    
    # Create pretrained_models directory
    mkdir -p pretrained_models
    
    # Download LHM prior model weights
    if [ ! -f "LHM_prior_model.tar" ]; then
        print_status "Downloading LHM prior model weights..."
        wget -O LHM_prior_model.tar https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar
        tar -xvf LHM_prior_model.tar
        print_success "Prior model weights downloaded and extracted"
    else
        print_warning "Prior model weights already exist, skipping download"
    fi
    
    # Download motion video examples
    if [ ! -f "motion_video.tar" ]; then
        print_status "Downloading motion video examples..."
        mkdir -p train_data
        wget -O motion_video.tar https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/motion_video.tar
        tar -xvf motion_video.tar -C train_data/
        print_success "Motion video examples downloaded and extracted"
    else
        print_warning "Motion video examples already exist, skipping download"
    fi
    
    # Download additional model weights for motion processing
    print_status "Downloading additional model weights for motion processing..."
    mkdir -p pretrained_models/human_model_files/pose_estimate
    
    if [ ! -f "pretrained_models/human_model_files/pose_estimate/yolov8x.pt" ]; then
        wget -P pretrained_models/human_model_files/pose_estimate https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/yolov8x.pt
    fi
    
    if [ ! -f "pretrained_models/human_model_files/pose_estimate/vitpose-h-wholebody.pth" ]; then
        wget -P pretrained_models/human_model_files/pose_estimate https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/vitpose-h-wholebody.pth
    fi
    
    # Download arcface model for metrics
    if [ ! -f "pretrained_models/arcface_resnet18.pth" ]; then
        wget -P pretrained_models https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/arcface_resnet18.pth
    fi
    
    print_success "Additional model weights downloaded"
}

# Install pose estimation dependencies
install_pose_estimation() {
    print_status "Installing pose estimation dependencies..."
    
    cd engine/pose_estimation
    
    # Install mmcv
    pip install mmcv==1.3.9
    
    # Install ViTPose
    if [ -d "third-party/ViTPose" ]; then
        pip install -v -e third-party/ViTPose
        print_success "ViTPose installed"
    else
        print_warning "ViTPose directory not found, skipping"
    fi
    
    # Install ultralytics
    pip install ultralytics
    
    cd ../../  # Back to LHM root
    print_success "Pose estimation dependencies installed"
}

# Download LHM models from HuggingFace
download_lhm_models() {
    print_status "Setting up automatic model download from HuggingFace..."
    
    # Install huggingface_hub
    pip install huggingface_hub
    
    print_status "Models will be downloaded automatically when needed."
    print_status "Available models: LHM-MINI, LHM-500M-HF, LHM-1B-HF"
    print_warning "Models are large (several GB each) and will be downloaded on first use."
}

# Create run scripts
create_run_scripts() {
    print_status "Creating run scripts..."
    
    # Main gradio app script
    cat > run_lhm.sh << 'EOF'
#!/bin/bash

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
conda activate lhm

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run LHM Gradio app
python app.py "$@"
EOF

    # Motion app script (requires more GPU memory)
    cat > run_lhm_motion.sh << 'EOF'
#!/bin/bash

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
conda activate lhm

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run LHM Motion app (requires 24GB+ GPU memory)
python app_motion.py "$@"
EOF

    # Memory-saving motion app script
    cat > run_lhm_motion_ms.sh << 'EOF'
#!/bin/bash

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
conda activate lhm

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run LHM Motion app with memory saving (works on 14GB+ GPU)
python app_motion_ms.py "$@"
EOF

    # Inference script
    cat > run_inference.sh << 'EOF'
#!/bin/bash

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
conda activate lhm

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run inference
# Usage: ./run_inference.sh MODEL_NAME IMAGE_PATH MOTION_SEQ
# Example: ./run_inference.sh LHM-500M-HF ./train_data/example_imgs/ ./train_data/motion_video/mimo1/smplx_params
bash inference.sh "$@"
EOF

    # Video to motion script
    cat > run_video2motion.sh << 'EOF'
#!/bin/bash

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
conda activate lhm

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Convert video to motion
# Usage: ./run_video2motion.sh VIDEO_PATH OUTPUT_PATH [FITTING_STEPS]
# Example: ./run_video2motion.sh ./train_data/demo.mp4 ./train_data/custom_motion

VIDEO_PATH="$1"
OUTPUT_PATH="$2"
FITTING_STEPS="${3:-100}"

if [ -z "$VIDEO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: $0 VIDEO_PATH OUTPUT_PATH [FITTING_STEPS]"
    echo "Example: $0 ./train_data/demo.mp4 ./train_data/custom_motion 100"
    exit 1
fi

python ./engine/pose_estimation/video2motion.py --video_path "$VIDEO_PATH" --output_path "$OUTPUT_PATH" --fitting_steps "$FITTING_STEPS" 0
EOF

    chmod +x run_lhm.sh run_lhm_motion.sh run_lhm_motion_ms.sh run_inference.sh run_video2motion.sh
    print_success "Run scripts created"
}

# Main installation process
main() {
    print_status "Starting LHM installation for Ubuntu 22.04 with CUDA 12.1"
    print_status "This process may take 60-90 minutes depending on your internet connection and hardware"
    
    # Create installation directory in /root
    INSTALL_DIR="/root/lhm_standard"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    print_status "Installation directory: $INSTALL_DIR"
    
    # Run installation steps
    check_conda
    install_system_deps
    setup_conda_env
    clone_repo
    install_pytorch
    install_requirements
    install_additional_deps
    download_models
    install_pose_estimation
    download_lhm_models
    create_run_scripts
    
    print_success "Installation completed successfully!"
    print_status ""
    print_status "🚀 Getting Started:"
    print_status "1. cd $INSTALL_DIR/LHM"
    print_status ""
    print_status "📱 Available Applications:"
    print_status "• Basic Gradio App:           ./run_lhm.sh"
    print_status "• Motion App (24GB+ GPU):     ./run_lhm_motion.sh" 
    print_status "• Motion App (14GB+ GPU):     ./run_lhm_motion_ms.sh"
    print_status ""
    print_status "🎬 Video Processing:"
    print_status "• Convert video to motion:    ./run_video2motion.sh VIDEO_PATH OUTPUT_PATH"
    print_status "• Run inference:              ./run_inference.sh MODEL_NAME IMAGE_PATH MOTION_SEQ"
    print_status ""
    print_status "📖 Examples:"
    print_status "• ./run_video2motion.sh ./train_data/demo.mp4 ./train_data/custom_motion"
    print_status "• ./run_inference.sh LHM-500M-HF ./train_data/example_imgs/ ./train_data/motion_video/mimo1/smplx_params"
    print_status ""
    print_status "🔧 Manual activation:"
    print_status "conda activate lhm"
    print_status ""
    print_status "📚 Available Models (downloaded automatically):"
    print_status "• LHM-MINI (16GB GPU) - Fast inference"
    print_status "• LHM-500M-HF (20GB GPU) - Good quality"  
    print_status "• LHM-1B-HF (24GB+ GPU) - Best quality"
    print_status ""
    print_status "🌐 Access Gradio interface at: http://localhost:7860"
    
    print_warning "📋 GPU Memory Requirements:"
    print_warning "• Basic app: 16GB+ recommended"
    print_warning "• Motion app: 24GB+ required"
    print_warning "• Memory-saving motion app: 14GB+ required"
    
    print_status ""
    print_status "📦 Installed Package Versions:"
    print_status "• pydantic: 2.5.0"
    print_status "• gradio: 4.36.0"
}

# Run main function
main "$@"