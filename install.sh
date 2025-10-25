#!/bin/bash
# Enhanced installation script for remote Docker environments with VS Code
# Run with: source install.sh


# Logging function
log() {
    echo "[INFO] $1"
}

warn() {
    echo "[WARN] $1"
}

error() {
    echo "[ERROR] $1"
}

success() {
    echo "[SUCCESS] $1"
}

# Setup git configuration
setup_git() {
    log "Setting up git configuration..."
    
    git config --global user.email "user@example.com"
    git config --global user.name "User"
    success "Git configured: User <user@example.com>"
}

# Install uv package manager
install_uv() {
    log "Installing uv package manager..."
    
    # Check if uv is already installed
    if command -v uv >/dev/null 2>&1; then
        success "uv is already installed: $(uv --version)"
        return
    fi
    
    # Download and install uv
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error "Failed to install uv."
        exit 1
    fi
    
    # Add to PATH for current session
    #export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! command -v uv >/dev/null 2>&1; then
        error "Failed to install uv."
        exit 1
    fi
    
    success "uv installed successfully: $(uv --version)"
}

# Create and setup virtual environment
setup_venv() {
    log "Creating virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d ".venv" ]; then
        warn "Removing existing virtual environment..."
        rm -rf .venv
    fi
    
    # Create new virtual environment
    if ! uv venv; then
        error "Failed to create virtual environment"
        exit 1
    fi
    
    success "Virtual environment created"
    
    # Install project dependencies
    log "Installing project dependencies..."
    if ! uv pip install -e . --python .venv/bin/python --prerelease=allow; then
        error "Failed to install project dependencies"
        exit 1
    fi
    
    success "Project dependencies installed"
}

# Find all YAML files in the project
find_yaml_files() {
    local yaml_files=()
    
    # Search for YAML files in common config directories
    for config_dir in train_configs test_configs configs; do
        if [ -d "$config_dir" ]; then
            for file in "$config_dir"/*.yaml "$config_dir"/*.yml; do
                if [ -f "$file" ]; then
                    yaml_files+=("$file")
                fi
            done
        fi
    done
    
    # Also search for YAML files in the root directory
    for file in ./*.yaml ./*.yml; do
        if [ -f "$file" ]; then
            yaml_files+=("$file")
        fi
    done
    
    printf '%s\n' "${yaml_files[@]}" | sort
}

# Determine script type based on YAML file location
determine_script_type() {
    local yaml_file="$1"
    if [[ "$yaml_file" == *"test_configs"* ]]; then
        echo "scripts/test.py"
    elif [[ "$yaml_file" == *"train_configs"* ]]; then
        echo "scripts/train.py"
    else
        echo "scripts/train.py"  # Default to train.py
    fi
}

# Generate display name from filename
generate_display_name() {
    local filename="$1"
    local basename=$(basename "$filename" .yaml)
    basename=$(basename "$basename" .yml)
    
    # Add prefix based on directory
    if [[ "$filename" == *"train_configs"* ]]; then
        basename="Train $basename"
    elif [[ "$filename" == *"test_configs"* ]]; then
        basename="Test $basename"
    fi
    
    # Replace underscores with spaces and capitalize
    echo "$basename" | sed 's/_/ /g' | sed 's/\b\w/\U&/g'
}

# Setup VS Code configuration for remote development
setup_vscode() {
    log "Setting up VS Code configuration for remote development..."
    
    mkdir -p .vscode
    
    # Find all YAML files
    local yaml_files
    mapfile -t yaml_files < <(find_yaml_files)
    
    if [ ${#yaml_files[@]} -eq 0 ]; then
        warn "No YAML files found for VS Code configuration"
        return
    fi
    
    log "Found ${#yaml_files[@]} YAML files: ${yaml_files[*]}"
    
    # Start building the JSON configuration
    cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
EOF

    # Generate configurations for each YAML file
    local first=true
    for yaml_file in "${yaml_files[@]}"; do
        local script_path=$(determine_script_type "$yaml_file")
        local display_name=$(generate_display_name "$yaml_file")
        
        if [ "$first" = false ]; then
            echo "," >> .vscode/launch.json
        fi
        first=false
        
        # Debug configuration
        cat >> .vscode/launch.json << EOF
        {
            "name": "Run $display_name (Debug)",
            "type": "debugpy",
            "request": "launch",
            "program": "\${workspaceFolder}/$script_path",
            "console": "integratedTerminal",
            "args": ["$yaml_file"],
            "python": "\${workspaceFolder}/.venv/bin/python",
            "cwd": "\${workspaceFolder}",
            "env": {
                "PYTHONPATH": "\${workspaceFolder}/train_src"
            }
        },
        {
            "name": "Run $display_name",
            "type": "python",
            "request": "launch",
            "program": "\${workspaceFolder}/$script_path",
            "console": "integratedTerminal",
            "args": ["$yaml_file"],
            "python": "\${workspaceFolder}/.venv/bin/python",
            "cwd": "\${workspaceFolder}",
            "env": {
                "PYTHONPATH": "\${workspaceFolder}/train_src"
            },
            "noDebug": true
        }
EOF
    done
    
    # Add test run profiles for test_configs
    local test_configs=()
    for yaml_file in "${yaml_files[@]}"; do
        if [[ "$yaml_file" == *"test_configs"* ]]; then
            test_configs+=("$yaml_file")
        fi
    done
    
    # Generate test profiles for each test config
    for test_config in "${test_configs[@]}"; do
        local test_display_name=$(generate_display_name "$test_config")
        echo "," >> .vscode/launch.json
        
        cat >> .vscode/launch.json << EOF
        {
            "name": "Test $test_display_name",
            "type": "python",
            "request": "launch",
            "program": "\${workspaceFolder}/scripts/test.py",
            "console": "integratedTerminal",
            "args": ["$test_config"],
            "python": "\${workspaceFolder}/.venv/bin/python",
            "cwd": "\${workspaceFolder}",
            "env": {
                "PYTHONPATH": "\${workspaceFolder}/test_src"
            },
            "noDebug": true
        }
EOF
    done
    
    # Close the JSON
    cat >> .vscode/launch.json << 'EOF'
    ]
}
EOF

    success "VS Code configuration created for ${#yaml_files[@]} YAML"
}


# Main installation process
main() {
    echo "Starting Installation"
    echo "=================================="
    
    # Run installation steps with virtual environment
    setup_git
    install_uv
    setup_venv
    setup_vscode
}

# Run main function
main "$@"
