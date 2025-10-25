#!/bin/bash
# Test runner script for probability analysis

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
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

# Function to show usage
show_usage() {
    echo "Usage: $0 [PROFILE]"
    echo ""
    echo "Available profiles:"
    echo "  quick        - Quick test with 50 samples (default)"
    echo "  baseline     - Test sequence-aware baseline model"
    echo "  comparison   - Compare base GPT-2 vs sequence-aware model"
    echo "  comprehensive - Run multiple test scenarios"
    echo "  visualization - Focus on generating visualizations"
    echo "  all          - Run all test profiles"
    echo ""
    echo "Examples:"
    echo "  $0 quick"
    echo "  $0 baseline"
    echo "  $0 all"
}

# Function to run a specific test profile
run_test_profile() {
    local profile="$1"
    
    case "$profile" in
        "quick")
            print_info "Running quick test..."
            ./test_profiles/quick_test.sh
            ;;
        "baseline")
            print_info "Running baseline test..."
            ./test_profiles/baseline_test.sh
            ;;
        "comparison")
            print_info "Running comparison test..."
            ./test_profiles/comparison_test.sh
            ;;
        "comprehensive")
            print_info "Running comprehensive test..."
            ./test_profiles/comprehensive_test.sh
            ;;
        "visualization")
            print_info "Running visualization test..."
            ./test_profiles/visualization_test.sh
            ;;
        "all")
            print_info "Running all test profiles..."
            echo ""
            run_test_profile "quick"
            echo ""
            run_test_profile "baseline"
            echo ""
            run_test_profile "comparison"
            echo ""
            run_test_profile "comprehensive"
            echo ""
            run_test_profile "visualization"
            ;;
        *)
            print_error "Unknown profile: $profile"
            show_usage
            exit 1
            ;;
    esac
}

# Main function
main() {
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found. Please run 'source install.sh' first."
        exit 1
    fi
    
    # Check if test profiles exist
    if [ ! -d "test_profiles" ]; then
        print_error "Test profiles not found. Please run 'source install.sh' first."
        exit 1
    fi
    
    # Get profile from command line argument or default to quick
    local profile="${1:-quick}"
    
    print_info "Starting probability analysis tests..."
    print_info "Profile: $profile"
    echo ""
    
    # Run the specified test profile
    run_test_profile "$profile"
    
    print_success "Test execution completed!"
}

# Run main function with all arguments
main "$@"
