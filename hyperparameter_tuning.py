import sys
import argparse
from src.hyperparameter_tuning import main as hyperopt_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for rainfall prediction')
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'lightgbm', 'catboost'], 
                        help='Model type to optimize')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--classification', action='store_true', help='Treat as classification problem')
    
    args = parser.parse_args()
    
    # Pass the parsed arguments to the main function
    sys.argv = [sys.argv[0]]  # Clear existing args
    if args.model:
        sys.argv.extend(['--model', args.model])
    if args.trials:
        sys.argv.extend(['--trials', str(args.trials)])
    if args.classification:
        sys.argv.append('--classification')
    
    # Call the main function from hyperparameter_tuning.py
    hyperopt_main()