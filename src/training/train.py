import argparse
from pathlib import Path
from trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Train cervical lesion classifier')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name of the experiment')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(
        config_path=args.config_path,
        experiment_name=args.experiment_name
    )
    
    trainer.train()

if __name__ == '__main__':
    main()
