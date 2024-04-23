import argparse
from Trainer import Trainer

def main(args):
    t = Trainer(args)
    if not args.test_only:
        t.train()
    t.test()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### Training arguments
    parser.add_argument("--test_only", action="store_true", help="Flag to avoid training")
    parser.add_argument("--load_model", type=str, default="", help="Weights model name (stored in 'weights/' directory)")

    parser.add_argument("--batch_size", type=int, default=16, help="Size of the batch for training")
    parser.add_argument("--hist_size", type=int, default=30, help="Number of timesteps to stack when getting data from dataset (only applicable if `tcn_*` model is used )")
    parser.add_argument("--model", type=str, default="tcn", choices=["tcn", "resnet"], help="Type of model to use: resnet (no time information), tcn (time information)")

    parser.add_argument("--stats_file_name", type=str, default="stats.json", help="Name of the file containing the statistics (mean, std) of each column in dataset")
    parser.add_argument("--force_data_stats", action="store_true", help="Recompute the dataset stats even if config is found")
    parser.add_argument("--avoid_input_normalization", action="store_true", help="Stop input normalization")
    parser.add_argument("--avoid_output_normalization", action="store_true", help="Stop output normalization")

    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Starting learning rate")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.3, help="Muliply the learning rate by the gamma factor every \{args.lr_cheduler_step\} steps")
    parser.add_argument("--lr_scheduler_step", type=int, default=2, help="Every how many epochs apply the gamma to the learning rate")
    parser.add_argument("--patience_epochs", type=int, default=4, help="After how many epochs of not improving the validation score stop the training")

    parser.add_argument("--disable_cuda", action="store_true", help="Even if cuda is available, dont use it")

    # Wandb arguments
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="Wandb mode")
    # parser.add_argument("--wandb_group", type=str, help="Wandb mode")
    

    args = parser.parse_args()
    main(args)