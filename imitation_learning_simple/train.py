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
    parser.add_argument("--seed", type=int, default=319029, help="Seed")

    # Train model/dataset settings
    parser.add_argument("--visual_extractor", type=str, default="frontnet", choices=["frontnet", "mobilenet"], help="Type of model to use")
    parser.add_argument("--input_type", type=str, default="RGB", choices=["RGB", "2CH"], help="Type of input to use: RGB is default image, 2CH uses pencil filter + depthmap")
    parser.add_argument("--label_type", type=str, default="commands", choices=["commands", "d_commands", "setpoints", "d_setpoints"], help="Type of labels to use: commands or setpoints ('d_' means using its derivative)")
    parser.add_argument("--stats_file_name", type=str, default="stats.json", help="Name of the file containing the statistics (mean, std) of each column in dataset")
    parser.add_argument("--force_data_stats", action="store_true", help="Recompute the dataset stats even if config is found")
    parser.add_argument("--avoid_input_normalization", action="store_true", help="Stop input normalization")
    parser.add_argument("--avoid_label_normalization", action="store_true", help="Stop label normalization")

    # Train procedure settings
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the batch for training")
    parser.add_argument("--hist_size", type=int, default=0, help="Number of timesteps to stack when getting data from dataset")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Starting learning rate")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.3, help="Muliply the learning rate by the gamma factor every {args.lr_cheduler_step} steps")
    parser.add_argument("--lr_scheduler_step", type=int, default=3, help="Every how many epochs apply the gamma to the learning rate")
    parser.add_argument("--patience_epochs", type=int, default=8, help="After how many epochs of not improving the validation score stop the training")

    parser.add_argument("--disable_cuda", action="store_true", help="Even if cuda is available, dont use it")

    # Wandb arguments
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"], help="Wandb mode")
    parser.add_argument("--wandb_group", type=str, default="auto", help="Wandb group name, if auto the trainer will give one")
    

    args = parser.parse_args()
    main(args)