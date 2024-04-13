from dataset.cifar_pairs import CifarPairs
from torch.utils.data import DataLoader
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', 
                        type=str, 
                        default="data/distillation_dataset_h5/cifar.hdf5", 
                        help='Path of the h5 dataset file'
                        )
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=64, 
                        help='Batch size used in training process'
                        )
    args = parser.parse_args()
    print(args.data_path)


    training_dataset = CifarPairs(args.data_path)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    for data in training_dataloader:
        print(data["instance_id"]) # Example usage
        # Available keys: (instance_id, image, latent, class_id, seed)
        break # For testing purpose

if __name__ == "__main__":
    main()





