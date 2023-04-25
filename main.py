import sys
from dataprep import DataPrep


def main(argv):
    data_prep = int(argv[1])
    train_mode = int(argv[2])
    if data_prep == 1:
        ob = DataPrep(
            "dataset", "Pre_processed_data/data.csv")
        ob.prepare_data()
        ob.transform_data()
    
    if train_mode == 1:
        pass


if __name__ == "__main__":
    main(sys.argv)
