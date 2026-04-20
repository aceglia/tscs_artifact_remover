from artifact_remover.optimizer.optimizer import Optimizer
from artifact_remover.automatic_remover import ArtefactRemover

if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r"D:\Downloads\T1_008_arm_sa_002.mat"
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\test_data\007Loc_sa_20_Avec000.txt"
    # path_file = r"D:\Documents\Programmation\tscs_artifact_remover\007Loc_sa_20_Avec000_processed.csv"
    # path_file = r"test001.txt"
    process_window = 4000
    h_delay = 1
    h_size = int((process_window / 8) / h_delay)
    print("Hankel matrix size is: ", "(", h_size, ",", process_window - (h_size - 1) * h_delay, ")")
    notch_filter = True
    artefact_remover = ArtefactRemover(data=path_file, signal_filter=True, center=True, cutoff=[10, 500])
    optimizer = Optimizer(artefact_remover, n_processes=1)
    optimizer.optimize(channels=[2, 3], batch=list(range(3, 6)), process_window=process_window) #optimizer.artifact_remover.data_loader.init_data.shape[0]