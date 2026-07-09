import numpy as np
from star_emg.io_utils import DataLoader


class DataStreamer:
    """
    Helper class to stream data in chunks for real-time processing.
    It can be used in both offline and online modes.
    """

    def __init__(
        self, data: str | np.ndarray = None, offline: bool = True, chunk_size: int = None, **data_loader_kwargs
    ):
        """
        Initialize the DataStreamer.
        Parameters:
        -----------
        data: str or np.ndarray, optional
            The path to the data file or the data array itself.
        offline: bool, optional
            Whether to operate in offline mode (True) or online mode (False).
        chunk_size: int, optional
            The size of the chunks to stream. Required if offline is True.
        data_loader_kwargs: dict, optional
            Additional keyword arguments to pass to the DataLoader.


        Raises:
        ValueError: If offline is True and chunk_size is not provided.
        ValueError: If offline is True and data is not provided.

        Returns:
        -------
        None

        """
        self.current_index = 0
        self.offline = offline
        self.chunk_size = chunk_size
        if self.offline and data is None:
            raise ValueError("Offline mode requires initial data.")
        self.load_data(data, data_loader_kwargs)
        self.is_data_loaded = False

    def load_data(self, data: str | np.ndarray, data_loader_kwargs: dict):
        """
        Load the data using the DataLoader.
        Parameters:
        data: str or np.ndarray
            The path to the data file or the data array itself.
        data_loader_kwargs: dict
            Additional keyword arguments to pass to the DataLoader.
        """
        self.data_loader = DataLoader(data, ignore_filtering=True, **data_loader_kwargs)
        if self.data_loader.init_data is None:
            return
        self.data_loader._apply_stack_epochs()
        self.is_data_loaded = True

    @property
    def init_data(self):
        # if not self.is_data_loaded:
        #     raise ValueError("Data not loaded.")
        return self.data_loader.init_data

    @property
    def num_chunks(self):
        return np.ceil(self.init_data.shape[-1] / self.chunk_size).astype(int)

    @property
    def data_rate(self):
        return self.data_loader.data_rate

    def get_next_chunk(self, chunk_size: int) -> tuple[bool, np.ndarray]:
        """
        Get the next chunk of data for processing.
        Parameters:
        chunk_size: int
            The size of the chunk to retrieve.
        Returns:
        bool: Whether there is more data to stream.
        np.ndarray: The next chunk of data.
        """
        if self.init_data is None:
            raise ValueError("No data loaded.")

        start_index = self.current_index
        if self.current_index + chunk_size > self.init_data.shape[-1]:
            return False, None

        end_index = min(self.current_index + chunk_size, self.init_data.shape[-1])
        chunk = self.init_data[..., start_index:end_index]
        if chunk.shape[-1] < chunk_size:
            chunk = np.concatenate([chunk, np.ones((chunk.shape[0], chunk_size - chunk.shape[-1])) * np.nan], axis=-1)
        self.current_index = end_index
        # self.current_index = 0 if end_index == self.init_data.shape[-1] else self.current_index + chunk_size
        # self.current_index = end_index % self.init_data.shape[-1]  # Wrap around if needed
        return True, chunk
