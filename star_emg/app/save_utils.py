from biosiglive import save


class SaveStream:
    def __init__(self, save_path, buffer_len=10000):
        self.save_path = save_path
        self.data = None
        self.n_chanels = None
        self.channels_name = None
        self.data_rate = None
        self.meta_info = None
        self.counter = 0

    def init(self, n_channels, channels_name, data_rate):
        self.data = np.empty((n_channels * 3, self.buffer_len))
        self.n_chanels = n_channels
        self.channels_name = channels_name
        self.data_rate = data_rate

    def add_data(self, data, process_params):
        self.data[:, self.counter : self.counter + data.shape[-1]] = data
        self.counter += data.shape[-1]
        # if self.counter

    def _get_meta(self):
        meta_dict = {
            "n_chanels": self.n_chanels,
            "channels_name": self.channels_name,
            "data_rate": self.data_rate,
        }
