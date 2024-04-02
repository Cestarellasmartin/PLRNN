from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
import numpy as np
import torch as tc



class GeneralDataset(Dataset):
    def __init__(self, data: np.ndarray, inputs: np.ndarray, seq_len: int,
                 batch_size: int, bpe: int):
        super().__init__()

        if not isinstance(data, list) and not isinstance(inputs, list):
            self._data = tc.tensor(data, dtype=tc.float)
            self._inputs = tc.tensor(inputs, dtype=tc.float)
        else:                        ######### for data in lists with different trial lengths
            self._data = [tc.tensor(t, dtype=tc.float) for t in data]
            self._inputs = [tc.tensor(t, dtype=tc.float) for t in inputs]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.bpe = bpe

    @property
    def data(self):
        #return self._data.clone()#, self._inputs.clone()
        return self._data.clone()

    @property   ## what exactly does this ?
    def inputs(self):
        #return self._inputs.clone()
        return self._inputs.clone()

    def __len__(self):                              ### overwrites the actual len function
        return len(self._data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self._data[idx:idx + self.seq_len]
        y = self._data[idx + 1:idx + self.seq_len + 1]
        s = self._inputs[idx + 1:idx + self.seq_len + 1]
        return x, y, s

    def get_rand_dataloader(self):
        indices = np.random.permutation(len(self))[:self.bpe*self.batch_size]
        subset = Subset(self, indices.tolist())
        dataloader = DataLoader(subset, batch_size=self.batch_size)
        return dataloader#, indices.tolist()

    #def get_rand_trialloader(self):
        #dataloader = DataLoader(self, batch_size=self.batch_size, shuffle=True)
        #return dataloader



    def to(self, device: tc.device):
        self._data = self._data.to(device)
        self._inputs = self._inputs.to(device)   #### nochmal prüfen



class TrialDataset(Dataset):
    def __init__(self, data: np.ndarray, inputs: np.ndarray, seq_len: int,
                 batch_size: int, bpe: int, W_trial: bool):
        super().__init__()

        if not isinstance(data, list) and not isinstance(inputs, list):
            self._data = tc.tensor(data, dtype=tc.float)
            self._inputs = tc.tensor(inputs, dtype=tc.float)
        else:                        ######### for data in lists with different trial lengths
            self._data = [tc.tensor(t, dtype=tc.float) for t in data]
            self._inputs = [tc.tensor(t, dtype=tc.float) for t in inputs]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.bpe = bpe
        self.W_trial = W_trial

    @property
    def data(self):
        return self._data.copy()#, self._inputs.clone()

    @property   ## what exactly does this ?
    def inputs(self):
        return self._inputs.copy()

    def __len__(self):                              ### overwrites the actual len function
        return len(self._data)

    def __getitem__(self, idx):
        i = np.random.randint(0, high=self._data[idx].shape[0] - self.seq_len, size=1, dtype=int).item()
        x = self._data[idx][i:i + self.seq_len]
        #x = self._data[idx][:-1, :]
        y = self._data[idx][i + 1:i + self.seq_len + 1]
        #y = self._data[idx][1:, :]
        s = self._inputs[idx][i + 1:i + self.seq_len + 1]
        #s = self._inputs[idx][1:, :]
        return x, y, s

    def get_seq_len(self, indices):
        ## getting shortest trial out of the batch ##
        seq_len = []
        for i in indices:
            seq_len.append(self._data[i].shape[0])
        return min(seq_len)-1

    def chunkify(self, lst, n):
        split = np.split(lst, n)
        return [x.tolist() for x in split]

    def get_rand_dataloader(self):

        #indices = np.random.permutation(len(self))[:self.bpe*self.batch_size]
        indices = np.random.randint(len(self), size=self.bpe*self.batch_size)
        #indices = np.concatenate([np.random.permutation(len(self))[:self.batch_size] for i in range(self.bpe)])
        if self.seq_len > self.get_seq_len(indices):
            self.seq_len = self.get_seq_len(indices)
        subset = Subset(self, indices.tolist())
        dataloader = DataLoader(subset, batch_size=self.batch_size, drop_last=False, num_workers=0, pin_memory=False)
        #sampler = RandomSampler(self, replacement=True, num_samples=self.bpe * self.batch_size)
        #dataloader = DataLoader(self, sampler=sampler, batch_size=self.batch_size)
        return dataloader, self.chunkify(indices, self.bpe)


    def to(self, device: tc.device):
        if isinstance(self._data, list):
            self._data = [t.to(device) for t in self._data]
            self._inputs = [t.to(device) for t in self._inputs]
        else:
            self._data = self._data.to(device)
            self._inputs = self._inputs.to(device)


class TestDataset(Dataset):
    def __init__(self, test_data: np.ndarray, test_inputs: np.ndarray):
        super().__init__()
        if not isinstance(test_data, list) and not isinstance(test_inputs, list):
            self._test_data = tc.tensor(test_data, dtype=tc.float)
            self._test_inputs = tc.tensor(test_inputs, dtype=tc.float)
        else:                        ######### for data in lists with different trial lengths
            self._test_data = [tc.tensor(t, dtype=tc.float) for t in test_data]
            self._test_inputs = [tc.tensor(t, dtype=tc.float) for t in test_inputs]
        #self.seq_len = data.shape[0]
        #self.batch_size = batch_size
        #self.bpe = bpe

    @property
    def data(self):
        return self._test_data.clone()

    @property
    def inputs(self):
        return self._test_inputs.clone()

    def __len__(self):
        return len(self._test_data)
#
#     def __len__(self):
#         return len(self._inputs)
#
    def __getitem__(self, idx):
        #if self._data[idx].shape[0]-self.seq_len-1 > 1:
            #i = np.random.permutation(self._data[idx].shape[0] - self.seq_len-1)[1]
        #else:
            #i = self._data[idx].shape[0]-self.seq_len-1 ### include random choice between 0 and 1
        #i = 0
        seq_len = self._test_data.shape[0]-1
        x = self._test_data[idx:idx + seq_len]
        #x = self._test_data[:-1, :]
        y = self._test_data[idx + 1:idx + seq_len + 1]
        #y = self._test_data[1:, :]
        s = self._test_inputs[idx + 1:idx + seq_len + 1]
        #s = self._test_inputs[1:, :]
        return x, y, s

    def to(self, device: tc.device):
        if isinstance(self._test_data, list):
            self._test_data = [t.to(device) for t in self._test_data]
            self._test_inputs = [t.to(device) for t in self._test_inputs]
        else:
            self._test_data = self._test_data.to(device)
            self._test_inputs = self._test_inputs.to(device)


class custom_TrialDataset(Dataset):
    def __init__(self, data: np.ndarray, inputs: np.ndarray, seq_len: int, batch_size: int, bpe: int, W_trial: bool
                 ):
        super().__init__()

        # data shape (trials x T x N)
        # assert data.ndim == 3
        # self._data = tc.tensor(data, dtype=tc.float)
        # self.tr, self.T, self.N = self._data.size()
        if not isinstance(data, list) and not isinstance(inputs, list):
            self._data = tc.tensor(data, dtype=tc.float)
            self._inputs = tc.tensor(inputs, dtype=tc.float)
        else:  ######### for data in lists with different trial lengths
            self._data = [tc.tensor(t, dtype=tc.float) for t in data]
            self._inputs = [tc.tensor(t, dtype=tc.float) for t in inputs]

        # full trial if seq_len is not specified
        #if seq_len == -1:
            #self.seq_len = self.T
        #else:
        self.seq_len = seq_len
        # self.batch_size = batch_size
        # self.bpe = bpe
        # self.seq_len = 500
        self.batch_size = batch_size
        self.bpe = bpe
        self.W_trial = W_trial
        #self.setofseq = None

    @property
    def data(self):
        return self._data.copy()

    @property  ## what exactly does this ?
    def inputs(self):
        # return self._inputs.clone()
        return self._inputs.copy()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx, seq_len=None):
        # self.set_of_seq =
        if seq_len == None:
            self.seq_len = self._data[idx].shape[0] - 1
        else:
            self.seq_len = seq_len

        if self._data[idx].shape[0] - self.seq_len - 1 > 1:
            i = np.random.permutation(self._data[idx].shape[0] - self.seq_len - 1)[1]
        else:
            i = self._data[idx].shape[0] - self.seq_len - 1  ### include random choice between 0 and 1
        x = self._data[idx][i:i + self.seq_len]
        # x = self._data[idx][:-1, :]
        y = self._data[idx][i + 1:i + self.seq_len + 1]
        # y = self._data[idx][1:, :]
        s = self._inputs[idx][i + 1:i + self.seq_len + 1]
        # s = self._inputs[idx][1:, :]
        return x, y, s

    # def custom_collate(self, indices):
    # return [self.get_seq_len(i) for i in indices ]

    def get_seq_len(self, indices):
        ## getting shortest trial out of the batch ##
        seq_len = []
        for i in indices:
            seq_len.append(self._data[i].shape[0])
        return min(seq_len) - 1

    def chunkify(self, lst, n):
        return [lst[i::n] for i in range(n)] if self.W_trial else [None for i in range(n)]

    def get_rand_dataloader(self):

        # indices = np.random.permutation(len(self))[:self.bpe*self.batch_size]
        indices = np.random.randint(len(self), size=self.bpe * self.batch_size)
        chunked_indices = self.chunkify(indices.tolist(), self.bpe)
        # print(chunked_indices)
        list_of_lengths = [self.get_seq_len(i) for i in chunked_indices]
        #print(len(list_of_lengths))
        observations = []
        targets = []
        inputs = []
        for i in range(len(list_of_lengths)):
            obs, targ, inp = self.__getitem__(chunked_indices[i][0], list_of_lengths[i])
            for j in chunked_indices[i][1:]:
                x, y, s = self.__getitem__(j, list_of_lengths[i])

                obs = tc.cat((obs, x), dim=0)
                targ = tc.cat((targ, y), dim=0)
                inp = tc.cat((inp, s), dim=0)

            observations.append(obs.reshape(self.batch_size, list_of_lengths[i], 141))
            targets.append(targ.reshape(self.batch_size, list_of_lengths[i], 141))
            inputs.append(inp.reshape(self.batch_size, list_of_lengths[i], 2))

        return zip(observations, targets, inputs), self.chunkify(indices.tolist(), self.bpe)

    def to(self, device: tc.device):
        if isinstance(self._data, list):
            self._data = [t.to(device) for t in self._data]
            self._inputs = [t.to(device) for t in self._inputs]
        else:
            self._data = self._data.to(device)
            self._inputs = self._inputs.to(device)