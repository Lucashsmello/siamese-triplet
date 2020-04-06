import numpy as np
import torch


class TripletDataset(torch.utils.data.Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.labels = self.dataset.targets
        self.labels_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)
            self.test_triplets = [self.generateTriplet(i, random_state)
                                  for i in range(len(dataset))]

    def generateTriplet(self, i, random_state=np.random):
        label1 = self.labels[i].item()
        positive_index = i
        while positive_index == i:
            positive_index = random_state.choice(self.label_to_indices[label1])
        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = random_state.choice(self.label_to_indices[negative_label])
        return [i, positive_index, negative_index]

    def __getitem__(self, index):
        if self.train:
            i1, i2, i3 = self.generateTriplet(index)
            item1, _ = self.dataset[i1]  # discarding label
            item2, _ = self.dataset[i2]
            item3, _ = self.dataset[i3]
        else:
            item1, _ = self.dataset[self.test_triplets[index][0]]
            item2, _ = self.dataset[self.test_triplets[index][1]]
            item3, _ = self.dataset[self.test_triplets[index][2]]

        return (item1, item2, item3), []

    def __len__(self):
        return len(self.dataset)
