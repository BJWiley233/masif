
import numpy as np
from default_config.masif_opts import masif_opts

TRAIN_TEST_SPLIT = 0.7

class SimDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, list_):
        self.list_ = list_
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.list_))
        train_up_to = int(len(self.list_) * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # # converts alias to id
        # self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        # self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])

        # self.max_age = self.df['age'].max()
        
        return train_idx, valid_idx, test_idx

    def generate_frames(self, frame_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        input_feats, rhos, thetas, masks, labels = [], [], [], [], []
        while True:
            for idx in frame_idx:
                sample = self.list_[idx]
                
                input_feat = sample[0]
                rho = sample[1]
                theta = sample[2]
                mask = sample[3]
                label = sample[4]
                
                input_feats.append(input_feat)
                rhos.append(rho)
                thetas.append(theta)
                masks.append(mask)
                labels.append(label)
                
                # yielding condition
                if len(input_feats) >= batch_size:
                    # print("True")
                    yield (np.array(input_feats), np.array(rhos), np.array(thetas), np.array(masks)), np.array(labels)
                    input_feats, rhos, thetas, masks, labels = [], [], [], [], []
                    
            if not is_training:
                break
                
