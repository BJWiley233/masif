import os
import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand import MaSIF_ligand
from masif_modules.MaSIF_ligand_Brian_tf1 import MaSIF_ligand2
from masif_modules.read_ligand_tfrecords_tf1 import _parse_function
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from default_config.masif_opts import masif_opts


custom_params = {}
custom_params['cache_dir'] = 'nn_models/sc05/cache/'
custom_params['model_dir'] = 'nn_models/sc05/all_feat/model_data/'
custom_params['desc_dir'] = 'output/sc05/all_feat/model_data/'
custom_params['gif_eval_out'] = 'nn_models/sc05/gif_eval/'
custom_params['min_sc_filt'] = 0.5
custom_params['max_sc_filt'] = 1.0
custom_params['pos_surf_accept_probability'] = 1.0

params = masif_opts["site"]
params["n_conv_layers"]=1
for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]


params["max_distance"]=5

import glob
ppi_pair_ids=glob.glob("data_preparation/04a-precomputation_12A/precomputation/Sim_1_traj_*")
ppi_pair_ids=[i.split('/')[3] for i in ppi_pair_ids]
len(ppi_pair_ids)


# create [batch=len(ml2) x N x K x 5]
batch_size=200
B=500
params["n_conv_layers"]=1

ml = []
ml2 = {}

# ppi_pair_id="Sim_1_traj_42_frame_1307"
if os.path.exists("inputs_B%d.npy"%B):
    print("LOADING...", "inputs_B%d.npy"%B)
    ml2 = np.load("inputs_B%d.npy"%B, allow_pickle=True).item()
    ml=[list(i.values()) for i in list(ml2.values())]
else:
    for i, ppi_pair_id in enumerate(ppi_pair_ids[0:B]):
        try:
            if i % 250 == 0:
                print(i)
            mydir="data_preparation/04a-precomputation_12A/precomputation/" + ppi_pair_id + "/"
            pdbid = ppi_pair_id.split(".")[0]
            pid=pdbid
            rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
            theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
            input_feat = np.load(mydir + pid + "_input_feat.npy")

            iface_labels = np.load(mydir + pid + "_iface_labels.npy")

            mask = np.load(mydir + pid + "_mask.npy")
            mask = np.expand_dims(mask, 2)
            indices = np.load(mydir + pid + "_list_indices.npy", allow_pickle=True).item()['protein']

            # indices is (n_verts x <30), it should be
            # indices = pad_indices(indices, mask.shape[1])
            tmp = np.zeros((len(iface_labels), 3))
            for i in range(len(iface_labels)):
                if iface_labels[i] == 1:
                    tmp[i, 1] = 1
                elif iface_labels[i] == 2:
                    tmp[i, 2] = 1
                else:
                    tmp[i, 0] = 1
            iface_labels_dc = tmp
            super_pos = np.where(iface_labels == 2)[0]
            pos_labels = np.where(iface_labels == 1)[0]
            neg_labels = np.where(iface_labels == 0)[0]
            # print("PNS",len(pos_labels),len(neg_labels), len(super_pos))


            n = min(len(pos_labels), len(neg_labels))
            # print(ppi_pair_id,"len(pos_labels)", len(pos_labels), 'len(super_pos)', len(super_pos),'len(neg_labels)', len(neg_labels))
            n = min(n, batch_size // 2)
            n_neg = 500-n-len(super_pos)
            subset = np.concatenate([neg_labels[:n_neg], pos_labels[:n], super_pos])
            # print(len(subset))

            rho_wrt_center = rho_wrt_center[subset]
            theta_wrt_center = theta_wrt_center[subset]
            input_feat = input_feat[subset]
            mask = mask[subset]
            iface_labels_dc = iface_labels_dc[subset]
            indices = indices[subset]
            # neg_labels = range(0, (n*5))
            # pos_labels = range((n*5), (n*5)+n)
            # super_pos_labels = range(n+(n*5),n+(n*5)+len(super_pos))
            neg_labels = range(0, n_neg)
            pos_labels = range(n_neg, n_neg+n)
            super_pos_labels = range(n+(n_neg),n+(n_neg)+len(super_pos))
            # print("PNS labels",len(neg_labels),len(pos_labels), len(super_pos_labels))
            print(ppi_pair_id,"len(pos_labels)", len(pos_labels[:n]), 'len(super_pos_labels)', len(super_pos_labels),'len(neg_labels)', len(neg_labels[:n_neg]), len(np.concatenate([neg_labels,pos_labels,super_pos_labels])))
            assert(len(subset) == 500)

            ml.append([input_feat,rho_wrt_center,theta_wrt_center,mask,iface_labels_dc])
            ml2[ppi_pair_id] = {
                'input_feat':input_feat,
                'rho_wrt_center':rho_wrt_center,
                'theta_wrt_center':theta_wrt_center,
                'mask':mask,
                'iface_labels_dc':iface_labels_dc
            }
        except Exception as e:
            print("Error", e, ppi_pair_id)

    np.save("inputs_B%d.npy"%B, ml2)


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
                
data_generator = SimDataGenerator(ml)
train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

batch_size = 32
valid_batch_size = 32
train_gen = data_generator.generate_frames(train_idx, is_training=True, batch_size=batch_size)
valid_gen = data_generator.generate_frames(valid_idx, is_training=True, batch_size=valid_batch_size)
test_gen = data_generator.generate_frames(test_idx, is_training=False, batch_size=valid_batch_size)


class MaSIF_site2(tf.keras.layers.Layer):

    def inference(
        self,
        input_feat, # learning_obj.input_feat
        rho_coords, # learning_obj.rho_coords
        theta_coords,
        mask,
        W_conv,
        b_conv,
        mu_rho,
        sigma_rho,
        mu_theta,
        sigma_theta,
        eps=1e-5,
        mean_gauss_activation=True,
    ):
        n_samples = rho_coords.shape[1]
        n_vertices = rho_coords.shape[2]
        # print("n_samples", rho_coords.shape)
        # print("vertices", n_vertices)

        all_conv_feat = []
        for k in range(self.n_rotations):
            # print("rotation", k)
            # print("\trotation", k+1)
            # rho_coords_ = tf.reshape(rho_coords, [-1, 1])  # batch_size*n_vertices
            rho_coords_ = tf.reshape(rho_coords, (-1,n_samples*n_vertices, 1))
            # print("Brian rho_coords,rho_coords_", rho_coords.shape, rho_coords_.shape)
            # thetas_coords_ = tf.reshape(theta_coords, [-1, 1])  # batch_size*n_vertices
            thetas_coords_ = tf.reshape(theta_coords, (-1,n_samples*n_vertices,1))
            # print("thetas_coords,thetas_coords_", theta_coords.shape, thetas_coords_.shape)
            thetas_coords_ += k * 2 * np.pi / self.n_rotations
            thetas_coords_ = tf.math.mod(thetas_coords_, 2 * np.pi)
            rho_coords_ = tf.exp(
                -tf.square(rho_coords_ - mu_rho) / (tf.square(sigma_rho) + eps)
            )
            thetas_coords_ = tf.exp(
                -tf.square(thetas_coords_ - mu_theta) / (tf.square(sigma_theta) + eps)
            )
            
            gauss_activations = tf.multiply(
                rho_coords_, thetas_coords_
            )  # batch_size*n_vertices, n_gauss
            # print("gauss_activations shape 1",gauss_activations.shape)
            # print("gauss_activations1", gauss_activations.shape, rho_coords_.shape, thetas_coords_.shape)
            gauss_activations = tf.reshape(
                # gauss_activations, [n_samples, n_vertices, -1]
                gauss_activations,[-1,n_samples, n_vertices ,tf.shape(gauss_activations)[-1]]
            )  # batch_size, n_vertices, n_gauss
            # print("gauss_activations2", gauss_activations.shape, n_samples, n_vertices,-1)
            # print("gauss_activations shape 2",gauss_activations.shape)
            # print("gauss_activations,mask", gauss_activations.shape, mask.shape)
            gauss_activations = tf.multiply(gauss_activations, mask)
            # print("this is actually working")
            
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians
                gauss_activations /= (
                    tf.reduce_sum(gauss_activations, 1, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss
            # print("1", gauss_activations.shape)
            gauss_activations = tf.expand_dims(
                # gauss_activations, 2
                gauss_activations, 3
            )  # batch_size, inputsize, n_vertices, 1, n_gauss,
            # print("2", gauss_activations.shape)
            input_feat_ = tf.expand_dims(
                # input_feat, 3
                input_feat, 4
            )  # batch_size, inputsize, n_vertices, n_feat, 1
            # print("3", input_feat_.shape)
            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            # print("4", gauss_desc.shape)
            gauss_desc = tf.reduce_sum(gauss_desc, 2)  # batch_size, n_feat, n_gauss,
            # print("5", gauss_desc.shape)
            gauss_desc = tf.reshape(
                gauss_desc, [-1,n_samples, self.n_thetas * self.n_rhos]
            )  # batch_size, 80
            
            # print("6", gauss_desc.shape)
            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, 80
            # print("7", conv_feat.shape)
            # rho_coords2 = tf.identity(conv_feat)
            all_conv_feat.append(conv_feat)

        all_conv_feat = tf.stack(all_conv_feat)
        conv_feat = tf.reduce_max(all_conv_feat, 0)
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat
        # return 1,2
        
    def __init__(
        self,
        max_rho,
        n_thetas=4,
        n_rhos=3,
        n_gamma=1.0,
        learning_rate=1e-3,
        n_rotations=1,
        idx_gpu="/device:GPU:0",
        feat_mask=[1.0, 1.0, 1.0, 1.0, 1.0],
        n_conv_layers=1,
    ):
        super().__init__()
        self.max_rho=max_rho
       
        self.n_rhos=n_rhos
        self.n_thetas=n_thetas
        self.sigma_rho_init = (
            max_rho / 8
        ) 
        self.sigma_theta_init = 1.0
        self.learning_rate=learning_rate
        self.n_rotations=n_rotations
        self.n_feat = int(sum(feat_mask))
        self.n_labels = 3

        initial_coords = self.compute_initial_coordinates()
        self.initial_coords = initial_coords
        mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype(
                    "float32"
                )
        mu_theta_initial = np.expand_dims(initial_coords[:, 1], 0).astype(
            "float32"
        )
        # print(mu_rho_initial[0:100])
        self.mu_rho = []
        self.mu_theta = []
        self.sigma_rho = []
        self.sigma_theta = []
        for i in range(self.n_feat):
            self.mu_rho.append(
                tf.Variable(mu_rho_initial, name="mu_rho_{}".format(i))
            )  # 1, n_gauss
            self.mu_theta.append(
                tf.Variable(mu_theta_initial, name="mu_theta_{}".format(i))
            )  # 1, n_gauss
            self.sigma_rho.append(
                tf.Variable(
                    np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                    name="sigma_rho_{}".format(i),
                )
            )  # 1, n_gauss
            self.sigma_theta.append(
                tf.Variable(
                    (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                    name="sigma_theta_{}".format(i),
                )
            )  # 1, n_gauss
        
        
        self.b_conv = []
        for i in range(self.n_feat):
            self.b_conv.append(
                tf.Variable(
                    tf.zeros([self.n_thetas * self.n_rhos]),
                    name="b_conv_{}".format(i),
                )
            )

        self.W_conv = []
        
        for i in range(self.n_feat):
            initializer=tf.keras.initializers.GlorotUniform(seed=(i+1)*100)
            self.W_conv.append(
                    tf.Variable(initializer((self.n_thetas * self.n_rhos, 
                                                self.n_thetas * self.n_rhos)),
                                name="W_conv_{}".format(i)
                                )
            )
        self.mlp2= tf.keras.layers.Dense(
                    self.n_thetas * self.n_rhos,
                    activation=tf.nn.relu,
                )
        self.mlp3= tf.keras.layers.Dense(
                    self.n_feat,
                    activation=tf.nn.relu,
                )
        self.mlp4= tf.keras.layers.Dense(
                    self.n_labels, activation='softmax'
                )
              
        # self.b = self.add_weight(shape=(self.n_ligands*,), initializer="zeros", trainable=True)

    def call(self, inputs):
        # print("INPUTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", len(inputs))
        input_feat=inputs[0]
        # print("\tinput_feat", input_feat.shape)
        rho_coords=inputs[1]
        # print("\trho_coords", rho_coords.shape)
        theta_coords=inputs[2]
        # print("\ttheta_coords", theta_coords.shape)
        mask=inputs[3]
        # print("\tmask", mask.shape)
        # self.keep_prob=inputs[4]
        # labels=inputs[5]
    
        self.global_desc_1 = []
        self.rho_coords2 = []
        for i in range(self.n_feat):
        # for i in range(1):
            # print('feature',['si', 'ddc', 'hbond', 'charge', 'hphob'][i])
            # my_input_feat = tf.keras.ops.expand_dims(input_feat[:, :, i], 2)
            my_input_feat = tf.keras.ops.expand_dims(input_feat[:, :, :, i], 3)
            # print("WHAT", my_input_feat.shape)
            conv_feat = self.inference(
                            my_input_feat,
                            rho_coords,
                            theta_coords,
                            mask,
                            self.W_conv[i],
                            self.b_conv[i],
                            self.mu_rho[i],
                            self.sigma_rho[i],
                            self.mu_theta[i],
                            self.sigma_theta[i],
                        )
            self.global_desc_1.append(conv_feat)

        self.global_desc_1 = tf.stack(self.global_desc_1, axis=2)  
        # print("8", self.global_desc_1.shape)

        self.global_desc_1 = tf.reshape(
                    self.global_desc_1, [-1,500, self.n_thetas * self.n_rhos * self.n_feat]
                )
        # print("9", self.global_desc_1.shape)

        self.global_desc_1=self.mlp2(self.global_desc_1)
        # print("10", self.global_desc_1.shape)

        self.global_desc_1=self.mlp3(self.global_desc_1)
        # print("11", self.global_desc_1.shape)
       
        self.logits=self.mlp4(self.global_desc_1)
        # print("shape logits", self.logits.shape)
       
        # return self.global_desc_2, self.logits, self.rho_coords2
        return self.logits


    def compute_initial_coordinates(self):
        range_rho = [0.0, self.max_rho]
        range_theta = [0, 2 * np.pi]

        grid_rho = np.linspace(range_rho[0], range_rho[1], num=self.n_rhos + 1)
        grid_rho = grid_rho[1:]
        grid_theta = np.linspace(range_theta[0], range_theta[1], num=self.n_thetas + 1)
        grid_theta = grid_theta[:-1]

        grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
        grid_rho_ = (
            grid_rho_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_theta_ = (
            grid_theta_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_rho_ = grid_rho_.flatten()
        grid_theta_ = grid_theta_.flatten()

        coords = np.concatenate((grid_rho_[None, :], grid_theta_[None, :]), axis=0)
        coords = coords.T  # every row contains the coordinates of a grid intersection
        # print(coords.shape)
        return coords


import tensorflow as tf
import tensorflow.keras as keras


class MyModel(keras.Model):
  def __init__(self, n_rotations=1):
    super(MyModel, self).__init__()
    self.n_rotations=n_rotations
    print("self.n_rotations", self.n_rotations)
    self.conv1 = MaSIF_site2(
        params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=self.n_rotations,
        idx_gpu="/gpu:1",
        feat_mask=[1.0]*5,
        n_conv_layers=params["n_conv_layers"],
    )

  def call(self, inputs):
    logits = self.conv1(inputs)
    return logits



# train
from keras.callbacks import ModelCheckpoint

model = MyModel(n_rotations=4)
# inputA=keras.layers.Input((500,50,5))
# inputB=keras.layers.Input((500,50))
# inputC=keras.layers.Input((500,50))
# inputD=keras.layers.Input((500,50,1))
# model([inputA,inputB,inputC,inputD])

callbacks = [
    ModelCheckpoint("./model_checkpoint.keras", monitor='val_loss')
]
model.compile(optimizer=keras.optimizers.Adam(learning_rate= 1e-3), 
    loss='categorical_crossentropy', 
    metrics = ['accuracy'])

print()
print("######## Training", len(train_idx)//batch_size)
print()
model.fit(train_gen, steps_per_epoch=len(train_idx)//batch_size, epochs=5, callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)


# test
print()
print("######## Testing")
print()
test_gen = data_generator.generate_frames(test_idx, is_training=False, batch_size=valid_batch_size)

model.evaluate(test_gen, steps=len(test_idx)//valid_batch_size, 
               callbacks=callbacks)