import os
import numpy as np
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand import MaSIF_ligand
from data_preparation.trajectories.masif_site import MaSIF_site2
from data_preparation.trajectories.data_generator import SimDataGenerator
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from default_config.masif_opts import masif_opts
import tensorflow as tf
import tensorflow.keras as keras
import glob



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


data = "/data/pompei/bw973/Oxygenases/PHD2/PHD2_50_O2IF/Bundle/Sim1"
ppi_pair_ids=glob.glob(data+"/data_preparation/04a-precomputation_12A/precomputation/Sim_1_traj_*")
ppi_pair_ids=[i.split('/')[-1] for i in ppi_pair_ids]
print(len(ppi_pair_ids))


# create [batch=B=len(ml2) x #Vertices=N x #Neighbors=K x #Features=5]
batch_size=200
B=6000
params["n_conv_layers"]=1

ml = []
ml2 = {}

# ppi_pair_id="Sim_1_traj_42_frame_1307"
if os.path.exists(data+"/inputs_B%d.npy"%B):
    print("LOADING...", "inputs_B%d.npy"%B)
    ml2 = np.load(data+"/inputs_B%d.npy"%B, allow_pickle=True).item()
    ml=[list(i.values()) for i in list(ml2.values())]
    print("len ml", len(ml))
else:
    for i, ppi_pair_id in enumerate(ppi_pair_ids[0:B]):
        try:
            if i % 250 == 0:
                print(i)
            mydir=data+"/data_preparation/04a-precomputation_12A/precomputation/" + ppi_pair_id + "/"
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

    np.save(data+"/inputs_B%d.npy"%B, ml2)


                
data_generator = SimDataGenerator(ml)
train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

batch_size = 32
valid_batch_size = 32
train_gen = data_generator.generate_frames(train_idx, is_training=True, batch_size=batch_size)
valid_gen = data_generator.generate_frames(valid_idx, is_training=True, batch_size=valid_batch_size)
test_gen = data_generator.generate_frames(test_idx, is_training=False, batch_size=valid_batch_size)



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
print("######## Training", "length batch steps", len(train_idx)//batch_size)
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