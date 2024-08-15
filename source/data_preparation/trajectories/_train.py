import os
import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand_Brian_tf1 import MaSIF_ligand2
from masif_modules.read_ligand_tfrecords_tf1 import _parse_function
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from default_config.masif_opts import masif_opts
'''
For testing with TF1
'''


params = masif_opts["ligand"]
params["n_classes"]=10
precom_dir = params["masif_precomputation_dir"]
ligand_coord_dir = params["ligand_coords_dir"]
tfrecords_dir = params["tfrecords_dir"]




precom_dir = params["masif_precomputation_dir"]
ligand_coord_dir = params["ligand_coords_dir"]
tfrecords_dir = params["tfrecords_dir"]

# Load dataset
training_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "training_data.tfrecord")
)
validation_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "validation_data.tfrecord")
)
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "testing_data.tfrecord")
)
training_data = training_data.map(_parse_function)
validation_data = validation_data.map(_parse_function)
testing_data = testing_data.map(_parse_function)
out_dir = params["model_dir"]
output_model = out_dir + "model"

best_validation_loss = 1000
best_validation_accuracy = 0.0
total_iterations = 0
num_epochs = 100
num_training_samples = 72
num_validation_samples = 8
num_testing_samples = 20

training_iterator = training_data.make_one_shot_iterator()
training_next_element = training_iterator.get_next()

validation_iterator = validation_data.make_one_shot_iterator()
validation_next_element = validation_iterator.get_next()

testing_iterator = testing_data.make_one_shot_iterator()
testing_next_element = testing_iterator.get_next()

with tf.Session() as sess:
    learning_obj = MaSIF_ligand2(
            sess,
            max_rho=params["max_distance"],
            n_ligands=params["n_classes"],
            idx_gpu="/gpu:0",
            feat_mask=params["feat_mask"],
            costfun=params["costfun"],
            n_rotations=4
        )
    
    # test single pdb
    data_element = sess.run(training_next_element)
    print(data_element[5])
    labels = data_element[4]
    n_ligands = labels.shape[1]
    all_ligand_types = np.load(
            os.path.join(
                ligand_coord_dir, "{}_ligtypes.npy".format(data_element[5].decode("utf-8"))
            )
        ).astype(str)
    random_ligand = np.random.choice(n_ligands, 1)
    pocket_points = np.where(labels[:, random_ligand] != 0.0)[0]
    label = np.max(labels[:, random_ligand]) - 1
    pocket_labels = np.zeros(params["n_classes"], dtype=np.float32)
    pocket_labels[label] = 1.0
    npoints = pocket_points.shape[0]
    
    # Sample 32 points randomly
    sample = np.random.choice(pocket_points, 20, replace=False)
    
    input_feat=data_element[0][sample, :, :]
    rho_coords=np.expand_dims(data_element[1], -1)[sample, :, :]
    theta_coords=np.expand_dims(data_element[2], -1)[sample, :, :]
    mask=data_element[3][pocket_points[:20], :, :] # shouldn't this be pocket_points[sample]?
    print("input_feat.shape",input_feat.shape)
    print("rho_coords.shape",rho_coords.shape)
    print("theta_coords.shape",theta_coords.shape)
    print("mask.shape",np.array(mask).shape)

    print('learning_obj.print',learning_obj.print)
    np.save("_test1.npy", input_feat)
    feed_dict = {
        # TODO: if the feed dictionary is only a sample of "ligand/gas" binding sites,
        # why create an entire mesh of the entire protein and featurize entire protein?
        learning_obj.input_feat: input_feat, # input_feat [N vertices, max_feat_len, # features = 5]
                                             # example (7461, 50, 5)
        learning_obj.rho_coords: rho_coords,     # rho_coords (7461, 50)
        learning_obj.theta_coords: theta_coords, # theta_coords (7461, 50)
        learning_obj.mask: mask,
        learning_obj.labels: pocket_labels,
        learning_obj.keep_prob: 1.0,
        # name of pdb file
        learning_obj.name: data_element[5].decode("utf-8"),
        # name of ligand
        learning_obj.ligand: all_ligand_types[random_ligand]
    }

    optimizer, name, training_loss, norm_grad, logits, logits_softmax, computed_loss = learning_obj.session.run(
        [
            learning_obj.optimizer,
            learning_obj.name,
            learning_obj.data_loss,
            learning_obj.norm_grad,
            learning_obj.logits,
            learning_obj.logits_softmax,
            learning_obj.computed_loss,
        ],
        feed_dict=feed_dict, # I have no idea where this goes, __init__??, inference?? into the placeholders from self.session.run(init)??
    )