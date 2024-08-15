import numpy as np
import tensorflow as tf
from random import shuffle
import os
import glob
from scipy import spatial
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM","O2IF","N2","AKG"]
# List all structures that have been preprocessed
precomputed_pdbs = glob.glob(
    os.path.join(params["masif_precomputation_dir"], "*_X.npy")
)
precomputed_pdbs.sort()
precomputed_pdbs = [p.split("/")[-1] for p in precomputed_pdbs][0:100]
print(precomputed_pdbs)
all_pdbs = ['_'.join(i.split('_')[0:6]) for i in precomputed_pdbs]
labels_dict = {"ADP": 1, "COA": 2, "FAD": 3, "HEM": 4, "NAD": 5, "NAP": 6, "SAM": 7, "O2IF":8, "AKG":9, "N2":10}

# Structures are randomly assigned to train, validation and test sets
shuffle(all_pdbs)
train = int(len(all_pdbs) * params["train_fract"])
val = int(len(all_pdbs) * params["val_fract"])
test = int(len(all_pdbs) * params["test_fract"])
print("Train", train)
print("Validation", val)
print("Test", test)
train_pdbs = all_pdbs[:train]
val_pdbs = all_pdbs[train : train + val]
test_pdbs = all_pdbs[train + val : train + val + test]

success = 0
precom_dir = params["masif_precomputation_dir"]
ligand_coord_dir = params["ligand_coords_dir"]
tfrecords_dir = params["tfrecords_dir"]
if not os.path.exists(tfrecords_dir):
    os.mkdir(tfrecords_dir)
with tf.io.TFRecordWriter(
    os.path.join(tfrecords_dir, "training_data.tfrecord")
) as writer:
    for i, pdb in enumerate(train_pdbs):
        print("Working on", pdb)
        try:
            # Load precomputed data
            # si, ddc, hbond, charge, hphob
            #TODO we need to add iface labels [0,1,2] and if iface >=1 then distance to AKG/cofactor
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, pdb + "_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, pdb + "_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligcoords.npy".format(pdb.split(".")[0])
                ), allow_pickle=True
            ).item()
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligtypes.npy".format(pdb.split(".")[0])
                )
            ).astype(str)
        except:
            print("Failed on", pdb)
            continue

        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int32
        )
        print(pdb, all_ligand_types, pocket_labels.shape)
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ln = structure_ligand[0]
            rn = structure_ligand[1].split('_')[1]
            ligand_coords = all_ligand_coords[int(rn)]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))
            print(len(pocket_points_flatten))
            pocket_labels[pocket_points_flatten, j] = labels_dict[ln]

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pocket_labels.shape)
        pocket_labels = tf.train.Int64List(value=pocket_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list), # data_element[0].shape
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list), # data_element[1]
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list), # data_element[2]
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list), # data_element[3]
            "pdb": tf.train.Feature(bytes_list=pdb_list), # data_element[5]
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels), # data_element[4]
        }
        # does this go in alphabetical?  what?????!!!!
        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1 == 0:
            print("Training data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(train_pdbs))


success = 0
with tf.io.TFRecordWriter(
    os.path.join(tfrecords_dir, "validation_data.tfrecord")
) as writer:
    for i, pdb in enumerate(val_pdbs):
        try:
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, pdb + "_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, pdb + "_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligcoords.npy".format(pdb.split(".")[0])
                ), allow_pickle=True
            ).item()
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligtypes.npy".format(pdb.split(".")[0])
                )
            ).astype(str)
        except:
            print("Failed on", pdb)
            continue

        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int32
        )
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ln = structure_ligand[0]
            rn = structure_ligand[1].split('_')[1]
            ligand_coords = all_ligand_coords[int(rn)]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))
            print(len(pocket_points_flatten))
            pocket_labels[pocket_points_flatten, j] = labels_dict[ln]

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pocket_labels.shape)
        pocket_labels = tf.train.Int64List(value=pocket_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        }

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1 == 0:
            print("Validation data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(val_pdbs))


success = 0
with tf.io.TFRecordWriter(
    os.path.join(tfrecords_dir, "testing_data.tfrecord")
) as writer:
    for i, pdb in enumerate(test_pdbs):
        try:
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, pdb + "_mask.npy")),-1)
            
            X = np.load(os.path.join(precom_dir, pdb + "_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligcoords.npy".format(pdb.split(".")[0])
                ), allow_pickle=True
            ).item()
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligtypes.npy".format(pdb.split(".")[0])
                )
            ).astype(str)
        except:
            print("Failed on", pdb)
            continue

        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int32
        )
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ln = structure_ligand[0]
            rn = structure_ligand[1].split('_')[1]
            ligand_coords = all_ligand_coords[int(rn)]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))
            print(len(pocket_points_flatten))
            pocket_labels[pocket_points_flatten, j] = labels_dict[ln]

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pocket_labels.shape)
        pocket_labels = tf.train.Int64List(value=pocket_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        }

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1 == 0:
            print("Testing data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(test_pdbs))

