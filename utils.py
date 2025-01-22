import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import plot_utils
import datetime
import os

loss_fn = mse
optimizer = Adam(learning_rate=0.0002)


def metrics_rmse(y_true, y_pred):  # RMSE : root mean square error
    result = np.sqrt(np.mean(np.square((y_true - y_pred))))
    return np.round(result, 2)


def metrics_mape(y_true, y_pred):  # MAPE : mean absolute percentage error
    result = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    return np.round(result, 2)


def train_step(inputs, target, model, config):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        if config["label_num"] == 3:
            loss = (loss_fn(target[:, 0], predictions[:, 0])
                    + loss_fn(target[:, 1], predictions[:, 1])
                    + loss_fn(target[:, 2], predictions[:, 2])) / 3 + sum(model.losses)
        elif config["label_num"] == 1:
            loss = loss_fn(target[:, 0], predictions[:, 0]) + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_eval(dataset_va, model, config):
    step = loss = 0
    target_set = []
    pred_set = []
    # idx_set = np.random.permutation(dataset_va.n_graphs)

    while step < dataset_va.n_graphs:
        # idx = idx_set[step]
        target = dataset_va[step].y
        pred = model([dataset_va[step].x[np.newaxis, :, :], dataset_va[step].a], training=False)#[np.newaxis, :, :]], training=False)
        step += 1

        if config["label_num"] == 3:
            loss += (loss_fn(target[:, 0], pred[:, 0])
                     + loss_fn(target[:, 1], pred[:, 1])
                     + loss_fn(target[:, 2], pred[:, 2])) / 3

        elif config["label_num"] == 1:
            loss += loss_fn(target, pred[:, 0])
            target_set.append(target)
            pred_set.append(pred[:, 0])

        if step == dataset_va.n_graphs:
            val_loss = loss / dataset_va.n_graphs
            Y_scaler = config['Y_scaler']
            y_true = Y_scaler.inverse_transform(target_set)
            y_pred = Y_scaler.inverse_transform(pred_set)
            val_mae = mean_absolute_error(y_true, y_pred)
            val_rmse = metrics_rmse(y_true, y_pred)
            val_mape = metrics_mape(y_true, y_pred)
            val_r = r2_score(y_true, y_pred)

            return val_loss, val_mae, val_rmse, val_mape, val_r


def model_train_eval(dataset, config, params):
    # Parameters
    global model
    model = None
    N = None
    M = dataset['dataset_tr'][0].n_nodes
    F = dataset['dataset_tr'].n_node_features  # Dimension of node features
    S = dataset['dataset_tr'].n_edge_features  # Dimension of edge features
    n_out = dataset['dataset_tr'].n_labels  # Dimension of the target


    # build model
    if config["transfer_learning"]:
        if config["mode_two"] == "LHS":
            config["iteration"] = 0

        # transfer learning for making a robust model considering the previous best model
        if config["iteration"] == 0:
            model_name = f'{config["label_name"]}_{config["transfer_learning"]}'
            model = params['model'](N, F, n_out, params, model_name)
            model.summary()

        elif config["transfer_best_or_prev"] == True:
            names = [file[:-3] for file in os.listdir(config["model_dir"]) if file[-3:] == '.h5']
            best_var_r_list = [float(file[:-3].split("_")[6]) for file in os.listdir(config["model_dir"]) if file[-3:] == '.h5']
            best_var_r_list_index = best_var_r_list.index(max(best_var_r_list))
            load_name = names[best_var_r_list_index]

            model = load_model(f'{config["model_dir"]}/{load_name}.h5', custom_objects=params['custom_objects'])
            print('Transfer Learning Model name', load_name)
            model.summary()
        elif config["transfer_best_or_prev"] == False:
            print("Transfer learning Names :", config["model_load_name"])
            model = load_model(f'{config["model_dir"]}/{config["model_load_name"]}.h5', custom_objects=params['custom_objects'])

            model.summary()

    else:
        # not transfer learning for searching the best num_subdivide and num_cluster
        model_name = f'{config["label_name"]}_{config["transfer_learning"]}'
        model = params['model'](N, F, n_out, params, model_name)
        model.summary()

    # early stopping parameters
    epoch = step = loss = 0
    patience = config['patience']
    breaker = True

    best_val_loss = np.inf
    best_val_r = np.inf
    best_weights = None
    patience_temp = None

    # Train
    for e in range(config["epochs"]):
        idx = np.random.permutation(dataset['dataset_tr'].n_graphs)
        for i in idx:
            step += 1

            A_arr = dataset['dataset_tr'][i].a#[np.newaxis, :, :]
            X_arr = dataset['dataset_tr'][i].x[np.newaxis, :, :]
            Y_arr = dataset['dataset_tr'][i].y[np.newaxis, :]

            loss += train_step([X_arr, A_arr], Y_arr, model, config)
            if step == dataset['dataset_tr'].n_graphs:
                step = 0
                epoch += 1
                val_loss, val_mae, val_rmse, val_mape, val_r = train_eval(dataset['dataset_va'], model, config)
                print("Ep.{} Train loss {:.5f} / Val loss {:.5f}, mae {:.2f}, rmse {:.2f}, mape {:.2f}, r {:.5f}".format(
                    epoch, (loss/dataset['dataset_tr'].n_graphs), val_loss, val_mae, val_rmse, val_mape, val_r))
                loss = 0

                # Check if loss improved for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_r = val_r
                    patience_temp = patience
                    print("New best val_loss {:.5f}".format(best_val_loss))
                    best_weights = model.get_weights()
                else:
                    patience_temp -= 1
                    if patience_temp == 0:
                        model.set_weights(best_weights)
                        config["best_val_loss"] = round(float(best_val_loss), 5)
                        config["best_val_mae"] = round(float(val_mae), 2)
                        config["best_val_rmse"] = round(float(val_rmse), 2)
                        config["best_val_mape"] = round(float(val_mape), 2)
                        config["best_val_r"] = round(float(best_val_r), 5)
                        config["iteration"] += 1
                        config["model_load_name"] = f'{config["iteration"]}_{config["model_save_name"]}_{config["num_subdivide"]}_{config["num_cluster"]}_{config["best_val_loss"]}_{config["best_val_r"]}'

                        print("Early stopping (best val_loss: {})".format(config["best_val_loss"]))
                        print("Save Model (best model: {})".format(config["model_load_name"]))

                        # save best val_loss BO model
                        model.save(f'{config["model_dir"]}/{config["model_load_name"]}.h5')
                        breaker = False
                        break
        if not breaker:
            break

    return best_val_r


def test_step(config, dataset, pre_model):
    step = loss = 0
    target_set = []
    pred_set = []

    while step < dataset.n_graphs:

        target = dataset[step].y
        pred = pre_model([dataset[step].x[np.newaxis, :, :], dataset[step].a], training=False)#[np.newaxis, :, :]], training=False)
        step += 1

        if config["label_num"] == 3:
            loss += (loss_fn(target[:, 0], pred[:, 0])
                     + loss_fn(target[:, 1], pred[:, 1])
                     + loss_fn(target[:, 2], pred[:, 2])) / 3

        elif config["label_num"] == 1:
            loss += loss_fn(target, pred[:, 0])
            target_set.append(target)
            pred_set.append(pred[:, 0])

        if step == dataset.n_graphs:
            loss = loss / dataset.n_graphs
            Y_scaler = config['Y_scaler']
            y_true = Y_scaler.inverse_transform(target_set)
            y_pred = Y_scaler.inverse_transform(pred_set)
            rmse = metrics_rmse(y_true, y_pred)
            mape = metrics_mape(y_true, y_pred)
            r = r2_score(y_true, y_pred)

    # fig = plt.figure(figsize=(15,4))
    # ax = fig.add_subplot(1,1,1)

    # if config["label_num"] == 'mass':
    #     color_ = 'orange'
    # elif config["label_num"] == 'rim':
    #     color_ = 'mediumseagreen'
    # elif config["label_num"] == 'disk':
    #     color_ = 'royalblue'

    plt.rc('font', size=10)
    plt.plot(y_true,y_true,alpha=0.5,label='ground-truth')
    plt.scatter(y_pred,y_true,s=50,alpha=0.7,edgecolor='royalblue', color = 'royalblue',label='R2 = {:.5f}'.format(r2_score(y_true,y_pred)))
    plt.title(f'{config["label_name"]}')
    plt.legend()
    plt.savefig(f'./{config["label_name"]}')

    return loss, rmse, mape, r

def model_test(dataset, config, params):

    dataset_tr = dataset["dataset_tr"]
    dataset_va = dataset["dataset_va"]
    dataset_te = dataset["dataset_te"]

    pre_model = load_model(f'{config["model_dir"]}/{config["model_load_name"]}.h5', custom_objects=params['custom_objects'])

    # model_test
    # loss_tr, rmse_tr, mape_tr, r_tr = test_step(config, dataset_tr, pre_model)
    # loss_va, rmse_va, mape_va, r_va = test_step(config, dataset_va, pre_model)
    loss_te, rmse_te, mape_te, r_te = test_step(config, dataset_te, pre_model)

    # print("Train: loss {:.5f}, RMSE {:.2f}, MAPE {:.2f}, R {:.3f}".format(loss_tr, rmse_tr, mape_tr, r_tr))
    # print("Valid: loss {:.5f}, RMSE {:.2f}, MAPE {:.2f}, R {:.3f}".format(loss_va, rmse_va, mape_va, r_va))
    print("Test: loss {:.5f}, RMSE {:.2f}, MAPE {:.2f}, R {:.5f}".format(loss_te, rmse_te, mape_te, r_te))

    # grad cam
    mask_ = test_grad_cam(dataset_te, pre_model)

    # plot and save
    if config["rotation_animation_bool"]: plot_utils.roatation_animation(dataset_te, mask_)  # animation save
    if config["remesh_animation_bool"]: plot_utils.remesh_animation(dataset_te, mask_)
    if config["image_plot_bool"]: plot_utils.image_plot(dataset_te, 1, 3)


def test_grad_cam(dataset_te, pre_model):
    inputs = (dataset_te[0].x[np.newaxis, :, :], dataset_te[0].a[np.newaxis, :, :])
    cam = plot_utils.CAM(pre_model)
    mask = cam.getMasks(inputs)
    mask = np.array(mask)

    return mask


def write_results_of_each_model(config, params, bayesian_results, one_iter_total_times, one_iter_train_times, total_iter_times):#, qual_min_angle_mins, qual_min_angle_avgs, qual_jacobian_mins, qual_jacobian_avgs, qual_AR_maxs, qual_AR_avgs):
    file = open(f'{config["model_dir"]}/{config["model_load_name"]}.txt', 'w', encoding='utf-8')
    file.write(f'# Model PARAMETERS\n')
    for key, value in params.items():
        file.write(f"{str(key)}: {str(value)}\n")

    file.write(f'\n\n# Model Config\n')
    for key, value in config.items():
        file.write(f"{str(key)}: {str(value)}\n")

    for result in bayesian_results:
        file.write(f"{result}\n")

    file.write(f'# one_iter_total_times\n')
    for i, time in enumerate(one_iter_total_times):
        file.write(f"{i+1}_{time}\n")

    file.write(f'# one_iter_train_times\n')
    for i, time in enumerate(one_iter_train_times):
        file.write(f"{i+1}_{time}\n")

    file.write(f'# total_iter_times\n')
    for i, time in enumerate(total_iter_times):
        file.write(f"{i+1}_{time}\n")
    #
    # file.write(f'\n\n# ------Mesh Qualities------\n\n')
    #
    # file.write(f'# qual_min_angle_mins\n')
    # for i, time in enumerate(qual_min_angle_mins):
    #     file.write(f"{i+1}_{time}\n")
    #
    # file.write(f'# qual_min_angle_avgs\n')
    # for i, time in enumerate(qual_min_angle_avgs):
    #     file.write(f"{i + 1}_{time}\n")
    #
    # file.write(f'# qual_jacobian_mins\n')
    # for i, time in enumerate(qual_jacobian_mins):
    #     file.write(f"{i + 1}_{time}\n")
    #
    # file.write(f'# qual_jacobian_avgs\n')
    # for i, time in enumerate(qual_jacobian_avgs):
    #     file.write(f"{i + 1}_{time}\n")
    #
    # file.write(f'# qual_AR_maxs\n')
    # for i, time in enumerate(qual_AR_maxs):
    #     file.write(f"{i + 1}_{time}\n")
    #
    # file.write(f'# qual_AR_avgs\n')
    # for i, time in enumerate(qual_AR_avgs):
    #     file.write(f"{i + 1}_{time}\n")

    file.close()

def write_results_of_bayesian(config, bayesian_results):
    dt_now = datetime.datetime.now()
    write_name = f'{dt_now.year}/{dt_now.month}/{dt_now.day}/{dt_now.hour}_{dt_now.minute}'
    file = open(f'{config["model_dir"]}/BO_results_{write_name}.txt', 'w', encoding='utf-8')
    file.write(f'# Results of Bayesian Optimization\n')
    for result in bayesian_results:
        file.write(f"{result}\n")
    file.close()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

