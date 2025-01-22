from pathlib import Path


def load_config(mode,best_iteration,best_model_loss,best_model_r,best_model_num_subdivide,best_model_num_cluster,mesh_ratio,model_name,trans_name,trans,label_num,label_name,epochs,patience,num_subdivide,num_cluster,learning_rate,transfer_learning,transfer_best_or_prev,mode_two,iteration):

    data_dir = f'data/{mesh_ratio}'
    model_load_name = None
    model_save_name = None

    if label_num == 3:
        model_load_name = f'{best_iteration}_{model_name}_{best_model_num_subdivide}_{best_model_num_cluster}_{best_model_loss}_{best_model_r}'
        model_save_name = f'{model_name}'
    elif label_num == 1:
        model_load_name = f'{best_iteration}_{model_name}_{label_name}_{best_model_num_subdivide}_{best_model_num_cluster}_{best_model_loss}_{best_model_r}'
        model_save_name = f'{model_name}_{label_name}'

    config = {
        # experiment parameters
        "mode": mode,
        "mode_two": mode_two,
        "mesh_ratio": mesh_ratio,
        "label_num": label_num,
        "label_name": label_name,
        "data_dir": data_dir,
        "trans_name": trans_name,
        "trans": trans,
        "model_name": model_name,

        # train parameters
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,

        # train BO parameters for load/save name
        "num_subdivide": num_subdivide,
        "num_cluster": num_cluster,
        "list_num_subdivide": None,
        "list_num_cluster": None,

        # results of metrics for 1 iteration
        "iteration": iteration,
        "best_val_loss": None,
        "best_val_mae": None,
        "best_val_rmse": None,
        "best_val_mape": None,
        "best_val_r": None,

        # directory paths
        "stl_dir": f'{data_dir}/stl',
        "A_dir": f'{data_dir}/{trans_name}/A',
        "X_dir": f'{data_dir}/{trans_name}/X',
        "Y_dir": f'{data_dir}/{trans_name}/Y',
        "cells_dir": f'{data_dir}/{trans_name}/cells',
        "graphs_dir": f'{data_dir}/{trans_name}/graphs',
        "model_dir": f'{data_dir}/{trans_name}/models/{mode_two}_{label_name}_{transfer_learning}',

        # train/test bool variables
        "make_original_graph_bool": False,
        "make_subdivided_graph_bool": True,     # default: True
        "transfer_learning": transfer_learning,     # transfer learning bool
        "transfer_best_or_prev":transfer_best_or_prev,
        "build_model_bool": True,     # default: True

        # test plot bool variables
        "image_plot_bool": False,
        "rotation_animation_bool": False,
        "remesh_animation_bool": False,

        # test parameters
        "best_model_num_subdivide": best_model_num_subdivide,
        "best_model_num_cluster": best_model_num_cluster,
        "model_load_name": model_load_name,     # {model_name}_{label_name}_{model_best_subdivide}_{model_best_cluster}_{model_best_loss}
        "model_save_name": model_save_name,     # {model_name}_{label_name}
    }

    # config["model_load_dir"] = f'{config["model_dir"]}_{model_load_name}'
    # config["model_save_dir"] = f'{config["model_dir"]}_{model_save_name}'


    # Make directories
    Path(config["A_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["X_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["Y_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["cells_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["graphs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(f'{config["graphs_dir"]}/original').mkdir(parents=True, exist_ok=True)
    Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)

    return config


def load_config_2(mode,best_iteration,best_model_loss,best_model_r,best_model_num_subdivide,best_model_num_cluster,mesh_ratio,model_name,trans_name,trans,label_num,label_name,epochs,patience,num_subdivide,num_cluster,learning_rate,transfer_learning,transfer_best_or_prev,mode_two,iteration):

    data_dir = f'/workspace/data/{mesh_ratio}'
    model_load_name = None
    model_save_name = None

    if label_num == 3:
        model_load_name = f'{best_iteration}_{model_name}_{best_model_num_subdivide}_{best_model_num_cluster}_{best_model_loss}_{best_model_r}'
        model_save_name = f'{model_name}'
    elif label_num == 1:
        model_load_name = f'{best_iteration}_{model_name}_{label_name}_{best_model_num_subdivide}_{best_model_num_cluster}_{best_model_loss}_{best_model_r}'
        model_save_name = f'{model_name}_{label_name}'

    config = {
        # experiment parameters
        "mode": mode,
        "mode_two": mode_two,
        "mesh_ratio": mesh_ratio,
        "label_num": label_num,
        "label_name": label_name,
        "data_dir": data_dir,
        "trans_name": trans_name,
        "trans": trans,
        "model_name": model_name,

        # train parameters
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,

        # train BO parameters for load/save name
        "num_subdivide": num_subdivide,
        "num_cluster": num_cluster,
        "list_num_subdivide": None,
        "list_num_cluster": None,

        # results of metrics for 1 iteration
        "iteration": iteration,
        "best_val_loss": None,
        "best_val_mae": None,
        "best_val_rmse": None,
        "best_val_mape": None,
        "best_val_r": None,

        # directory paths
        "stl_dir": f'{data_dir}/stl',
        "A_dir": f'{data_dir}/{trans_name}/A',
        "X_dir": f'{data_dir}/{trans_name}/X',
        "Y_dir": f'{data_dir}/{trans_name}/Y',
        "cells_dir": f'{data_dir}/{trans_name}/cells',
        "graphs_dir": f'{data_dir}/{trans_name}/graphs',
        "model_dir": f'{data_dir}/{trans_name}/models/{mode_two}_{label_name}_{transfer_learning}',

        # train/test bool variables
        "make_original_graph_bool": False,
        "make_subdivided_graph_bool": True,     # default: True
        "transfer_learning": transfer_learning,     # transfer learning bool
        "transfer_best_or_prev":transfer_best_or_prev,
        "build_model_bool": True,     # default: True

        # test plot bool variables
        "image_plot_bool": False,
        "rotation_animation_bool": False,
        "remesh_animation_bool": False,

        # test parameters
        "best_model_num_subdivide": best_model_num_subdivide,
        "best_model_num_cluster": best_model_num_cluster,
        "model_load_name": model_load_name,     # {model_name}_{label_name}_{model_best_subdivide}_{model_best_cluster}_{model_best_loss}
        "model_save_name": model_save_name,     # {model_name}_{label_name}
    }

    # config["model_load_dir"] = f'{config["model_dir"]}_{model_load_name}'
    # config["model_save_dir"] = f'{config["model_dir"]}_{model_save_name}'


    # Make directories
    Path(config["A_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["X_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["Y_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["cells_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["graphs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(f'{config["graphs_dir"]}/original').mkdir(parents=True, exist_ok=True)
    Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)

    return config



