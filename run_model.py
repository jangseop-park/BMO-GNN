from spektral.layers import GCNConv, GCSConv, GlobalAvgPool, MinCutPool
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt.util import UtilityFunction

import argparse
import time
import random
import pickle
import os
import copy

# build custom modules
from config import load_config
from utils import (model_train_eval, str2bool, model_test, write_results_of_each_model, write_results_of_bayesian)
from preprocess import (subdivide_graphs, make_A_X, minmax_scaler, split_dataset, DataTransform)
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from models import (gcn3, gcs3, gcn4, gcn15, gcn_custom, gcs_mincut)

parser = argparse.ArgumentParser()
parser.add_argument('--transfer_learning', type=str2bool, default=False)
parser.add_argument('--transfer_best_or_prev', type=str2bool, default=False) # In case of TR
parser.add_argument('--botry_retry', type=str2bool, default=True)
parser.add_argument('--add_initial_cluster', type=str2bool, default=True)
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--mode_two', type=str, default="BO") # BO or LHS or BO_10,~50
parser.add_argument('--data_start', type=int, default=0) # LHS mode 0~10, 10~20,
parser.add_argument('--data_end', type=int, default=10) # LHS mode

parser.add_argument('--mesh_ratio', type=float, default=0.005)
parser.add_argument('--label_num', type=int, default=1)
parser.add_argument('--model_name', type=str, default="gcs3")
parser.add_argument('--trans_name', type=str, default="GCSConv")
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--num_subdivide', type=int, default=None)
parser.add_argument('--num_cluster', type=int, default=None)
parser.add_argument('--iteration', type=int, default=0)

parser.add_argument('--num_tr', type=int, default=740)
parser.add_argument('--num_val', type=int, default=92)

## rim
parser.add_argument('--label_name', type=str, default="rim")
parser.add_argument('--best_iteration', type=int, default=34)
parser.add_argument('--best_model_num_subdivide', type=int, default=3)
parser.add_argument('--best_model_num_cluster', type=int, default=4626)
parser.add_argument('--best_model_loss', type=float, default=0.0003)
parser.add_argument('--best_model_r', type=float, default=0.993)

## disk
# parser.add_argument('--label_name', type=str, default="disk")
# parser.add_argument('--best_iteration', type=int, default=23)
# parser.add_argument('--best_model_num_subdivide', type=int, default=3)
# parser.add_argument('--best_model_num_cluster', type=int, default=3438)
# parser.add_argument('--best_model_loss', type=float, default=0.00093)
# parser.add_argument('--best_model_r', type=float, default=0.979)

## mass
# parser.add_argument('--label_name', type=str, default="mass")
# parser.add_argument('--best_iteration', type=int, default=11)
# parser.add_argument('--best_model_num_subdivide', type=int, default=3)
# parser.add_argument('--best_model_num_cluster', type=int, default=4557)
# parser.add_argument('--best_model_loss', type=float, default=0.00053)
# parser.add_argument('--best_model_r', type=float, default=0.984)
args = parser.parse_args()

global trans, config
if args.trans_name == "GCSConv":
    trans = GCSConv
elif args.trans_name == "GCNConv":
    trans = GCNConv

# set config
config = load_config(
    # select transfer learning or not
    transfer_learning=args.transfer_learning,
    transfer_best_or_prev=args.transfer_best_or_prev,

    # train or test parameters
    mode=args.mode,
    mode_two=args.mode_two,
    mesh_ratio=args.mesh_ratio,
    label_num=args.label_num,
    label_name=args.label_name,
    model_name=args.model_name,
    trans_name=args.trans_name,
    trans=trans,

    # train parameters
    epochs=args.epochs,
    patience=args.patience,
    learning_rate=args.learning_rate,
    iteration=args.iteration,

    # test parameters
    # - select subdivided data
    num_subdivide=args.num_subdivide,     # best num_subdivide for test
    num_cluster=args.num_cluster,       # best num_cluster for test

    # - select best model
    best_iteration=args.best_iteration,
    best_model_num_subdivide=args.best_model_num_subdivide,
    best_model_num_cluster=args.best_model_num_cluster,
    best_model_loss=args.best_model_loss,
    best_model_r=args.best_model_r,

)

# set MODELS parameters
MODELS = {
    'gcs_mincut': dict(dim=512, dense1=500, dense2=200, dense3=25, model=gcs_mincut,
                  custom_objects={"GCSConv": GCSConv, "GlobalAvgPool": GlobalAvgPool, "MinCutPool": MinCutPool}),
    'gcn3': dict(dim=128, dense1=300, dense2=100, dense3=25, model=gcn3,
                  custom_objects={"GCNConv": GCNConv, "GlobalAvgPool": GlobalAvgPool}),
    'gcs3': dict(dim=512, dense1=500, dense2=200, dense3=25, model=gcs3,
                  custom_objects={"GCSConv": GCSConv, "GlobalAvgPool": GlobalAvgPool}),
    'gcn4': dict(dim=512, dense1=300, dense2=100, dense3=50, model=gcn4,
                  custom_objects={"GCNConv": GCNConv, "GlobalAvgPool": GlobalAvgPool}),
    'gcn15': dict(dim=512, dense1=500, dense2=300, dense3=25, model=gcn15,
                   custom_objects={"GCNConv": GCNConv, "GlobalAvgPool": GlobalAvgPool}),
    'gcn_custom': dict(dim=512, dense1=300, dense2=150, dense3=25, model=gcn_custom,
                   custom_objects={"GCNConv": GCNConv, "GlobalAvgPool": GlobalAvgPool})
}

params = MODELS[config["model_name"]]


def check_previous_graph(config):
    config["make_subdivided_graph_bool"] = True

    if config["make_subdivided_graph_bool"]:
        names = [file[:-4] for file in os.listdir(config["graphs_dir"]) if file[-4:] == '.pkl']
        for name in names:
            if name == f'subdivided_{config["num_subdivide"]}_{config["num_cluster"]}':
                with open(f'{config["graphs_dir"]}/{name}.pkl', 'rb') as f:
                    load_graphs = pickle.load(f)
                    config["make_subdivided_graph_bool"] = False

    if config["make_subdivided_graph_bool"]:
        with open(f'{config["graphs_dir"]}/graphs.pkl', 'rb') as f:
            load_graphs = pickle.load(f)

    return load_graphs

def main():
    # Load data
    if config["mode_two"] == 'BO' or config["mode_two"] == 'BO_10' or config["mode_two"] == 'BO_20' or config["mode_two"] == 'BO_30' or config["mode_two"] == 'BO_40' or config["mode_two"] == 'BO_50'or config["mode_two"] == 'BO_RE':
        if config["mode"] == 'train':
            bayesian_results = []
            one_iter_total_times = []
            one_iter_train_times = []
            total_iter_times = []

            total_start_time = time.time()

            def bayesian_optimize_subdivide(num_subdivide, num_cluster):
                one_iter_total_start_time = time.time()

                # update num_subdivide and num_cluster
                config['num_subdivide'] = int(round(num_subdivide))
                config['num_cluster'] = int(round(num_cluster))
                print('num_subdivide: ', config['num_subdivide'])
                print('num_cluster: ', config['num_cluster'])

                # load graphs data
                load_graphs = check_previous_graph(config)

                if config["make_subdivided_graph_bool"]:
                    # subdivide mesh
                    subdivided_graphs = subdivide_graphs(load_graphs, config)
                    subdivided_A_X_graphs = make_A_X(subdivided_graphs)

                    count_time = time.time()
                    subdivided_A_X_trans_graphs = DataTransform(subdivided_A_X_graphs, config,transforms=[LayerPreprocess(config["trans"]),AdjToSpTensor()])
                    print("-----------transform elapsed time (sec): ", round(time.time() - count_time, 3))

                    with open(f'{config["graphs_dir"]}/subdivided_{config["num_subdivide"]}_{config["num_cluster"]}.pkl','wb') as f:
                        pickle.dump(subdivided_A_X_trans_graphs, f)

                    scaled_dataset, X_scaler, Y_scaler = minmax_scaler(config, subdivided_A_X_trans_graphs)

                else:
                    subdivided_A_X_trans_graphs = load_graphs
                    scaled_dataset, X_scaler, Y_scaler = minmax_scaler(config, subdivided_A_X_trans_graphs)

                # save X_scaler and Y_scaler
                config['X_scaler'], config['Y_scaler'] = X_scaler, Y_scaler

                # split dataset
                dataset = split_dataset(scaled_dataset,args.num_tr,args.num_val)

                # model train
                one_iter_train_start_time = time.time()
                r_score = model_train_eval(dataset, config, params)

                # results of bayesian optimization
                bayesian_results.append(config['model_load_name'])

                # calculate times
                one_iter_total_time = (time.time() - one_iter_total_start_time)/60
                one_iter_train_time = (time.time() - one_iter_train_start_time)/60
                total_iter_time = (time.time() - total_start_time)/60

                print('one_iter_total_time (min.): ', one_iter_total_time)
                print('one_iter_train_time (min.): ', one_iter_train_time)
                print("total_iter_time (min.) :", total_iter_time)

                one_iter_total_times.append(one_iter_total_time)
                one_iter_train_times.append(one_iter_train_time)
                total_iter_times.append(total_iter_time)

                # write results of BO and times
                write_results_of_each_model(config, params, bayesian_results, one_iter_total_times, one_iter_train_times, total_iter_times)

                return r_score

            bounds = {'num_subdivide': (2.5, 4.49), 'num_cluster': (3000, 5000)}
            print('pbounds: ', bounds)

            if args.botry_retry:
                bo = BayesianOptimization(f=bayesian_optimize_subdivide, pbounds=bounds, verbose=2, random_state=1)
                logger = JSONLogger(path=f"{config['model_dir']}/logs.json")
                bo.subscribe(Events.OPTIMIZATION_STEP, logger)

                # UtilityFunction을 사용하여 acquisition function 설정
                utility = UtilityFunction(kind="ei", xi=0.01)
                bo.maximize(init_points=3, n_iter=300, acquisition_function=utility)
                print(bo.max)

            else:
                new_optimizer = BayesianOptimization(f=bayesian_optimize_subdivide, pbounds=bounds, verbose=2, random_state=1)
                load_logs(new_optimizer, logs=[f"{config['model_dir']}/logs.json"])
                print("New optimizer is now aware of {} points.".format(len(new_optimizer.space)))
                config['iteration'] = len(new_optimizer.space)
                logger = JSONLogger(path=f"{config['model_dir']}/logs.json")
                new_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

                # UtilityFunction을 사용하여 acquisition function 설정
                utility = UtilityFunction(kind="ei", xi=0.01)
                new_optimizer.maximize(init_points=0, n_iter=300, acquisition_function=utility)

                print(new_optimizer.max)

            # write the final results of bayesian optimization
            # write_results_of_bayesian(config, bayesian_results)



        elif config["mode"] == 'test':

            # subdivide mesh
            subdivided_x_cells_graphs = subdivide_graphs(origin_graphs, config)
            subdivided_trans_graphs = LoadDataset(config, subdivide_bool=True, graphs=subdivided_x_cells_graphs)
            scaled_dataset, X_scaler, Y_scaler = minmax_scaler(config, subdivided_trans_graphs)

            # save X_scaler and Y_scaler
            config['X_scaler'], config['Y_scaler'] = X_scaler, Y_scaler

            # split dataset
            dataset = split_dataset(scaled_dataset,args.num_tr,args.num_val)

            # model test
            model_test(dataset, config, params)

    elif config['mode_two'] == 'MC':
        bayesian_results = []
        one_iter_total_times = []
        one_iter_train_times = []
        total_iter_times = []

        total_start_time = time.time()

        for i in range(300):
            one_iter_total_start_time = time.time()

            # update num_subdivide and num_cluster
            config['num_subdivide'] = random.randint(2,4)
            config['num_cluster'] = random.randint(3000,5000)

            print('random.randint(2,4)')
            print('random.randint(3000,5000)')


            # update num_subdivide and num_cluster
            print('num_subdivide: ', config['num_subdivide'])
            print('num_cluster: ', config['num_cluster'])

            load_graphs = check_previous_graph(config)

            if config["make_subdivided_graph_bool"]:
                # subdivide mesh
                subdivided_graphs = subdivide_graphs(load_graphs, config)
                subdivided_A_X_graphs = make_A_X(subdivided_graphs)

                count_time = time.time()
                subdivided_A_X_trans_graphs = DataTransform(subdivided_A_X_graphs, config, transforms=[LayerPreprocess(config["trans"]), AdjToSpTensor()])
                print("-----------transform elapsed time (sec): ", round(time.time() - count_time, 3))

                with open(f'{config["graphs_dir"]}/subdivided_{config["num_subdivide"]}_{config["num_cluster"]}.pkl','wb') as f:
                    pickle.dump(subdivided_A_X_trans_graphs, f)

                scaled_dataset, X_scaler, Y_scaler = minmax_scaler(config, subdivided_A_X_trans_graphs)

            else:
                subdivided_A_X_trans_graphs = load_graphs
                scaled_dataset, X_scaler, Y_scaler = minmax_scaler(config, subdivided_A_X_trans_graphs)

            # save X_scaler and Y_scaler
            config['X_scaler'], config['Y_scaler'] = X_scaler, Y_scaler

            # split dataset
            dataset = split_dataset(scaled_dataset, args.num_tr, args.num_val)

            # model train
            one_iter_train_start_time = time.time()
            r_score = model_train_eval(dataset, config, params)

            # results of bayesian optimization
            bayesian_results.append(config['model_load_name'])

            # calculate times
            one_iter_total_time = (time.time() - one_iter_total_start_time) / 60
            one_iter_train_time = (time.time() - one_iter_train_start_time) / 60
            total_iter_time = (time.time() - total_start_time) / 60

            print('one_iter_total_time (min.): ', one_iter_total_time)
            print('one_iter_train_time (min.): ', one_iter_train_time)
            print("total_iter_time (min.) :", total_iter_time)

            one_iter_total_times.append(one_iter_total_time)
            one_iter_train_times.append(one_iter_train_time)
            total_iter_times.append(total_iter_time)

            # qual_min_angle_mins.append(mesh_qualities['qual_min_angle_min'])
            # qual_min_angle_avgs.append(mesh_qualities['qual_min_angle_avg'])
            # qual_jacobian_mins.append(mesh_qualities['qual_jacobian_min'])
            # qual_jacobian_avgs.append(mesh_qualities['qual_jacobian_avg'])
            # qual_AR_maxs.append(mesh_qualities['qual_AR_max'])
            # qual_AR_avgs.append(mesh_qualities['qual_AR_avg'])

            # write results of BO and times
            write_results_of_each_model(config, params, bayesian_results, one_iter_total_times, one_iter_train_times, total_iter_times)


if __name__ == '__main__':
    main()
