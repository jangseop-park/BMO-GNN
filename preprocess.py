import numpy as np
from sklearn.preprocessing import MinMaxScaler
from spektral.data.dataset import Dataset, Graph
from tqdm import tqdm
from multiprocessing import Pool
import scipy.sparse as sp
import pyvista as pv
import pyacvd
import time


def list_origin_graphs_temp(i):
    nodes, faces = origin_graphs_temp[i].x, origin_graphs_temp[i].cells
    return nodes, faces


def sub_clus(input_tuple):
    points = input_tuple[0]
    cells_ = input_tuple[1]

    cells = np.concatenate((np.full((len(cells_), 1), 3), cells_), axis=1)
    mesh = pv.PolyData(points, cells)

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(config["num_subdivide"])
    clus.cluster(config["num_cluster"])
    # clus.subdivide(1)
    # clus.cluster(1000)
    remesh = clus.create_mesh()

    return np.array(remesh.points), np.array(np.delete(remesh.faces.reshape((-1, 4)), 0, axis=1))

def mesh_qualities(input_tuple):

    points_temp = input_tuple[0]
    cells_ = input_tuple[1]
    cells_temp = np.concatenate((np.full((len(cells_), 1), 3), cells_), axis=1)

    mesh_temp = pv.PolyData(points_temp, cells_temp)
    clus_temp = pyacvd.Clustering(mesh_temp)
    # clus_temp.subdivide(config['num_subdivide'])
    # clus_temp.cluster(config['num_cluster'])
    clus_temp.subdivide(1)
    clus_temp.cluster(1000)
    remesh_temp = clus_temp.create_mesh()

    qual_min_angle = remesh_temp.compute_cell_quality(quality_measure='min_angle')
    qual_jacobian = remesh_temp.compute_cell_quality(quality_measure='scaled_jacobian')
    qual_AR = remesh_temp.compute_cell_quality(quality_measure='aspect_ratio')

    mesh_qualities = {
        'qual_min_angle_min': min(qual_min_angle.cell_data.items()[0][1]),
        'qual_min_angle_avg': np.average(qual_min_angle.cell_data.items()[0][1]),
        'qual_jacobian_min': min(qual_jacobian.cell_data.items()[0][1]),
        'qual_jacobian_avg': np.average(qual_jacobian.cell_data.items()[0][1]),
        'qual_AR_max': max(qual_AR.cell_data.items()[0][1]),
        'qual_AR_avg': np.average(qual_AR.cell_data.items()[0][1])
    }

    return mesh_qualities

def subdivide_graphs(origin_graphs, config_temp):

    global origin_graphs_temp, config
    config = config_temp
    origin_graphs_temp = origin_graphs

    with Pool(64) as p:
        global mesh_qualities_temp
        count_time = time.time()
        mesh_qualities_temp = list(p.map(list_origin_graphs_temp, list(range(925))))
        graphs_list = list(p.map(sub_clus, mesh_qualities_temp))
        # mesh_qualities_list = list(p.map(mesh_qualities, mesh_qualities_temp))

    for i, j in tqdm(enumerate(graphs_list)):
        x_, cells_ = j
        origin_graphs_temp[i].x = x_
        origin_graphs_temp[i].cells = cells_

    # del graphs_list

    print("-----------subdivide elapsed time (sec): ", round(time.time() - count_time, 5))

    return origin_graphs_temp


def list_subdivided_graphs_temp(i):
    nodes, faces = subdivided_graphs_temp[i].x, subdivided_graphs_temp[i].cells
    return nodes, faces


def make_A_X_GCS(input_tuple):
    nodes = input_tuple[0]
    faces = input_tuple[1]
    n = len(nodes)
    adj_mat = np.zeros([n, n])
    faces_ = np.asarray(faces)
    for i in range(n):
        idx, _ = np.where(faces_ == i)
        edge = np.reshape(faces_[idx], (1, faces_[idx].shape[0] * faces_[idx].shape[1]))
        edge = np.unique(edge)
        js = list(edge)
        # js.remove(i)
        for j in js:
            adj_mat[i, j] = np.round(
                ((nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2 + (
                        nodes[i][2] - nodes[j][2]) ** 2) ** (1 / 2), 4)
    return adj_mat, np.array(nodes)


def make_A_X(subdivided_graphs):

    global subdivided_graphs_temp
    subdivided_graphs_temp = subdivided_graphs

    with Pool(64) as p:
        count_time = time.time()
        graphs_list = list(p.map(make_A_X_GCS, list(p.map(list_subdivided_graphs_temp, list(range(925))))))

    for i, j in tqdm(enumerate(graphs_list)):
        a_, x_ = j
        subdivided_graphs_temp[i].a = a_
        subdivided_graphs_temp[i].x = x_

    del graphs_list

    print("-----------make_A_X elapsed time (sec): ", round(time.time() - count_time, 3))

    return subdivided_graphs_temp


class DataTransform(Dataset):
    def __init__(self,graphs,config,**kwargs):
        self.graphs = graphs
        self.config = config
        super().__init__(**kwargs)

    def read(self):
        subdivided_A_X_trans_graphs=[]

        for i in range(925):
            subdivided_A_X_trans_graphs.append(
                Graph(a=self.graphs[i].a,x=self.graphs[i].x,y=self.graphs[i].y))

        return subdivided_A_X_trans_graphs


def minmax_scaler(config,graphs):
    print("-----------start minmax_scaler-------------")

    # save X_scaler and Y_scaler
    count_time = time.time()

    global num_1, num_2
    X = np.empty(shape=(0, 3))
    Y = np.empty(shape=(0, 1))
    if config["label_name"] == "mass":
        num_1, num_2 = 0, 1
    elif config["label_name"] == "rim":
        num_1, num_2 = 1, 2
    elif config["label_name"] == "disk":
        num_1, num_2 = 2, 3

    for i in range(graphs.n_graphs):
        x_, y_ = graphs[i].x, graphs[i].y[0][num_1:num_2].reshape(1, config["label_num"])
        graphs[i].y = y_
        X, Y = np.concatenate((X, x_), axis=0), np.concatenate((Y, y_), axis=0)

    X_scaler, Y_scaler = MinMaxScaler(), MinMaxScaler()
    X_scaler.fit(X)
    Y_scaler.fit(Y)

    for i in range(graphs.n_graphs):
        x_, y_ = graphs[i].x, graphs[i].y
        x_, y_ = X_scaler.transform(x_), Y_scaler.transform(y_)
        graphs[i].x = x_
        graphs[i].y = y_.reshape(1,)

    print("-----------scaler elapsed time (sec): ", round(time.time()-count_time,3))

    return graphs,X_scaler,Y_scaler

# def minmax_scaler(config,graphs):
#     # save X_scaler and Y_scaler
#     global num_1, num_2, X_temp, Y_temp, graphs_temp
#     graphs_temp = graphs
#
#     X_temp = np.empty(shape=(0, 3))
#     Y_temp = np.empty(shape=(0, 1))
#
#     if config["label_name"] == "mass":
#         num_1, num_2 = 0, 1
#     elif config["label_name"] == "rim":
#         num_1, num_2 = 1, 2
#     elif config["label_name"] == "disk":
#         num_1, num_2 = 2, 3
#
#     def concate_X_Y(i):
#         x_, y_ = graphs_temp[i].x, graphs_temp[i].y[0][num_1:num_2].reshape(1, config["label_num"])
#         graphs_temp[i].y = y_
#         X_temp, Y_temp = np.concatenate((X_temp, x_), axis=0), np.concatenate((Y_temp, y_), axis=0)# 스케일용도
#         return X_temp, Y_temp
#
#     X_temp, Y_temp = map(concate_X_Y, list(range(925)))
#
#     X_scaler, Y_scaler = MinMaxScaler(), MinMaxScaler()
#
#     X_scaler.fit(X_temp)
#     Y_scaler.fit(Y_temp)
#
#     del X_temp
#     del Y_temp
#
#     def scaled_X_Y(i):
#         x_, y_ = graphs_temp[i].x, graphs_temp[i].y
#         x_, y_ = X_scaler.transform(x_), Y_scaler.transform(y_)
#         graphs_temp[i].x = x_
#         graphs_temp[i].y = y_.reshape(1,)
#
#     map(scaled_X_Y, list(range(925)))
#
#     return graphs_temp,X_scaler,Y_scaler


def evaluate_mesh_quality(qual):
    _, quality = qual.cell_data.items()[0]
    df_list = []
    df_list.append(qual.n_cells)
    df_list.append(qual.n_points)
    df_list.append(np.min(quality))
    df_list.append(np.max(quality))
    df_list.append(np.mean(quality))
    df_list.append(np.var(quality))
    df_list.append(np.std(quality))
    return df_list

def split_dataset(scaled_dataset,num_tr,num_val):
    # Train/val/test split
    # idxs = np.random.permutation(len(scaled_transformed_graphs))

    idx_num = list(range(925))
    num_val_ = num_tr+num_val

    idx_tr = np.array(idx_num[:num_tr])
    idx_va = np.array(idx_num[num_tr:num_val_])
    idx_te = np.array(idx_num[832:])

    dataset_tr = scaled_dataset[idx_tr]
    dataset_va = scaled_dataset[idx_va]
    dataset_te = scaled_dataset[idx_te]

    dataset = {
        'dataset_tr': dataset_tr,
        'dataset_va': dataset_va,
        'dataset_te': dataset_te,
    }

    return dataset


def padding(scaled_transformed_graphs,mask_bool):
    n_max = max(g.n_nodes for g in scaled_transformed_graphs)
    ## X 특징행렬 패딩
    for i in range(scaled_transformed_graphs.n_graphs):
        Feature_mat = scaled_transformed_graphs[i].x
        n = Feature_mat.shape[0]
        diff = n_max - n
        m=3
        if mask_bool:
            Feature_mat = np.hstack((Feature_mat, np.ones([n,1]))) # mask
            m+=1
        padding = np.vstack((Feature_mat, np.zeros([diff, m])))
        scaled_transformed_graphs[i].x = padding

    ## A 인접행렬 패딩
    for i in range(scaled_transformed_graphs.n_graphs):
        Adj_mat = scaled_transformed_graphs[i].a
        n_i = Adj_mat.shape[0]
        diff = n_max - n_i
        padding = np.pad(Adj_mat, (0, diff), constant_values=0, mode='constant')
        scaled_transformed_graphs[i].a = padding

    return scaled_transformed_graphs # padd_graphs



class _Graph:

    def __init__(self, x=None, a=None, e=None, y=None, cells=None, **kwargs):
        if x is not None:
            if not isinstance(x, np.ndarray):
                raise ValueError(f"Unsupported type {type(x)} for x")
            if len(x.shape) == 1:
                x = x[:, None]
                raise ValueError(f"x was automatically reshaped to {x.shape}")
            if len(x.shape) != 2:
                raise ValueError(
                    f"x must have shape (n_nodes, n_node_features), got "
                    f"rank {len(x.shape)}"
                )
        if a is not None:
            if not (isinstance(a, np.ndarray) or sp.isspmatrix(a)):
                raise ValueError(f"Unsupported type {type(a)} for a")
            if len(a.shape) != 2:
                raise ValueError(
                    f"a must have shape (n_nodes, n_nodes), got rank {len(a.shape)}"
                )
        if e is not None:
            if not isinstance(e, np.ndarray):
                raise ValueError(f"Unsupported type {type(e)} for e")
            if len(e.shape) not in (2, 3):
                raise ValueError(
                    f"e must have shape (n_edges, n_edge_features) or "
                    f"(n_nodes, n_nodes, n_edge_features), got rank {len(e.shape)}"
                )
        self.x = x
        self.a = a
        self.e = e
        self.y = y
        self.cells = cells

        # Read extra kwargs
        for k, v in kwargs.items():
            self[k] = v

    def numpy(self):
        return tuple(ret for ret in [self.x, self.a, self.e, self.y, self.cells] if ret is not None)

    def get(self, *keys):
        return tuple(self[key] for key in keys if self[key] is not None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __contains__(self, key):
        return key in self.keys

    def __repr__(self):
        return "Graph(n_nodes={}, n_node_features={}, n_edge_features={}, n_labels={}, n_cells={})".format(
            self.n_nodes, self.n_node_features, self.n_edge_features, self.n_labels, self.n_cells)

    @property
    def n_nodes(self):
        if self.x is not None:
            return self.x.shape[-2]
        elif self.a is not None:
            return self.a.shape[-1]
        else:
            return None

    @property
    def n_edges(self):
        if sp.issparse(self.a):
            return self.a.nnz
        elif isinstance(self.a, np.ndarray):
            return np.count_nonzero(self.a)
        else:
            return None

    @property
    def n_node_features(self):
        if self.x is not None:
            return self.x.shape[-1]
        else:
            return None

    @property
    def n_edge_features(self):
        if self.e is not None:
            return self.e.shape[-1]
        else:
            return None

    @property
    def n_labels(self):
        if self.y is not None:
            shp = np.shape(self.y)
            return 1 if len(shp) == 0 else shp[-1]
        else:
            return None

    @property
    def n_cells(self):
        if self.cells is not None:
            shp = np.shape(self.cells)
            return 1 if len(shp) == 0 else shp[-1]
        else:
            return None

    @property
    def keys(self):
        keys = [
            key
            for key in self.__dict__.keys()
            if self[key] is not None and not key.startswith("__")
        ]
        return keys

