from flatland.envs.rail_env import RailEnv
import numpy as np
import networkx as nx
import warnings
from itertools import product

class GraphBuilder:
    def __init__(self, width, height, rail):
        self.transform_dict = {
            0: (0, -1), # West
            1: (1, 0), # South
            2: (0, 1), # East
            3: (-1, 0) # North
        }
        # index: from direction, value to direction
        # self.valid_transition = [2, 3, 0, 1]
        self.bit_index_to_dir = {ind:self.convert_bit_ind_2_dir(ind) for ind in range(16)}

        self.rail = rail
        self.width = width
        self.height = height
        # initialize self.cell_transitions
        self.grid2cells()

    def transform_indexes(self, i, j, transition):
        tr = self.transform_dict[transition % 4]
        return i + tr[0], j + tr[1]

    def get_cell_dirs(self, i, j):
        transitions = self.rail[i, j]
        res = []
        bit = 1 
        for bit_ind in range(0, 16):
            if transitions & bit != 0:
                res.append(self.bit_index_to_dir[bit_ind])
            bit = bit << 1
        return res

    def grid2cells(self):
        # param: env.rail.grid
        self.cell_transitions = []
        for i in range(self.height):
            for j in range(self.width):
                transitions = self.rail[i, j]
                bit = 1 
                for bit_ind in range(0, 16):
                    if transitions & bit != 0:
                        neighbor_inds = self.transform_indexes(i, j, bit_ind)
                        self.cell_transitions.append(((i,j), neighbor_inds, self.convert_bit_ind_2_dir(bit_ind)))
                    bit = bit << 1

        # filter dublicates
        # todo
        # return cell_transitions

    def convert_bit_ind_2_dir(self, bit_index):
        complementer = 15 - bit_index
        dir0 = complementer // 4
        dir1 = complementer % 4
        assert(dir0 >= 0 and dir1 >=0)
        if dir0 < dir1:
            return dir0, dir1
        else:
            return dir1, dir0

    def convert_indexes_2_node(self, inds):
        return inds[0]*self.width + inds[1]

    def convert_node_2_indexes(self, node):
        return node // self.width, node % self.width

    def graph_from_cell_neighbors(self, whitelist=None):
        if whitelist is None:
            whitelist = set()
        g = nx.Graph()
        g.add_edges_from([(self.convert_indexes_2_node(x), 
                self.convert_indexes_2_node(y),
                {'dir0':dirs[0], 'dir1':dirs[1]}
            ) for x, y, dirs in self.cell_transitions])
        print(g.number_of_nodes(), g.number_of_edges())

        nodes2remove = set(node for node in g.nodes() if len(g[node]) == 2 and node not in whitelist)
        for node in nodes2remove:
            e = tuple(g[node])
            dirs0 = g[node][e[0]]["dir0"], g[node][e[0]]["dir1"]
            dirs1 = g[node][e[1]]["dir0"], g[node][e[1]]["dir1"]
            paths = self.get_cell_dirs(*self.convert_node_2_indexes(node))

            new_dirs = (-1, -1)
            for i, j in product(dirs0, dirs1):
                # whatever
                if i == j and (i, j) in paths:
                    if i < j:
                        new_dirs = (i, j)
                    else:
                        new_dirs = (j, i)
                    break

            # if dirs0[0] == dirs1[0]:
            #     new_dirs = (dirs0[1], dirs1[1])
            # elif dirs0[0] == dirs1[1]:
            #     new_dirs = (dirs0[1], dirs1[0])
            # elif dirs0[1] == dirs1[0]:
            #     new_dirs = (dirs0[0], dirs1[1])
            # elif dirs0[1] == dirs1[1]:
            #     new_dirs = (dirs0[0],dirs1[0])
            if new_dirs[0] >= 0 and new_dirs[1] >= 0:
                g.add_edge(e[0], e[1], dir0=new_dirs[0], dir1=new_dirs[1])
                g.remove_node(node)
            else:
                warnings.warn(f"Could not join edges ({node}, {e[0]}) ({node}, {e[1]}) - directions do not match")
        return g
