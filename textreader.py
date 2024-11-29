# import math
# file = open("map.txt","r")
# file_string = file.read()
# print(len(file_string))
# file_grid = []
# grid_table = file_string.split(" ")
# file_grid_size = int(grid_table[0])
# last = 0
# for x in range(0,file_grid_size ** 2, file_grid_size):
#     row = []
#     for y in range(file_grid_size):
#         file_grid.append(list(file_string[last:x]))
#         last = x
# print(file_grid)

class Node:
    def __init__(self,x,y,node_type,pheremone_types,pheremone_intensities,pheremone_sources):  # x and y in grid, node type, pheremone intensity tuple, pheremone source tuple of (x,y) tuples
        self.x = x  # x index w/in grid
        self.y = y  # y index w/in grid
        self.node_type = node_type  # type of physical node (ground, air, food, ant, etc)
        self.pheremone_types = pheremone_types  # list of pheremone types i.e. [FOOD_PHER,NO_FOOD_PHER]
        self.pheremone_intensities = pheremone_intensities  # list of intensities of the pheremones
        self.pheremone_sources = pheremone_sources  # list of (x,y) tuples of source pheremones
        # self.nonzeropheremonelistindex = None

    # def __eq__(self,other):
    #     # assert isinstance(other,Node)
    #     return self.x == other.x and self.y == other.y

    def remove_pheremone(self,pheremone_type):
        if pheremone_type in self.pheremone_types:
            index = self.pheremone_types.index(pheremone_type)
            self.pheremone_types.pop(index)
            self.pheremone_intensities.pop(index)
            self.pheremone_sources.pop(index)
        if len(self.pheremone_types) == 0:
            # if self.nonzeropheremonelistindex:
            #     print("self in non zero thing!")
            #     nonzero_pheremone_list.pop(self.nonzeropheremonelistindex)
            #     self.nonzeropheremonelistindex = None
            self.pheremone_types.clear()
            self.pheremone_intensities.clear()
            self.pheremone_sources.clear()
            # self.nonzeropheremonelistindex = None
        
    
    def remove_all_pheremones(self):
        # if self in nonzero_pheremone_list:
        #     print("self in non zero thing!")
        #     nonzero_pheremone_list.pop(self.nonzeropheremonelistindex)
        #     self.nonzeropheremonelistindex = None
        self.pheremone_types.clear()
        self.pheremone_intensities.clear()
        self.pheremone_sources.clear()
        # self.nonzeropheremonelistindex = None


def return_empty_grid(grid_size):
    return [[Node(x,y,1,[],[],[]) for x in range(grid_size)] for y in range(grid_size)]

def read_txt(GRID_SIZE,map_name):
    file = open("Ant Colony\\colony_maps\\" + map_name,"r")
    file_string = file.read()
    if len(file_string) > 0:
        file_data = file_string.split(" ")
        file_grid_size = int(file_data[0])  # works
        if GRID_SIZE == file_grid_size:
            grid_string = file_data[1]
            test_tbl = [0 for _ in range(file_grid_size**2)]
            test_tbl = []
            for x in range(0,file_grid_size**2,file_grid_size):
                print(max(0,x-1))
                int_list = []
                for y in range(file_grid_size):
                    int_list.append(int(grid_string[y]))
                test_tbl.append(int_list)
                grid_string = grid_string[file_grid_size:]
            return [[Node(x,y,test_tbl[y][x],[],[],[]) for x in range(file_grid_size)] for y in range(file_grid_size)] 
        else:
            return return_empty_grid(GRID_SIZE)
    else:
        return return_empty_grid(GRID_SIZE)