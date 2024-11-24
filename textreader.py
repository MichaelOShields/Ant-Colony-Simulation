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
def return_empty_grid(grid_size):
    empty_grid = []
    for _ in range(grid_size):  # Create physical grid
        row = []
        for _ in range(grid_size): # Create columns
            row.append(0)  # Initialize as undug earth
        empty_grid.append(row)
    return empty_grid

def read_txt(GRID_SIZE,map_name):
    file = open("colony_maps/" + map_name,"r")
    file_string = file.read()
    if len(file_string) > 0:
        file_data = file_string.split(" ")
        file_grid_size = int(file_data[0])  # works
        if GRID_SIZE == file_grid_size:
            grid_string = file_data[1]
            goal_tbl = [[0,0,0],[0,0,0],[1,1,1]]
            test_tbl = [0 for _ in range(file_grid_size**2)]
            test_tbl = []
            for x in range(0,file_grid_size**2,file_grid_size):
                print(max(0,x-1))
                int_list = []
                for y in range(file_grid_size):
                    int_list.append(int(grid_string[y]))
                test_tbl.append(int_list)
                grid_string = grid_string[file_grid_size:]
            return test_tbl 
        else:
            return return_empty_grid(GRID_SIZE)
    else:
        return return_empty_grid(GRID_SIZE)