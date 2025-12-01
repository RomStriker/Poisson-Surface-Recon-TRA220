from src.io.load_pointcloud import load_ply_points_normals



if __name__ == '__main__':
    data_path = './data/bunny.ply'
    points, normals = load_ply_points_normals(data_path)

    print('Data Loaded')


