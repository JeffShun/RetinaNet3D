import numpy as np
import argparse
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_data/processed_data')
    parser.add_argument('--save_path', type=str, default='./train_data/processed_data/anchors.txt')
    parser.add_argument('--n_cluster', type=int, default=3)
    args = parser.parse_args()
    return args

def iou(box, cluster):
    z = np.minimum(cluster[:,0], box[0])
    y = np.minimum(cluster[:,1], box[1])
    x = np.minimum(cluster[:,2], box[2])

    intersection = x * y * z
    area1 = box[0] * box[1] * box[2]
    area2 = cluster[:,0] * cluster[:,1] * cluster[:,2]
    iou = intersection / (area1 + area2 -intersection)
    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(iou(box[i],cluster)) for i in range(box.shape[0])])


def kmeans(boxes, k):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.random.random((rows,))
    np.random.seed()
    clusters = boxes[np.random.choice(rows, k, replace=True)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            cluster_menbers = boxes[nearest_clusters == cluster]
            # 加入异常判断，在有cluster成员时才计算
            if cluster_menbers.size > 0:
                clusters[cluster] = np.median(cluster_menbers, axis=0)
        last_clusters = nearest_clusters

    return clusters


def load_data(data_dir):
    boxes3D = []
    for sample in glob.glob(os.path.join(data_dir, '*.npz')):
        data = np.load(sample, allow_pickle=True)
        z_min, y_min, x_min, z_max, y_max, x_max = data['box3D']
        boxes3D.append([z_max-z_min, y_max-y_min, x_max-x_min])
    return np.array(boxes3D)


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path
    n_cluster = args.n_cluster
    data = load_data(data_path)
    # 使用k聚类算法
    out = kmeans(data,n_cluster) 
    out = out[np.argsort(out[:,0])]
    print(out)
    print('acc:{:.5f}%'.format(avg_iou(data,out) * 100))
    f = open(save_path, 'w')
    for i in range(out.shape[0]):
        if i == 0:
            x_y_z = "%d,%d,%d" % (out[i][0], out[i][1], out[i][2])
        else:
            x_y_z = "\n%d,%d,%d" % (out[i][0], out[i][1], out[i][2])
        f.write(x_y_z)
    f.close()
