import csv
import numpy as np
from sklearn.neighbors import KDTree

class Ranking():
    def __init__(self, data_path='feature.csv'):
        self.data_path = data_path
    
    def kdtree(self, k=5):
        """
        make a kdtree and retrieve k-nearest neighbors
        do not hold whole data in a memory
        This function is just for a testing purpose (printing a ranking with a small portion of data)
        """

    def linear_search(self, feature, k=5):
        """
        do the brute force search on data
        and find top 5 similar items
        """
        min_dist = []
        with open(self.data_path) as csvfile:
            reader = csv.reader(csvfile)
            for i_r, row in enumerate(reader):
                vector = np.empty([128])
                name = ''
                category = ''
                cnt = 0
                for i_c, col in enumerate(row):
                    if i_c == 0:
                        name = col
                        continue
                    elif i_c == 1:
                        category = col
                        continue
                    elif i_c > 129 or cnt >= 128:
                        break
                    if col == '' or col == ' ':
                        break
                    
                    value = float(col)
                    vector[cnt] = value
                    cnt += 1
                dist = np.linalg.norm(vector-feature)
                if len(min_dist) < 5:
                    min_dist.append({'dist':dist, 'index':i_r, 'name':name})
                    min_dist.sort(key=lambda d: d['dist'], reverse=True)
                    continue
                for i, item in enumerate(min_dist):
                    if dist <= item['dist']:
                        min_dist[i] = ({'dist':dist, 'index':i_r, 'name':name})
                        min_dist.sort(key=lambda d: d['dist'], reverse=True)
                        break
        return min_dist
    
    def store_data(self):
        """
        store data in a database
        """

if __name__ == '__main__':
    r = Ranking(data_path='feature.csv')
