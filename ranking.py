import csv
import numpy as np
from sklearn.neighbors import KDTree

class Ranking():
    def __init__(self, data_path='C:/Users/HCIL/InfoVisProject/PolySquare/server/lmvcnn/Features/feature.csv'):
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
    r = Ranking(data_path='C:/Users/HCIL/lmvcnn/caffe_train/Features/feature.csv')
    vector = [0.0765 ,-0.0312 ,0.1596 ,0.0161 ,-0.0523 ,-0.1144 ,0.0318 ,0.0373 ,-0.1138 ,0.1036 ,0.1412 ,-0.0956 ,0.0037 ,-0.0329 ,-0.0352 ,0.1075 ,-0.0518 ,0.1618 ,0.0043 ,0.0944 ,-0.0535 ,0.0149 ,-0.0370 ,-0.0576 ,0.0687 ,-0.0239 ,0.0163 ,0.0558 ,0.0302 ,-0.0180 ,-0.0311 ,0.0768 ,-0.0854 ,-0.0632 ,0.0490 ,-0.0762 ,-0.0033 ,0.0388 ,-0.0315 ,-0.0786 ,0.0311 ,0.0064 ,0.0699 ,0.0900 ,-0.0392 ,0.0258 ,-0.0445 ,0.0048 ,0.0381 ,0.0469 ,0.0438 ,-0.0909 ,0.1549 ,0.0185 ,0.0542 ,0.0206 ,0.1022 ,-0.0077 ,-0.0925 ,-0.1141 ,0.0124 ,0.0136 ,-0.0390 ,0.0527 ,-0.1233 ,-0.0680 ,0.0671 ,-0.0213 ,0.0803 ,-0.0619 ,-0.0450 ,0.0402 ,0.0969 ,0.0375 ,-0.0688 ,0.0467 ,0.0245 ,0.0237 ,-0.1202 ,0.0323 ,0.0971 ,-0.0390 ,-0.0210 ,0.0728 ,0.0695 ,-0.0275 ,-0.0703 ,0.0337 ,-0.0787 ,-0.0460 ,0.0501 ,-0.0807 ,0.1037 ,-0.0573 ,0.0034 ,-0.1736 ,-0.0005 ,0.1548 ,-0.0638 ,0.0448 ,0.0420 ,-0.1179 ,0.1300 ,0.0243 ,0.0556 ,0.0489 ,-0.0139 ,0.0878 ,-0.0787 ,-0.0370 ,0.0141 ,-0.0163 ,0.0519 ,-0.0382 ,0.0994 ,-0.1341 ,-0.1304 ,-0.0485 ,-0.0151 ,-0.0718 ,-0.0092 ,0.0717 ,-0.0298 ,-0.0564 ,-0.0492 ,0.0120 ,-0.0086 ,-0.0638]
    #vector = [-42.5655 ,-97.1187 ,11.6460 ,-96.6869 ,0.9101 ,30.4769 ,-52.1079 ,-82.1451 ,-67.9520 ,-81.3517 ,-53.1972 ,64.0663 ,-91.1955 ,37.4309 ,-56.5758 ,72.5316 ,-35.8349 ,42.6362 ,97.7473 ,-40.2870 ,30.2276 ,-39.9755 ,115.5099 ,95.7821 ,-24.6125 ,-21.2997 ,81.8698 ,-74.8089 ,-36.4983 ,63.4259 ,-63.5867 ,-88.0633 ,-15.5733 ,72.9128 ,30.9571 ,59.7860 ,-43.9954 ,-41.8210 ,76.6809 ,50.2609 ,11.8807 ,-48.2343 ,-5.9486 ,-29.9408 ,20.9381 ,55.6561 ,-50.2363 ,38.8176 ,-12.6181 ,-36.9945 ,65.5446 ,-58.9399 ,-87.5202 ,1.4017 ,104.1516 ,73.7646 ,-34.0576 ,-100.3527 ,-89.0087 ,-39.1493 ,-39.6581 ,-29.9936 ,26.0784 ,67.5247 ,59.7264 ,2.6589 ,-36.2904 ,-11.1568 ,-60.7388 ,71.4340 ,-20.0221 ,37.3974 ,45.2014 ,-84.8438 ,-79.7053 ,-25.1616 ,-79.6815 ,30.5212 ,-42.1868 ,-69.8908 ,77.5353 ,38.5074 ,10.2106 ,59.3581 ,3.4547 ,-21.3193 ,-94.9204 ,-4.3237 ,9.5474 ,39.5432 ,34.4700 ,-26.1978 ,90.2539 ,-71.8593 ,49.1045 ,-98.3675 ,36.0792 ,-92.9800 ,-65.4653 ,-21.9143 ,-69.1713 ,-47.4218 ,37.2721 ,13.2990 ,98.1486 ,72.1711 ,-9.4795 ,-77.7154 ,-113.3739 ,-29.6635 ,-58.9337 ,-57.2837 ,94.6233 ,-31.7681 ,56.9319 ,2.4653 ,91.7978 ,-32.8952 ,-89.9682 ,-9.7197 ,-83.8921 ,2.0682 ,-82.1534 ,-29.8158 ,72.7353 ,5.6759 ,58.6568 ,-79.7993]
    vector = np.asarray(vector)
    result = r.linear_search(vector)
    for re in result:
        print(re)
