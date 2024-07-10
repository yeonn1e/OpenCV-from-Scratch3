import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import joblib 

"""
파일 저장을 위한 함수: open_f, compute
수행 시간 단축을 위해 해당 코드는 일단 주석 처리
저장된 pkl 파일을 불러와 실행하는 코드를 평가시 사용

혹시나 pkl 파일 등의 실행 오류가 발생한다면 실제 compute 함수 실행해주시면 감사하겠습니다.
compute 함수 실행 시간은 1분 이내로 빨리 끝납니다.
"""

def open_f(i, sift_path, cnn_path):
    sift_file = os.path.join(sift_path, f'{i:04d}.sift')
    with open(sift_file, 'rb') as f:
        file = np.fromfile(f, dtype=np.uint8)
    sdata = file.reshape((file.size // 128, 128))

    cnn_file = os.path.join(cnn_path, f'{i:04d}.cnn')
    with open(cnn_file, 'rb') as f:
        file = np.fromfile(f, dtype=np.float32)
    cdata = file.reshape((14, 14, 512))
    return sdata, cdata

'''def compute(dir, des_file, n_clusters=31, pca_cmp=64, batch_size=20000, 
         max_iter=500, random_state=42):
    """
    파일 저장: kmeans_sift.pkl, pca_cnn.pkl, A4.des
    """
    sift_path = os.path.join(dir, 'sift')
    cnn_path = os.path.join(dir, 'cnn')
    dim = n_clusters * 128 + pca_cmp
    des = np.zeros((2000, dim), dtype=np.float32)

    # sift -> kmeans->vlad, cnn -> pca
    # 실행 시간 단축 위해 미리 저장할 npy files: kmeans, pca
    sifts = list()
    cnns = list()

    for i in range(2000): # image 수만큼 반복
        sdata, cdata = open_f(i, sift_path, cnn_path)
        sifts.append(sdata)
        cnns.append(cdata.reshape(-1, 512))

    sifts = np.vstack(sifts)
    cnns = np.vstack(cnns)

    pca = PCA(n_components=pca_cmp)
    pca.fit(cnns)
    # 1. pca cnn 저장 -> 평가시에는 사용하지 않음.
    joblib.dump(pca, 'pca_cnn.pkl')

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        verbose=1,
        init='k-means++',
        max_iter=max_iter,
        random_state=random_state
    ).fit(sifts)
    # 2. kmeans sift 저장 -> 평가시에는 사용하지 않음.
    joblib.dump(kmeans, 'kmeans_sift.pkl')

    # 3. des 파일 저장
    for i in range(2000): # image 수만큼 반복
        sdata, cdata = open_f(i, sift_path, cnn_path)

        # VLAD
        base = np.zeros((kmeans.n_clusters, sdata.shape[1]), dtype=np.float32)
        for idx, lab in enumerate(kmeans.predict(sdata)):
            base[lab] += sdata[idx] - kmeans.cluster_centers_[lab]
        v = base.flatten()
        vlad = (np.sign(v) * np.sqrt(np.abs(v)))
        vlad /= np.linalg.norm(vlad)
        desvlad = vlad

        descnn = pca.transform(cdata.reshape(-1, 512)).mean(axis=0)

        descriptors = np.concatenate((desvlad, descnn))
        des[i, :] = descriptors

    with open(des_file, 'wb') as f:
        np.array([2000], dtype=np.int32).tofile(f)
        np.array([dim], dtype=np.int32).tofile(f)
        des.tofile(f)
'''

def test(dir, des_file, pca_file, kmeans_file, n_clusters=31, pca_cmp=64, batch_size=20000, 
         max_iter=500, random_state=42):
    """
     1-2-b) precomputed vectors file을 불러와서 사용하는 코드
    """
    sift_path = os.path.join(dir, 'sift')
    cnn_path = os.path.join(dir, 'cnn')
    dim = n_clusters * 128 + pca_cmp
    des = np.zeros((2000, dim), dtype=np.float32)

    # 1. pca_cnn.pkl 파일 불러옴
    pca = joblib.load(pca_file)

    # 2. kmeans_sift.pkl 파일 불러옴
    kmeans = joblib.load(kmeans_file)

    for i in range(2000):
        sdata, cdata = open_f(i, sift_path, cnn_path)

        # VLAD
        base = np.zeros((kmeans.n_clusters, sdata.shape[1]), dtype=np.float32)
        for idx, lab in enumerate(kmeans.predict(sdata)):
            base[lab] += sdata[idx] - kmeans.cluster_centers_[lab]
        v = base.flatten()
        vlad = (np.sign(v) * np.sqrt(np.abs(v)))
        vlad /= np.linalg.norm(vlad)
        vlad_descriptor = vlad

        cnn_descriptor = pca.transform(cdata.reshape(-1, 512)).mean(axis=0)

        combined_descriptor = np.concatenate((vlad_descriptor, cnn_descriptor))
        des[i, :] = combined_descriptor 

    with open(des_file, 'wb') as f:
        np.array([2000], dtype=np.int32).tofile(f)
        np.array([dim], dtype=np.int32).tofile(f)
        des.tofile(f)
     
if __name__ == '__main__':
    # pkl, des 파일 저장을 위한 코드를 실행해야 한다면 아래 코드 주석 해제
    #compute('./features', 'A4.des')

    # 저장된 파일을 불러와 평가시 사용하는 코드
    test('./features', 'A4.des', 'pca_cnn.pkl', 'kmeans_sift.pkl')