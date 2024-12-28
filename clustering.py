# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import rc
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기 및 전처리
file_path = './환경부 국립환경과학원_대기오염도 확정자료_20231231.xlsx'
columns_to_use = ['시도', '미세먼지', '초미세먼지']
data = pd.read_excel(file_path, usecols=columns_to_use)

# 결측치 제거 및 음수값 제거
data = data.dropna()
data = data[(data['미세먼지'] >= 0) & (data['초미세먼지'] >= 0)]

# 2. 지역별 통계값 계산
regional_summary = data.groupby('시도').agg({
    '미세먼지': ['mean', 'max', 'min', 'std'],  # 평균, 최대, 최소, 표준편차
    '초미세먼지': ['mean', 'max', 'min', 'std']
}).reset_index()

# 열 이름 정리
regional_summary.columns = ['시도', 
                            'PM10_mean', 'PM10_max', 'PM10_min', 'PM10_std', 
                            'PM2.5_mean', 'PM2.5_max', 'PM2.5_min', 'PM2.5_std']

# 3. K-Means Clustering (군집화)
features = regional_summary[['PM10_mean', 'PM10_std', 'PM2.5_mean', 'PM2.5_std']]
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features)
regional_summary['Cluster'] = clusters

# 4. 시각화: 지역명을 그래프에 표시
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'orange']

for cluster_id in range(3):
    cluster_data = regional_summary[regional_summary['Cluster'] == cluster_id]
    plt.scatter(cluster_data['PM10_mean'], cluster_data['PM2.5_mean'], 
                label=f"Cluster {cluster_id}", s=100, color=colors[cluster_id])
    
    # 지역명 표시
    for _, row in cluster_data.iterrows():
        plt.text(row['PM10_mean'], row['PM2.5_mean'], row['시도'], fontsize=9, ha='right')

# 군집 중심 표시
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 2], c='red', marker='X', s=200, label='Centroids')

plt.title('Cluster Analysis with Region Names')
plt.xlabel('PM10 Mean Concentration (µg/m³)')
plt.ylabel('PM2.5 Mean Concentration (µg/m³)')
plt.legend()
plt.tight_layout()
plt.show()
