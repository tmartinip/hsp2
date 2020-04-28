
def fit(data,k,fold):
    from sklearn.cluster import KMeans
    import timeit 
    import numpy as np
    k += 1
    label = np.ones((data['r'].shape[0],k-2))
    centroid = []  
    X = data['r']
    print(str(X.shape[0]) + " spectra")
    for i in range(2,k):
        t = timeit.default_timer()
        kmeans = KMeans(n_clusters=i,n_init=fold,random_state=0).fit(X)
        label[:,i-2] = (kmeans.labels_+1)
        centroid.append(kmeans.cluster_centers_)
        print( str(i) + " clusrter")
        print(str(np.round(timeit.default_timer() - t,decimals=2)) + " seconds")
    data['km_label'] = label 
    data['km_centroid']=centroid
    return data

def sh(data,k):
    import numpy as np
    import matplotlib.pyplot as plt 
    import matplotlib.colors as colors
    dplot = np.zeros(data['dx']*data['dy'])
    dplot[data['sel']] =data['km_label'][:,k-2]
    dplot =dplot.reshape(data['dx'],data['dy'])
    n = np.max([data['dx'],data['dy']])
    colmap = [ (0,0,0), (1,0,0),(0,1,0),(0,0,1),(0.41,0.41,0.41),(0,1,1),
    (0.58,0,0.82),(0,0.50,0),(0.98,0.50,0.44),(1,	1,0.87),
    (0.39,0.58,0.92),(0.50,0.50,0),(1,0.89,0.76),(0.96,0.96,0.86),
    (0,1,1)]        
    cmap = colors.ListedColormap(colmap)
    boundaries = list(range(15))
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)  
    plt.axes([0.1, 0.1, (data['dy']/n)*0.7,(data['dx']/n)*0.7])
    plt.pcolor(dplot,cmap=cmap, norm=norm)
    plt.axes([0.85, 0.1, 0.05, 0.75])
    plt.pcolor(np.arange(14)[:,None],cmap=cmap, norm=norm)

def spc(data,k):
    import numpy as np
    import matplotlib.pyplot as plt 
    cmap =[ (0,0,0), (1,0,0),(0,1,0),(0,0,1),(0.41,0.41,0.41),(0,1,1),
    (0.58,0,0.82),(0,0.50,0),(0.98,0.50,0.44),(1,	1,0.87),
    (0.39,0.58,0.92),(0.50,0.50,0),(1,0.89,0.76),(0.96,0.96,0.86),
    (0,1,1)]
    for i in list(range(k)):
        sel = data['km_label'][:,k-2] == i+1
        plt.plot(data['wn'],np.mean(data['r'][sel,:],axis=0),color=cmap[i]
        , linewidth=2)