
## recorta o especto entre a e b
        
def cut(data,a,b):
    sel1 = (data['wn'] > a )
    sel2 = (data['wn'] < b )
    ver = (sel1.astype(int) + sel2.astype(int))-1
    sel = ver.astype(bool)
    data['r'] = data['r'][:,sel]
    data['wn'] = data['wn'][sel]
    return data

## plota o histograma de area da imagem

def norm(data):
    import numpy as np
    spc = data['r']
    media = np.mean(spc,axis=1)
    std = np.std(spc,axis=1)
    data['r'] = np.divide((spc - media[:,None]),std[:,None])
    return data


def golay(data,diff,order,win):
    import numpy as np
    from scipy.signal import savgol_coeffs
    from scipy.sparse import spdiags
    import numpy.matlib
    n = int((win-1)/2)
    sgcoeff = savgol_coeffs(win, order, deriv=diff)[:,None]
    sgcoeff = np.matlib.repmat(sgcoeff,1,data['r'].shape[1])
    diags = np.arange(-n,n+1)
    D = spdiags(sgcoeff,diags,data['r'].shape[1],data['r'].shape[1]).toarray()
    D[:,0:n] = 0
    D[:,data['r'].shape[1]-5:data['r'].shape[1]] = 0
    data['r'] = np.dot(data['r'],D)
    return data

       
def norm2r(data,ini1,fim1,ini2,fim2):
    import numpy as np
    sel = np.logical_and(data['wn'] > int(ini1),data['wn'] < int(fim1))
    r1 = data['r'][:,sel]
    wn1 = data['wn'][sel][:,None]
    media = np.mean(r1,axis=1)
    std = np.std(r1,axis=1)
    r1 = np.divide((r1 - media[:,None]),std[:,None])
            
    sel = np.logical_and(data['wn'] > int(ini2),data['wn'] < int(fim2))
    r2 = data['r'][:,sel]
    wn2 = data['wn'][sel][:,None]
    media = np.mean(r2,axis=1)
    std = np.std(r2,axis=1)
    r2 = np.divide((r2 - media[:,None]),std[:,None])
    data['r'] = np.column_stack((r1,r2))
    data['wn'] = np.vstack((wn1,wn2))
    data['wn'] = data['wn'].reshape(-1)
    return data                

def pcares(data,n):
    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA()
    media = np.mean(data['r'],axis=0)
    pca.fit(data['r']-media)
    scoress = pca.transform(data['r'])
    scoress[:,n-1:-1] = 0 
    coeff= pca.components_
    data['r'] =media + np.dot(scoress,coeff)
    return data

def offset(data,ini,fim):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = np.logical_and(data['wn'] > int(ini),data['wn'] < int(fim));
    r = data['r'][:,sel];
    minino = np.min(r,axis=1);
    minino = np.reshape(minino,(-1,1));
    minino = np.tile(minino,data['r'].shape[1]);
    data['r'] = data['r']-minino;
    return data

def binned(data):
    import numpy as np
    import matplotlib.pyplot as plt    
    r = data['r']
    r = r.reshape(data['dx'],data['dy'],-1)
    dx = r.shape[0]
    dy = r.shape[1]
    dz = r.shape[2]
    dxbin = int(np.floor(dx/2))+2
    dybin = int(np.floor(dy/2))+2
    rbin = np.ones((dxbin,dybin,dz))
    jj = 0
    ii = 0
    for i in range(0,dy-2,2):
        for j in range(0,dx-2,2):
            sel = r[j:j+2,i:i+2,:];
            sel  = np.mean(sel.reshape(4,dz),axis=0)
            rbin[jj,ii,:] = sel
            jj = jj + 1
            
        jj = 0
        ii = ii + 1
    data['r'] = rbin.reshape((dxbin*dybin,dz)) 
    data['sel'] = np.ones((dxbin*dybin,)).astype('bool')
    data['dx'] = dxbin
    data['dy'] = dybin
    
    return data

