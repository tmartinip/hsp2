
def age(path,file):
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    # descobrindo o numero de tiles
    dx = []
    dy = []
    for i in glob.glob(path + file + '*.dmd'):
         i = i[-13:-4].split('_')
         dx.append(int(i[0]))
         dy.append(int(i[1]))
    dx = np.array(dx).max()+1
    dy = np.array(dy).max()+1 
    dados ={}
    dados['dx'] = np.array(32*dx)
    dados['dy'] = np.array(32*dy)
    dados['file'] = np.array(path+file)
    
    # lendo os arquivo dmt construir o vetor wn
    
    arq = path + file + '.dmt'
    data = np.fromfile(arq,dtype='i4')
    npoints = data[559]
    start = data[557]
    data = np.fromfile(arq)
    steps = data[389]
    dados['wn'] = np.linspace(start*steps,(start+npoints)*steps,npoints)
    
    
    # lendo arquivos dmd
    r = np.zeros((int(32*dx),int(32*dy),int(npoints)))   
    arq = glob.glob(path + file + '*.dmd')
    for i in arq:
        y=int(i[-8:-4])
        x=int(i[-13:-9])
        print('x = ',x,'y=',y)
        data = np.fromfile(i,'f4')[255:]
        data = np.reshape(data,(-1,32,32))
        data = np.transpose(data,(2,1,0))
        data = data[:,::-1,:]
        r[32*x:32*(x+1),32*y:32*(y+1),:] = data
    dados['r'] = r.reshape(1024*dx*dy,-1)
    dados['sel'] =  np.ones(dados['r'].shape[0]).astype('bool')
    return dados
