  
def intt(datta,b):
    import matplotlib.pyplot as plt
    import numpy as np
    xx = 2
    yy = 2
    sel = datta ['wn'] == b
    ver = datta['r'][:,sel]
    ver = ver[:,0]
    dplot = np.zeros(datta['dx']*datta['dy']);
    dplot[datta['sel']] = ver
    dplot =dplot.reshape(datta['dx'],datta['dy'])
    plt.figure()
    plt.pcolor(dplot)
    plt.show()

## cria uma imagem baseado em area de uma banda 

def area(data,a,b):
    import numpy as np
    import matplotlib.pyplot as plt
    sel1 = (data['wn'] > int(a) )
    sel2 = (data['wn'] < int(b) )
    ver = (sel1.astype(int) + sel2.astype(int))-1
    sel = ver.astype(bool)
    r = data['r'][:,sel]
    area = np.trapz(r)
    dplot = np.zeros(data['dx']*data['dy']);
    dplot[data['sel']] = area
    dplot =dplot.reshape(data['dx'],data['dy'])
    plt.figure()
    plt.pcolor(dplot)
    plt.show()
        
    
def mean(data,ini1,fim1):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import  LinearRegression
    sel = np.logical_and(data['wn'] > int(ini1),data['wn'] < int(fim1))
    r1 = data['r'][:,sel]
    media = np.mean(r1,axis=0).reshape(-1,1)
    meansvalue = np.zeros((data['r'].shape[0]))
    
    for i in list(range(data['r'].shape[0])):
        reg = LinearRegression().fit(r1[i,:].reshape(-1,1), media)
        meansvalue[i] = ( reg.coef_)   
    dplot = np.zeros(data['dx']*data['dy']);
    dplot[data['sel']] = meansvalue
    dplot =dplot.reshape(data['dx'],data['dy'])
    plt.pcolor(dplot)
    plt.figure()
    plt.show()
    
def int_plt(datta,b):
    import matplotlib.pyplot as plt
    import numpy as np
    xx = 2
    yy = 2
    sel = datta ['wn'] ==b
    ver = datta['r'][:,sel]
    data = ver.reshape(datta['dx'],datta['dy'])
    plt.figure(1)
    plt.pcolor(data)
    x = plt.ginput(1)
    xx = int(x[0][0])
    yy = int(x[0][1])
    print(xx,yy)
    while (yy > 10 and xx > 10):
        plt.figure(1)
        plt.pcolor(data)
        x = plt.ginput(1)
        xx = int(x[0][0])
        yy = int(x[0][1])
        verr = np.array ([((xx-1)*datta['dx'])+yy])
        line = np.sum(datta['sel'][0:verr[0]])
        plt.close(2)
        plt.figure(2)
        plt.plot(datta['wn'],datta['r'][line,:])
        print(xx,yy)
        
def pplot(data,nspc):
    import numpy as np
    import matplotlib.pyplot as plt
    r = data['r']
    k = np.random.randint(0,r.shape[0],(nspc),dtype='uint32')
    plt.figure()
    for i in k:
        plt.plot(data['wn'],r[i][:])
    plt.xlabel(' Número de onda')
    plt.show()