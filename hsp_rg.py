def mean(data,ini1,fim1):
    import numpy as np
    from sklearn.linear_model import  LinearRegression
    import matplotlib.pyplot as plt
    sel = np.logical_and(data['wn'] > int(ini1),data['wn'] < int(fim1))
    r1 = data['r'][:,sel]
    media = np.mean(r1,axis=0).reshape(-1,1)
    meansvalue = np.zeros((data['r'].shape[0],1))
    
    for i in list(range(data['r'].shape[0])):
        reg = LinearRegression().fit(r1[i,:].reshape(-1,1), media)
        meansvalue[i,0] = ( reg.coef_)
    plt.figure()
    plt.hist(meansvalue,1000)   
     
       
def area(data,ini,fim):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = np.logical_and(data['wn'] > int(ini),data['wn'] < int(fim))
    r = data['r'][:,sel]
    area = np.trapz(r)  
    plt.figure()
    plt.hist(area,300)   
    
def intt(data,b):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = data ['wn'] == b
    ver = data['r'][:,sel]
    plt.figure()
    plt.hist(ver,300)      

def emsc(data,a):
    import numpy as np
    import matplotlib.pyplot as plt
    ver = data['emsc_coeff'][:,a]
    plt.figure()
    plt.hist(ver,300)      
