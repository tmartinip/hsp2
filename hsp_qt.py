def area(data,ini,fim,a,b):
    import numpy as np
    sel = np.logical_and(data['wn'] > int(ini),data['wn'] < int(fim))
    r = data['r'][:,sel]
    area = np.trapz(r)  
    sel = np.logical_and(area > float(a),area < float(b))
    data['r'] = data['r'][sel,:]   
    data['sel'][data['sel']] = (sel) 
    return data

def intt(data,ini,a,b):
    import numpy as np
    import matplotlib.pyplot as plt
    sel = data ['wn'] == ini
    area = data['r'][:,sel]
    sel = np.logical_and(area > float(a),area < float(b))
    sel = np.reshape(sel,(-1,))
    data['r'] = data['r'][sel,:] 
    data['sel'][data['sel']] = (sel) 
    return data

def emsc(data,ini,a,b):
    import numpy as np
    import matplotlib.pyplot as plt
    area = data['emsc_coeff'][:,ini];
    sel = np.logical_and(area > float(a),area < float(b))
    sel = np.reshape(sel,(-1,))
    data['r'] = data['r'][sel,:] 
    data['sel'][data['sel']] = (sel) 
    return data

def dsample(data):
    import numpy as np
    n = 2
    sel = np.ones((data['dx'],data['dy']))
    XX = list(range(0,sel.shape[0]-1,n));
    YY = list(range(0,sel.shape[1]-2,n));
    sel[XX,:] = 0
    sel[:,YY] = 0
    sel = sel.reshape(-1,)
    sel = sel.astype('bool')
    data['r'] = data['r'][sel,:] 
    data['sel'][data['sel']] = (sel) 
    return data

