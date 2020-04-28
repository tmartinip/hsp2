def emsc_base(r,poly_order):
    import numpy as np
    from sklearn.linear_model import  LinearRegression
    from sklearn.decomposition import PCA
    base = np.linspace(-1,1,r.shape[1])
    base = base.reshape(-1,1)
    base = np.tile(base,(1,poly_order+1))
    pot = np.arange(poly_order+1).reshape(1,-1)
    pot = np.tile(pot,(r.shape[1],1))
    base = np.power(base,pot)
    meanspc = np.mean(r,axis=0).reshape(-1,1)
    results = np.zeros_like(r);
    correted = np.zeros_like(r)
    for i in range(r.shape[0]):
        X = np.hstack((meanspc,base))
        reg = LinearRegression(fit_intercept=False).fit(X, r[i,:])
        coeff = np.hstack((reg.intercept_,reg.coef_[1:])).reshape(-1,1)
        X = np.hstack((np.ones((r.shape[1],1)),X[:,1:]))
        results[i,:] = np.dot(X,coeff).reshape(1,-1)
        r[i,:] = r[i,:] - results[i,:] 
    return r



def create_model(dados,parafin,npcs,poly_order):
    import emsc
    import copy as c
    import numpy as np
    from sklearn.decomposition import PCA
    r = c.deepcopy(parafin['r'])
    spc =c.deepcopy(dados['r'])
    wn = dados['wn']   
# se poly_order = 0 para o caso de
# os dados estarem em 2nd derivada    
    if (poly_order == 0):
        sel = np.logical_or(wn<1300,wn>1500)
        r[:,sel] = 0
    
    if (poly_order > 0):
        emsc.emsc_base(r,poly_order)
        
    #Snv - parafin
    media = np.mean(r,axis=1)
    std = np.std(r,axis=1)
    r= np.divide((r - media[:,None]),std[:,None])
        
    ## criando baseline
    base = np.linspace(-1,1,r.shape[1])
    base = base.reshape(-1,1)
    base = np.tile(base,(1,poly_order+1))
    pot = np.arange(poly_order+1).reshape(1,-1)
    pot = np.tile(pot,(r.shape[1],1))
    base = np.power(base,pot)
    
    ## calculando o pca da parafina
    pca = PCA()
    meanparafin = np.mean(r,axis=0)
    #pca.fit(r-meanparafin)
    pca.fit(r)
    
    ## construindo os labels do modelo 
    
    model_labels = ['mean_spc']
    for i in range(poly_order+1):
         a = 'b_' + str(i)
         model_labels.append(a)
     
    model_labels.append('mean_parafin')    
    for i in range(1,npcs+1):
         a = 'Pc_' + str(i)
         model_labels.append(a)     
    
    ## construindo o modelo
    meanparafin = meanparafin.reshape(-1,1)
    meanspc = np.mean(spc,axis=0).reshape(-1,1)
    pca_coeff = pca.components_[0:npcs,:]
    pca_coeff = np.transpose(pca_coeff)
    model = np.hstack((meanspc,base,meanparafin,pca_coeff))
    var = np.cumsum(100*pca.explained_variance_ratio_)
    emsc_model = {'cumvar':var}
    emsc_model['model'] = model
    emsc_model['npcs'] = npcs
    emsc_model['poly_order'] = poly_order
    emsc_model['data_used'] = r
    emsc_model['n_spc_parafin'] = r.shape[0]
    emsc_model['n_spc_target'] = spc.shape[0]
    emsc_model['model_labels']= np.array(model_labels).reshape(1,-1)
    emsc_model['wn'] = dados['wn'] 
    return emsc_model




def view(emsc_model):
    import matplotlib.pyplot as plt
    for i in range(emsc_model['model'].shape[1]):
        plt.figure(i+1)
        plt.plot(emsc_model['wn'],emsc_model['model'][:,i])
        plt.legend(emsc_model['model_labels'][0:,i])
  
    
    
def app_model(dados,emsc_model):
    import numpy as np
    from sklearn.linear_model import  LinearRegression
    import copy as c
    # corrigindo os dados
    r = c.deepcopy(dados['r'])
    r_corrected = np.zeros_like(r)
    results = np.zeros_like(r)
    emsc_coeff = np.zeros((r.shape[0],emsc_model['model'].shape[1]))
    for i in range(r.shape[0]):
        X = emsc_model['model']
        reg = LinearRegression(fit_intercept=False).fit(X, r[i,:])
        emsc_coeff[i,:] = reg.coef_
        coeff = reg.coef_[1:].reshape(-1,1)
        X = X[:,1:]
        results[i,:] = np.dot(X,coeff).reshape(1,-1)
        r_corrected[i,:] = r[i,:] - results[i,:] 
        r_corrected[i,:] = np.divide(r_corrected[i,:],reg.coef_[0])
        print(i)
    dados['emsc_coeff'] = emsc_coeff;
    dados['emsc_model'] = emsc_model['model'];
    dados['emsc_model_labels'] = emsc_model['model_labels'];
    dados['emsc_data_fitted'] = results;
    dados['r'] = r_corrected;
    return dados
        
