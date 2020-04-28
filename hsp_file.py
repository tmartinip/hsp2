# função que importa um arquivo fsm da perkin elmer
def get_fsm_files(path):
    import glob, os
    arq = []
    os.chdir(path)
    for file in glob.glob("*.fsm"):
        arq.append(path + file)
    return arq

def fsm(arq):
    from specio import specread
    import numpy as np
    ver = specread(arq)
    meta = ver.meta
    data = {'r':np.fliplr(ver.amplitudes)}
    data['wn'] = np.flipud(ver.wavelength)
    data['r'] = np.log10(0.01*data['r'])
    data['r'] = -1*data['r']
    dx = data['dy'] = meta['n_x'] 
    dy = data['dx'] = meta['n_y'] 
    data['filename'] = meta['filename'] 
    data['sel'] = np.ones((dx*dy), dtype=bool)
    return data
