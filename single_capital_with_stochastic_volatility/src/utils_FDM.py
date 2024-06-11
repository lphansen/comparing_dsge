import numpy as np

def finiteDiff_3D(data, dim, order, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 3:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / (2 * dlt)) * (data[2:,:,:] - data[:-2,:,:])
                res[-1,:,:] = (1 / dlt) * (data[-1,:,:] - data[-2,:,:])
                res[0,:,:] = (1 / dlt) * (data[1,:,:] - data[0,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / (2 * dlt)) * (data[:,2:,:] - data[:,:-2,:])
                res[:,-1,:] = (1 / dlt) * (data[:,-1,:] - data[:,-2,:])
                res[:,0,:] = (1 / dlt) * (data[:,1,:] - data[:,0,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,2:] - data[:,:,:-2])
                res[:,:,-1] = (1 / dlt) * (data[:,:,-1] - data[:,:,-2])
                res[:,:,0] = (1 / dlt) * (data[:,:,1] - data[:,:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
                res[-1,:,:] = (1 / dlt ** 2) * (data[-1,:,:] + data[-3,:,:] - 2 * data[-2,:,:])
                res[0,:,:] = (1 / dlt ** 2) * (data[2,:,:] + data[0,:,:] - 2 * data[1,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
                res[:,-1,:] = (1 / dlt ** 2) * (data[:,-1,:] + data[:,-3,:] - 2 * data[:,-2,:])
                res[:,0,:] = (1 / dlt ** 2) * (data[:,2,:] + data[:,0,:] - 2 * data[:,1,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
                res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-1] + data[:,:,-3] - 2 * data[:,:,-2])
                res[:,:,0] = (1 / dlt ** 2) * (data[:,:,2] + data[:,:,0] - 2 * data[:,:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')
    elif l == 2:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])
                res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
                res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
                res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
                res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
                res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
                res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
                res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
                res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')

            
    else:
        raise ValueError("Dimension NOT supported")
        
    if cap is not None:
        res[res < cap] = cap
    return res

def finiteDiff_1D_first(data, dim, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    res[1:-1] = (1 / (2 * dlt)) * (data[2:] - data[:-2])
    res[0] = (1 / (dlt)) * (data[1] - data[0])
    res[-1] = (1 / (dlt)) * (data[-1] - data[-2])
    return res

def finiteDiff_1D_second(data, dim, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    res[1:-1] = (1 / dlt ** 2) * (data[2:] + data[:-2] - 2 * data[1:-1])
    res[-1] = (1 / dlt ** 2) * (data[-1] + data[-2] - 2 * data[-1])
    res[0] = (1 / dlt ** 2) * (data[1] + data[0] - 2 * data[0])
    return res

def finiteDiff_2D_first(data, dim, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    if dim == 0:                  # to first dimension

        res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])

    elif dim == 1:                # to second dimension

        res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
    return res

def finiteDiff_3D_second(data, dim, dlt, cap = None):  

    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    if dim == 0:                  # to first dimension
        res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
        res[-1,:,:] = (1 / dlt ** 2) * (data[-1,:,:] + data[-2,:,:] - 2 * data[-1,:,:])
        res[0,:,:] = (1 / dlt ** 2) * (data[1,:,:] + data[0,:,:] - 2 * data[0,:,:])

    elif dim == 1:                # to second dimension

        res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
        res[:,-1,:] = (1 / dlt ** 2) * (data[:,-1,:] + data[:,-2,:] - 2 * data[:,-1,:])
        res[:,0,:] = (1 / dlt ** 2) * (data[:,1,:] + data[:,0,:] - 2 * data[:,0,:])

    elif dim == 2:                # to third dimension

        res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
        res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-1] + data[:,:,-2] - 2 * data[:,:,-1])
        res[:,:,0] = (1 / dlt ** 2) * (data[:,:,1] + data[:,:,0] - 2 * data[:,:,0])
    
    return res

def finiteDiff_2D_second(data, dim, dlt, cap = None):  

    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    if dim == 0:                  # to first dimension
        res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
        res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-2,:] - 2 * data[-1,:])
        res[0,:] = (1 / dlt ** 2) * (data[1,:] + data[0,:] - 2 * data[0,:])

    elif dim == 1:                # to second dimension

        res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
        res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-2] - 2 * data[:,-1])
        res[:,0] = (1 / dlt ** 2) * (data[:,1] + data[:,0] - 2 * data[:,0])
    
    return res

def finiteDiff_2D_cross(data, dlt1, dlt2, cap = None):

    res = np.zeros(data.shape)
    res[1:-1,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,1:-1] +  data[2:,:-2] + data[:-2,2:] - data[2:, 1:-1] - data[:-2, 1:-1] - data[1:-1, 2:] - data[1:-1, :-2])
    res[-1,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,1:-1] + data[-1,:-2] + data[-2,2:] - data[-1, 1:-1] - data[-2, 1:-1] - data[-1, 2:] - data[-1, :-2])
    res[0,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[0,1:-1] + data[1,:-2] + data[0,2:] - data[1, 1:-1] - data[0, 1:-1] - data[0, 2:] - data[0, :-2])
    res[1:-1,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,-1] + data[2:,-2] + data[:-2,-1] - data[2:, -1] - data[:-2, -1] - data[1:-1, -2] - data[1:-1, -1])
    res[1:-1,0] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,0] + data[2:,0] + data[:-2,1] - data[2:, 0] - data[:-2, 0] - data[1:-1, 1] - data[1:-1, 0])

    res[-1,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,-1] + data[-1,-2] + data[-2,-1] - data[-1, -1] - data[-2, -1] - data[-1, -2] - data[-1, -1])
    res[-1,0] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,0] + data[-1,0] + data[-2,1] - data[-1, 0] - data[-2, 0] - data[-1, 1] - data[-1, 0])
    res[0,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[0,-1] + data[1,-2] + data[0,-1] - data[1, -1] - data[0, -1] - data[0, -2] - data[0, -1])
    res[0,0] = (1 / (2* dlt1 * dlt2)) * (2*data[0,0] + data[1,0] + data[0,1] - data[1, 0] - data[0, 0] - data[0, 1] - data[0, 0])

    return res

def finiteDiff_3D_cross(data, dim1, dim2, dlt1, dlt2, cap = None):  
    res = np.zeros(data.shape)

    if (dim1 == 0) and (dim2 == 1):

        res[1:-1,1:-1,:] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,1:-1,:] +  data[2:,:-2,:] + data[:-2,2:,:] - data[2:, 1:-1, :] - data[:-2, 1:-1, :] - data[1:-1, 2:, :] - data[1:-1, :-2, :])

        res[-1,1:-1,:] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,1:-1,:] + data[-1,:-2,:] + data[-2,2:,:] - data[-1, 1:-1, :] - data[-2, 1:-1, :] - data[-1, 2:, :] - data[-1, :-2, :])
        res[0,1:-1,:] = (1 / (2* dlt1 * dlt2)) * (2*data[0,1:-1,:] + data[1,:-2,:] + data[0,2:,:] - data[1, 1:-1, :] - data[0, 1:-1, :] - data[0, 2:, :] - data[0, :-2, :])
        res[1:-1,-1,:] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,-1,:] + data[2:,-2,:] + data[:-2,-1,:] - data[2:, -1, :] - data[:-2, -1, :] - data[1:-1, -2, :] - data[1:-1, -1, :])
        res[1:-1,0,:] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,0,:] + data[2:,0,:] + data[:-2,1,:] - data[2:, 0, :] - data[:-2, 0, :] - data[1:-1, 1, :] - data[1:-1, 0, :])

        res[-1,-1,:] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,-1,:] + data[-1,-2,:] + data[-2,-1,:] - data[-1, -1, :] - data[-2, -1, :] - data[-1, -2, :] - data[-1, -1, :])
        res[-1,0,:] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,0,:] + data[-1,0,:] + data[-2,1,:] - data[-1, 0, :] - data[-2, 0, :] - data[-1, 1, :] - data[-1, 0, :])
        res[0,-1,:] = (1 / (2* dlt1 * dlt2)) * (2*data[0,-1,:] + data[1,-2,:] + data[0,-1,:] - data[1, -1, :] - data[0, -1, :] - data[0, -2, :] - data[0, -1, :])
        res[0,0,:] = (1 / (2* dlt1 * dlt2)) * (2*data[0,0,:] + data[1,0,:] + data[0,1,:] - data[1, 0, :] - data[0, 0, :] - data[0, 1, :] - data[0, 0, :])

    elif (dim1 == 0) and (dim2 == 2):

        res[1:-1,:,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,:,1:-1] + data[2:,:,:-2] + data[:-2,:,2:] - data[2:,:, 1:-1] - data[:-2,:, 1:-1] - data[1:-1,:, 2:] - data[1:-1,:, :-2])

        res[-1,:,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,:,1:-1] + data[-1,:,:-2] + data[-2,:,2:] - data[-1,:, 1:-1] - data[-2,:, 1:-1] - data[-1,:, 2:] - data[-1,:, :-2])
        res[0,:,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[0,:,1:-1] + data[1,:,:-2] + data[0,:,2:] - data[1,:, 1:-1] - data[0,:, 1:-1] - data[0,:, 2:] - data[0,:, :-2])
        res[1:-1,:,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,:,-1] + data[2:,:,-2] + data[:-2,:,-1] - data[2:,:, -1] - data[:-2,:, -1] - data[1:-1,:, -2] - data[1:-1,:, -1])
        res[1:-1,:,0] = (1 / (2* dlt1 * dlt2)) * (2*data[1:-1,:,0] + data[2:,:,0] + data[:-2,:,1] - data[2:,:, 0] - data[:-2,:, 0] - data[1:-1,:, 1] - data[1:-1,:, 0])
        
        res[-1,:,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,:,-1] + data[-1,:,-2] + data[-2,:,-1] - data[-1,:, -1] - data[-2,:, -1] - data[-1,:, -2] - data[-1,:, -1])
        res[-1,:,0] = (1 / (2* dlt1 * dlt2)) * (2*data[-1,:,0] + data[-1,:,0] + data[-2,:,1] - data[-1,:, 0] - data[-2,:, 0] - data[-1,:, 1] - data[-1,:, 0])
        res[0,:,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[0,:,-1] + data[1,:,-2] + data[0,:,-1] - data[1,:, -1] - data[0,:, -1] - data[0,:, -2] - data[0,:, -1])
        res[0,:,0] = (1 / (2* dlt1 * dlt2)) * (2*data[0,:,0] + data[1,:,0] + data[0,:,1] - data[1,:, 0] - data[0,:, 0] - data[0,:, 1] - data[0,:, 0])
    
    elif (dim1 == 1) and (dim2 == 2):
        
        res[:,1:-1,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[:,1:-1,1:-1] + data[:,2:,:-2] + data[:,:-2,2:] - data[:,2:, 1:-1] - data[:,:-2, 1:-1] - data[:,1:-1, 2:] - data[:,1:-1, :-2])

        res[:,-1,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[:,-1,1:-1] + data[:,-1,:-2] + data[:,-2,2:] - data[:,-1, 1:-1] - data[:,-2, 1:-1] - data[:,-1, 2:] - data[:,-1, :-2])
        res[:,0,1:-1] = (1 / (2* dlt1 * dlt2)) * (2*data[:,0,1:-1] + data[:,1,:-2] + data[:,0,2:] - data[:,1, 1:-1] - data[:,0, 1:-1] - data[:,0, 2:] - data[:,0, :-2])
        res[:,1:-1,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[:,1:-1,-1] + data[:,2:,-2] + data[:,:-2,-1] - data[:,2:, -1] - data[:,:-2, -1] - data[:,1:-1, -2] - data[:,1:-1, -1])
        res[:,1:-1,0] = (1 / (2* dlt1 * dlt2)) * (2*data[:,1:-1,0] + data[:,2:,0] + data[:,:-2,1] - data[:,2:, 0] - data[:,:-2, 0] - data[:,1:-1, 1] - data[:,1:-1, 0])

        res[:,-1,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[:,-1,-1] + data[:,-1,-2] + data[:,-2,-1] - data[:,-1, -1] - data[:,-2, -1] - data[:,-1, -2] - data[:,-1, -1])
        res[:,-1,0] = (1 / (2* dlt1 * dlt2)) * (2*data[:,-1,0] + data[:,-1,0] + data[:,-2,1] - data[:,-1, 0] - data[:,-2, 0] - data[:,-1, 1] - data[:,-1, 0])
        res[:,0,-1] = (1 / (2* dlt1 * dlt2)) * (2*data[:,0,-1] + data[:,1,-2] + data[:,0,-1] - data[:,1, -1] - data[:,0, -1] - data[:,0, -2] - data[:,0, -1])
        res[:,0,0] = (1 / (2* dlt1 * dlt2)) * (2*data[:,0,0] + data[:,1,0] + data[:,0,1] - data[:,1, 0] - data[:,0, 0] - data[:,0, 1] - data[:,0, 0])

    return res

def finiteDiff_3D2(data, dim, order, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 3:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / (2 * dlt)) * (data[2:,:,:] - data[:-2,:,:])
                res[-1,:,:] = (1 / dlt) * (data[-1,:,:] - data[-2,:,:])
                res[0,:,:] = (1 / dlt) * (data[1,:,:] - data[0,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / (2 * dlt)) * (data[:,2:,:] - data[:,:-2,:])
                res[:,-1,:] = (1 / dlt) * (data[:,-1,:] - data[:,-2,:])
                res[:,0,:] = (1 / dlt) * (data[:,1,:] - data[:,0,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,2:] - data[:,:,:-2])
                res[:,:,-1] = (1 / dlt) * (data[:,:,-1] - data[:,:,-2])
                res[:,:,0] = (1 / dlt) * (data[:,:,1] - data[:,:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
                res[-1,:,:] = (1 / dlt ** 2) * (data[-2,:,:] + 0 - data[-1,:,:])
                res[0,:,:] = (1 / dlt ** 2) * (data[1,:,:] + 0 - data[0,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
                res[:,-1,:] = (1 / dlt ** 2) * (data[:,-2,:] + 0 -data[:,-1,:])
                res[:,0,:] = (1 / dlt ** 2) * (data[:,1,:] + 0 - data[:,0,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
                res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-2] + 0 - data[:,:,-1])
                res[:,:,0] = (1 / dlt ** 2) * (data[:,:,1] + 0 - data[:,:,0])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')
    elif l == 2:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])
                res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
                res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
                res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
                res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
                res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
                res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
                res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
                res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')

            
    else:
        raise ValueError("Dimension NOT supported")
        
    if cap is not None:
        res[res < cap] = cap
    return res


