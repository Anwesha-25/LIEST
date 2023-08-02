import numpy as np
import pandas as pd

# Input: Preprocessed input matrix of size TxN,
# Output: Top influencial nodes
def LIEST(df, y, path, Stype='both', top=15, **kwargs):
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 1
    
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    else:
        beta = 0.02
        
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    else:
        gamma = 0.1
        
    if 'alpha_1' in kwargs.keys():
        alpha_1 = kwargs['alpha_1']
    else:
        alpha_1 = 1
        
    if 'beta_1' in kwargs.keys():
        beta_1 = kwargs['beta_1']
    else:
        beta_1 = 0.1
        
    if 'gamma_1' in kwargs.keys():
        gamma_1 = kwargs['gamma_1']
    else:
        gamma_1 = 0.7
        
    Re_NSE, Re_Corr = log_return_NSE(df)
    Prob_Mat = probability_matrix(Re_Corr)
    adj_mat = adjacency_matrix(Prob_Mat, Re_Corr)
    
    S, mat = influence_matrix(adj_mat, length=3)
    if Stype=='positive' or Stype=='both':
        S_positive = positive_influence(S)
        n = top
        pos_inf_node = {}
        f = {}
        f_1 ={}
        Int_com_pos = {}
        for i in y.keys():
            f[i] = target_node(path, y[i])
            f_1[i] = non_target_node(path, y[i])
            Int_com_pos[i] = inter_comp(S_positive, path,  y[i])
            pos_inf_node[i] = pos_influential_nodes(S_positive, f[i], f_1[i], Int_com_pos[i],
                                                    path, alpha, beta ,gamma , n)
        if Stype=='positive':
            return pos_inf_node
    if Stype=='negative' or Stype=='both':
        S_negative = negative_influence(S)
        n = top
        neg_inf_node = {}
        fN ={}
        fN_1 ={}
        Int_com_neg ={}
        for i in y.keys():
            fN[i] = target_node(path, y[i])
            fN_1[i] = non_target_node(path, y[i])
            Int_com_neg[i] = inter_comp(S_negative, path,  y[i])
            neg_inf_node[i] = neg_influential_nodes(S_negative, fN[i], fN_1[i], Int_com_neg[i],
                                                    path, alpha_1, beta_1, gamma_1, n)
        if Stype == 'negative':
            return neg_inf_node
    
    return pos_inf_node, neg_inf_node


# ========================================= Intermediate Codes =============================================


# Transform input data into log-return data. 
# Input(DataFrame): Preprocessed input matrix of size TxN, 
# T-> number of days, N-> number of companies.
# Output(DataFrames): Log-return data and the correlation between companies.  
def log_return_NSE(A):
    log_NSE = np.log(A)
    log_NSE_shift= np.log(A.shift(-1))
    Re_NSE = log_NSE_shift - log_NSE
    Re_NSE.drop(Re_NSE.tail(1).index,inplace=True)
    Re_Corr = Re_NSE.corr()
    return Re_NSE, Re_Corr


# Converts the correlation matrix into Probability matrix.
# Input(DataFrame): Correaltion between companies (Size->NxN).
# Output(np.array): Probability matrix. 
def probability_matrix(A):
    P = np.zeros(A.shape)
    B = np.absolute(np.asmatrix(A))
    D = np.diag(np.sum(pd.DataFrame(B), axis = 0))  
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            P[i][j]= np.abs(np.asmatrix(A)[i,j])/D[i][i]
    return P


# Converts the Probability matrix into adjacency matrix.
# Input(DataFrame, np.array): Correlation Matrix, Probability Matrix.
# Output(np.array): Adjacency Matrix. (it considers the correlation signs).
def adjacency_matrix(A, B):
    M = np.zeros(A.shape)
    for i in range(A.shape[0]):
          for j in range(A.shape[0]):
                if (np.asmatrix(B)[i,j]) > 0:
                    M[i][j]= A[i][j]
                else:
                    M[i][j]= -A[i][j]
    return M


# Takes Adjacency matrix and hoping distance(=3) and calculates the
# influence matrices and the transition impact matrix.
# Input(np.array, int): Adjacency matrix and hop leangth(=3).
# Output(np.array, dict of np.array): the transition impact matrix (Size-> NxN),
# dict of influence matrices (each having size NxN).
def influence_matrix(A, length=3):
    A_1 = diag_zero(A)
    sum = np.zeros(A.shape)
    sum = sum + A_1
    matrices = {}
    matrices[0] = np.eye(A.shape[0])
    matrices[1] = A_1
    for k in range(length):
        B = np.dot(A,matrices[k+1])
        matrices[k+2] = diag_zero(B)
        sum = sum + matrices[k+2]
    return sum, matrices


# Intermideate step:
# Makes all diagonal entry zero of a matrix.
def diag_zero(A):
    A_1 = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j:
                A_1[i][j] = A[i][j]
    return A_1


# Takes the transition impact matrix and finds its positive part.
# Input(np.array): the transition impact matrix
# Output(np.array): S_P Matrix.
def positive_influence(A):
    S_positive = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if A[i][j] > 0:
                S_positive[i][j] = A[i][j]
    return S_positive 


# Takes the transition impact matrix and finds its negative counter-part.
# Input(np.array): the transition impact matrix
# Output(np.array): S_N Matrix.
def negative_influence(A):
    S_negative = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if A[i][j] < 0:
                S_negative[i][j] = -A[i][j]
    return S_negative 


# Set the target sector by selecting the nodes belonging to that particular sector.
# One hot representation is created using this function.
# Input(list, list): list of all nodes, list of sector specific nodes.
# Output(np.array): One hot encoding (Size->Nx1)
def target_node(l1, l2):
    f = np.zeros((len(l1),1))
    for i in range(len(l1)):
        if (l1[i] in l2):
            f[i] = 1
    return f


# Set the non target sectors by selecting the nodes not belonging to a particular sector.
# One hot representation is created using this function.
# Input(list, list): list of all nodes, list of sector specific nodes.
# Output(np.array): One hot encoding (Size->Nx1)
def non_target_node(l1,l2):
    f_1 =np.zeros((len(l1),1))
    for i in range(len(l1)):
        if (l1[i] not in l2):
            f_1[i] = 1
    return f_1



# competition matrix.
# Takes the positive or negetive influence matrix, all nodes and sector sector specific target nodes as input
# Finds the competition matrix.
def inter_comp(mat, l1, l2):
    S_int = np.zeros(mat.shape)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if (l1[i] in l2) or (l1[j] in l2):
                S_int[i][j] = mat[i][j] 
    return S_int


# Top positive influential nodes for S_lim_pos
# S_P-> S_positive, f1-> target node, a-> alpha,  b-> beta,  c-> gamma,  n-> top n nodes,
# f2-> non target node vector, IP-> inter competition matrix for positive network
# Input: Positive influence matrix, target nodes, non-target nodes, competition matrix, all nodes
# Output: target specific top positive influencial nodes. 
def pos_influential_nodes(S_P, f1, f2, IP, names, a=1, b=0.02, c=0.1, n=15):
    S_lim_pos = a*np.dot(S_P,f1) - b*np.dot(S_P, f2) - c*np.dot(IP,f1)
    pos_inf_node = pd.DataFrame()
    pos_inf_node['Node'] = names      
    pos_inf_node['Value']= S_lim_pos
    Top_pos_node = pos_inf_node.nlargest(n, 'Value')
    return Top_pos_node


# Top negative influential nodes for S_lim_neg
# S_N-> S_negative, f1-> target node, a-> alpha,  b-> beta,  c-> gamma,  n-> top n nodes,
# f2-> non target node vector, IN-> inter competition matrix for -ve network
# Input: Negative influence matrix, target nodes, non-target nodes, competition matrix, all nodes
# Output: target specific top negative influencial nodes.
def neg_influential_nodes( S_N, f1,f2, IN, names, a=1, b=0.1, c=0.7, n=15):
    S_lim_neg = a*np.dot(S_N,f1) - b*np.dot(S_N, f2) + c*np.dot(IN,f1)
    neg_inf_node = pd.DataFrame()
    neg_inf_node['Node'] = names  
    neg_inf_node['Value']= S_lim_neg
    Top_neg_node = neg_inf_node.nlargest(n, 'Value')
    return Top_neg_node    
