# association rules metrics definitions
import numpy as np

def supp(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    
    A, B lists
    
    
    Support measure for a term or rule.
    If B is provided then returns the support for rule A -> B. Otherwise, the support of term A is returned.
    """
    
    if type(A) not in accepted_type:
        print("Please provide items as valid type (received {})\nif tuple of single string,\
        then please pass a list".format(type(A)))
        return None
    
    if B == None:
        try:
            selected_indices = [item_index_dict[a] for a in A]
        except KeyError as e:
            print("Please provide valid items.")
            return None
    else:
        
        if type(B) not in accepted_type:
            print("Please provide items as valid type (received {})\nif tuple of single string,\
            then please pass a list".format(type(B)))
            return None
        
        try:
            selected_indices = [item_index_dict[ab] for ab in A+B]
        except KeyError as e:
            print("Please provide valid items.")
            return None
        
    mult_arr=csc_occs[:, selected_indices[0]]
    if len(selected_indices)>1:
        for ind in selected_indices[1:]:
            mult_arr = mult_arr.multiply(csc_occs[:, ind])
        
    return mult_arr.getnnz()/csc_occs.shape[0]
        
def supp_neg(A, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    returns Supp(!A) = 1-Supp(A)
    """
    if type(A) not in accepted_type:
        print("Please provide items as valid type (received {})\nif tuple of single string,\
        then please pass a list".format(type(A)))
        return None
        
    return 1 - supp(A, None, csc_occs, item_index_dict, [list, tuple])

def conf(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    Confidence measure over rule A -> B
    Returns float in [0, 1] as a result of P(AB)/P(A)
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    try:
        A_selected_indices = [item_index_dict[a] for a in A]
        AB_selected_indices = [item_index_dict[b] for b in B]+A_selected_indices
    except KeyError as e:
        print("Please provide valid items.")
        return None
    
    AB_mult_arr=csc_occs[:, AB_selected_indices[0]]
    if len(AB_selected_indices)>1:
        for ind in AB_selected_indices[1:]:
            AB_mult_arr = AB_mult_arr.multiply(csc_occs[:, ind])
            
    A_mult_arr=csc_occs[:, A_selected_indices[0]]
    if len(A_selected_indices)>1:
        for ind in A_selected_indices[1:]:
            A_mult_arr = A_mult_arr.multiply(csc_occs[:, ind])
    
    return AB_mult_arr.getnnz()/A_mult_arr.getnnz()
    

def lift(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    Lift measure for rule A->B
    
    Returns float f so that
        - if f < 0: negative correlation
        - if f == 1: no correlation
        - if f > 1: positive correlation
    
    Lift(A->B) = P(B | A) / P(B) = P(AB)/(P(A)P(B))
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    try:
        A_selected_indices = [item_index_dict[a] for a in A]
        B_selected_indices = [item_index_dict[b] for b in B]
        AB_selected_indices = A_selected_indices+B_selected_indices
    except KeyError as e:
        print("Please provide valid items.")
        return None
    
    AB_mult_arr=csc_occs[:, AB_selected_indices[0]]
    if len(AB_selected_indices)>1:
        for ind in AB_selected_indices[1:]:
            AB_mult_arr = AB_mult_arr.multiply(csc_occs[:, ind])
            
    A_mult_arr=csc_occs[:, A_selected_indices[0]]
    if len(A_selected_indices)>1:
        for ind in A_selected_indices[1:]:
            A_mult_arr = A_mult_arr.multiply(csc_occs[:, ind])
            
    B_mult_arr=csc_occs[:, B_selected_indices[0]]
    if len(B_selected_indices)>1:
        for ind in B_selected_indices[1:]:
            B_mult_arr = B_mult_arr.multiply(csc_occs[:, ind])
    
    return (AB_mult_arr.getnnz()/A_mult_arr.getnnz())/(B_mult_arr.getnnz()/csc_occs.shape[0])


def conf_neg2(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    Confidence measure over rule A -> not B
    Returns float in [0, 1]
    Conf(A->!B) = 1- Conf(A->B)
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    return 1-conf(A, B, csc_occs, item_index_dict, [list, tuple])

def conf_neg1(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    Confidence measure over rule !A -> B
    Returns float in [0, 1]
    Conf(!A->B) = (Supp(B)(1-Conf(B->A)))/(1-P(A))
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    if supp(A, None, csc_occs, item_index_dict) == 1:
        print("Please provide an item whose support is <1. (supp({})=1)".format(A))
        return None
    else:
        return (supp(B, None, csc_occs, item_index_dict, [list, tuple])*(1-conf(B, A, csc_occs, item_index_dict, [list, tuple])))/(1-supp(A, None, csc_occs, item_index_dict, [list, tuple]))

def conf_neg12(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    Confidence measure over rule !A -> !B
    Returns float in [0, 1]
    Conf(!A->!B) = (1-Supp(A)-Supp(B)+Supp(A u B))/(1-P(A))
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    if supp(A, None, csc_occs, item_index_dict) == 1:
        print("Please provide an item whose support is <1. (supp({})=1)".format(A))
        return None
    
    return (1-supp(A, None, csc_occs, item_index_dict, [list, tuple])-supp(B, None, csc_occs, item_index_dict, [list, tuple])+supp(list(A)+list(B), None, csc_occs, item_index_dict, [list, tuple]))/(1-supp(A, None, csc_occs, item_index_dict, [list, tuple]))


def supp_union_neg2(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    support for A u not B
    supp(A u !B) = Supp(A) - Supp(A u B)
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    return supp(A, None, csc_occs, item_index_dict, [list, tuple]) - supp(list(A)+list(B), None, csc_occs, item_index_dict, [list, tuple])


def supp_union_neg1(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    support for not A u B
    supp(!A u B) = Supp(B) - Supp(A u B)
    """
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    return supp(B, None, csc_occs, item_index_dict, [list, tuple]) - supp(list(A)+list(B), None, csc_occs, item_index_dict, [list, tuple])
    

def supp_union_neg12(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    support for not A u not B
    Supp(!A u !B) = 1 - Supp(A) - Supp(B) + Supp(AuB)
    """
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    return 1 - supp(A, None, csc_occs, item_index_dict, [list, tuple]) - supp(B, None, csc_occs, item_index_dict, [list, tuple]) + supp(list(A)+list(B), None, csc_occs, item_index_dict, [list, tuple])


def lift_neg2(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    Lift measure for rule A->!B
    
    Returns float f so that
        - if f < 0: negative correlation
        - if f == 1: no correlation
        - if f > 1: positive correlation
    
    Lift(A->!B) = Supp(Au!B)/(Supp(A)Supp(!B))
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    if supp_neg(B, csc_occs, item_index_dict) == 0:
        print("Please provide B so that supp(B)<1")
        return None
    
    return supp_union_neg2(A, B, csc_occs, item_index_dict, [list, tuple])/(supp(A, None, csc_occs, item_index_dict, [list, tuple])*supp_neg(B, csc_occs, item_index_dict, [list, tuple]))


def lift_neg1(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    Lift measure for rule !A->B
    
    Returns float f so that
        - if f < 0: negative correlation
        - if f == 1: no correlation
        - if f > 1: positive correlation
    
    Lift(!A->B) = Supp(!AuB)/(Supp(!A)Supp(B))
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    return supp_union_neg1(A, B, csc_occs, item_index_dict, [list, tuple])/(supp_neg(A, csc_occs, item_index_dict, [list, tuple])*supp(B, None, csc_occs, item_index_dict, [list, tuple]))


def lift_neg12(A, B, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    Lift measure for rule !A->!B
    
    Returns float f so that
        - if f < 0: negative correlation
        - if f == 1: no correlation
        - if f > 1: positive correlation
    
    Lift(!A->!B) = Supp(!A u !B)/(Supp(!A)Supp(!B))
    """
    
    if type(A) not in accepted_type or type(B) not in accepted_type:
        print("Please provide items as valid type (received {} and {})\nif tuple of single string,\
        then please pass a list".format(type(A), type(B)))
        return None
    
    return supp_union_neg12(A, B, csc_occs, item_index_dict, [list, tuple])/(supp_neg(A, csc_occs, item_index_dict, [list, tuple])*supp_neg(B, csc_occs, item_index_dict, [list, tuple]))


def IDF(i, csc_occs, item_index_dict, accepted_type = [list, tuple]):
    """
    
    Inverse Document Frequency
    
    IDF(i) = log(|D|/d_i)
    
    where
        - |D| = number of transactions in the datasets D
        - d_i = number of transactions in D containing i
        
    """
    
    if type(i) not in accepted_type:
        print("Please provide terms as valid types. (received {}. \
        If tuple of single element please provide list.)".format(type(i)))
        return None
    
    try:
        i_selected_indices = [item_index_dict[x] for x in i]
    except ValueError as e:
        print("Please provide valid items.")
        return None
    
    mult_arr=csc_occs[:, i_selected_indices[0]]
    if len(i_selected_indices)>1:
        for ind in i_selected_indices[1:]:
            mult_arr = mult_arr.multiply(csc_occs[:, ind])
    
    return np.log(csc_occs.shape[0]/mult_arr.getnnz())
