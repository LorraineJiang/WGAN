import torch

'''-------------------------------Gini不纯度-------------------------------'''
class Gini_Impurity():
    def __init__(self, pred, lambda_gini):
        self.pred = pred
        self.lambda_gini = lambda_gini

    def value(self):
        # 计算Gini不纯度
        gini = torch.pow(self.pred, 2)  # 平方
        gini = torch.sum(gini)          # 求和
        gini = torch.sub(1.0, gini)     # 1-求和所得
        gini = torch.mean(gini)         # 求均值
        return self.lambda_gini * gini  # 乘上Gini系数并返回


'''-------------------------------Tsallis熵-------------------------------'''
class Tsallis_Entropy():
    def __init__(self, pred, lambda_tsallis, q=2):
        self.pred = pred
        self.lambda_tsallis = lambda_tsallis
        if q<0 or q==1:
            print('q should be nonnegative real number and q!=1')
            return
        else:
            self.q = q

    # Tsallis熵的ln_func(x),其中0≤q<1或q>1
    def ln_func(self):
        if self.q > 1:                                      # 求p(x)的(1-q)次方
            ln_q = torch.pow(self.pred, abs(1 - self.q))
            ln_q = torch.reciprocal(ln_q)
        else:
            ln_q = torch.pow(self.pred, 1-self.q)
        ln_q = torch.add(ln_q, -1)                          # 再减1
        ln_q = torch.div(ln_q*1.0, 1-self.q)                # 最后除以(1-q)
        return ln_q
        
    # Tsallis熵:S_q(x)=−∑P(x)^qln_qP(x)
    def value(self):
        tsallis_ent = torch.pow(self.pred, self.q)                  # 求p(x)的q次方
        tsallis_ent = torch.mul(tsallis_ent, self.ln_func())        # 再乘上ln_func
        tsallis_ent = torch.mean(torch.sum(tsallis_ent))            # 求和再求均值
        return self.lambda_tsallis * tsallis_ent


'''-------------------------------Tsallis互熵(对有标样本)-------------------------------'''
class Tsallis_Mutual_Entropy():
    def __init__(self, pred, target, lambda_tsallis_mutual, q=2):
        self.pred = pred
        self.target = target
        self.lambda_tsallis_mutual = lambda_tsallis_mutual
        if q < 1:
            print('q should be greater than 1')
            return
        else:
            self.q = q

    # Tsallis互熵的ln_func(x),其中q>1
    def ln_func(self):
        ln_q = torch.pow(self.pred, abs(1 - self.q))        # 求p(x)的(1-q)次方
        ln_q = torch.reciprocal(ln_q)
        ln_q = torch.add(ln_q, -1)                          # 再减1
        ln_q = torch.div(ln_q*1.0, 1-self.q)                # 最后除以(1-q)
        return ln_q

    # Tsallis条件熵S_q(X|Y)=−∑P(x,y)^qln_qP(x|y)=∑P(y)^qS_q(X|y)
    def tsallis_cond_entropy(self):
        pass
    
'''-------------------------------Tsallis相对熵(对有标样本)【同KL散度】-------------------------------'''
class Tsallis_Relative_Entropy():
    def __init__(self, pred, target, lambda_tsallis_relative, q=2):
        self.pred = pred
        self.target = target
        self.lambda_tsallis_relative = lambda_tsallis_relative
        if q<0 or q==1:
            print('q should be nonnegative real number and q!=1')
            return
        else:
            self.q = q
    
    # Tsallis相对熵的ln_func(x),其中0≤q<1或q>1
    def ln_func(self):
        target_div_pred = torch.div(self.target, self.pred)
        if self.q > 1:                                      # 求p(x)的(1-q)次方
            ln_q = torch.pow(target_div_pred, abs(1 - self.q))
            ln_q = torch.reciprocal(ln_q)
        else:
            ln_q = torch.pow(target_div_pred, 1-self.q)
        ln_q = torch.add(ln_q, -1)                          # 再减1
        ln_q = torch.div(ln_q*1.0, 1-self.q)                # 最后除以(1-q)
        return ln_q

    # Tsallis相对熵D_q(X|Y)=-∑P(x)*ln_q(P(x)/Q(x))
    def value(self):
        tsallis_relative_ent = self.pred * self.ln_func()       # 求P(x)*ln_q(P(x)/Q(x))
        tsallis_relative_ent = torch.sum(tsallis_relative_ent)  # 求和
        tsallis_relative_ent = torch.neg(tsallis_relative_ent)  # 取负
        return tsallis_relative_ent