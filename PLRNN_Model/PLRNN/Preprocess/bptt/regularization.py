import torch as tc

def l1_norm(x):
    return tc.norm(x, p=1)


def l2_norm(x):
    return tc.norm(x, p=2)


def set_norm(reg_norm):
    norm = l2_norm
    if reg_norm == 'l1':
        norm = l1_norm
    return norm


def distribute_states_by_ratios(n_states, ratios_states):
    assert n_states != 0
    ratios_states = prepare_ratios(ratios_states)
    numbers_states = tc.round(n_states * ratios_states.float())
    difference = n_states - numbers_states.sum()
    biggest_diff_at = tc.argmax(tc.abs(n_states * ratios_states - numbers_states))
    numbers_states[biggest_diff_at] += difference
    numbers_states = numbers_states.int()
    return numbers_states


def set_lambdas(reg_ratios, reg_lambdas, n_states_total):
    reg_group_ratios = tc.tensor(reg_ratios)
    reg_group_lambdas = reg_lambdas
    reg_group_lambdas = list(reg_group_lambdas)
    reg_group_lambdas.append(0.)
    reg_group_lambdas = tc.tensor(reg_group_lambdas)
    reg_group_n_states = distribute_states_by_ratios(n_states=n_states_total, ratios_states=reg_group_ratios)
    alphas = tc.cat([a * tc.ones(d) for a, d in zip(reg_group_lambdas, reg_group_n_states)])
    # regularize non-read-out states
    return alphas.flip((0,))


def prepare_ratios(ratios):
    assert ratios.sum() <= 1
    missing_part = tc.abs(1 - ratios.sum())
    ratio_list = list(ratios)
    ratio_list.append(missing_part)
    return tc.tensor(ratio_list)





class Regularizer:
    def __init__(self, args):
        self.norm = set_norm(reg_norm=args.reg_norm)
        self.lambda1 = set_lambdas(reg_ratios=args.reg_ratios, reg_lambdas=args.reg_lambda1, n_states_total=args.dim_z)
        self.lambda1_W = set_lambdas(reg_ratios=args.reg_ratios_W, reg_lambdas=args.reg_lambda1, n_states_total=args.dim_hidden)
        self.lambda2 = set_lambdas(reg_ratios=args.reg_ratios_W, reg_lambdas=args.reg_lambda2, n_states_total=args.dim_hidden) 
        self.lambda3 = set_lambdas(reg_ratios=args.reg_ratios_W, reg_lambdas=args.reg_lambda3, n_states_total=args.dim_hidden)


    def loss_regularized_parameter(self, parameter, to_value, weighting_of_states):
        diff = parameter - to_value * tc.ones(parameter.size(), device=parameter.device)
        loss = weighting_of_states * self.norm(diff)
        return loss

    def loss_regularized_parameter_W_trial(self, parameter, weighting_of_states, weighting_of_states_lambda2, weighting_of_states_lambda3):
        W = parameter
        n, M1, M2 = W.size()
        W_hat  = W
        indices1 = [0] + list(range(n-1))
        W1_hat = W[indices1]
        indices2 = [0, 1] + list(range(n-2))
        W2_hat = W[indices2]

        l2_reg = 1 / (2*n) * weighting_of_states[0, 0] * self.norm(W_hat.flatten())
        term_1st_order = W_hat - W1_hat
        loss_1st_order = 1 / (2*n) * weighting_of_states_lambda2[0, 0] * self.norm(term_1st_order.flatten())
        term_2nd_order = W_hat - 2*W1_hat + W2_hat
        term_2nd_order[1, :, :] = 0
        loss_2nd_order = 1 / (2*n) *  weighting_of_states_lambda3[0, 0] * self.norm(term_2nd_order.flatten())

        return l2_reg, loss_1st_order, loss_2nd_order

    def loss(self, parameters):
        A, W1, W2, h1, h2, C = parameters
        
        loss = 0.
        if W1.dim() == 2:
            loss += self.loss_regularized_parameter(parameter=A, to_value=1., weighting_of_states=self.lambda1[0])
            loss += self.loss_regularized_parameter(parameter=W1, to_value=0., weighting_of_states=self.lambda1[0])
            loss += self.loss_regularized_parameter(parameter=W2, to_value=0., weighting_of_states=self.lambda1[0])
            loss += self.loss_regularized_parameter(parameter=h1, to_value=0., weighting_of_states=self.lambda1[0])
            loss += self.loss_regularized_parameter(parameter=h2, to_value=0., weighting_of_states=self.lambda1[0])
            loss += self.loss_regularized_parameter(parameter=C, to_value=0., weighting_of_states=self.lambda1[0])
        else:
            n = W1.size(0)
            loss += self.loss_regularized_parameter(parameter=A, to_value=1., weighting_of_states=self.lambda1[0]/ (2*n))
            
            terms = self.loss_regularized_parameter_W_trial(parameter=W1, weighting_of_states=self.lambda1_W.view(-1, 1),
                                                            weighting_of_states_lambda2=self.lambda2.view(-1, 1),
                                                            weighting_of_states_lambda3=self.lambda3.view(-1, 1))
            loss += sum(terms)
            terms = self.loss_regularized_parameter_W_trial(parameter=W2, weighting_of_states=self.lambda1_W.view(-1, 1),
                                                            weighting_of_states_lambda2=self.lambda2.view(-1, 1),
                                                            weighting_of_states_lambda3=self.lambda3.view(-1, 1))
            loss += sum(terms)

            loss += self.loss_regularized_parameter(parameter=h1, to_value=0., weighting_of_states=self.lambda1[0]/ (2*n))
            loss += self.loss_regularized_parameter(parameter=h2, to_value=0., weighting_of_states=self.lambda1[0]/ (2*n))
            loss += self.loss_regularized_parameter(parameter=C, to_value=0., weighting_of_states=self.lambda1[0]/ (2*n))
        return loss
    
    def to(self, device: tc.device) -> None:
        self.lambda1 = self.lambda1.to(device)
        self.lambda1_W = self.lambda1_W.to(device)
        self.lambda2 = self.lambda2.to(device)
        self.lambda3 = self.lambda3.to(device)