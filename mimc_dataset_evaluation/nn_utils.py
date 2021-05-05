from torch import nn
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import scipy.sparse as sp
import pandas as pd 
import gc
import RL_data_util_ehr as RL
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def load_records(cancer_data,min_items,dataset,train_test_split=0.7):
    if cancer_data:
        records = pd.read_csv('medical.txt', sep='\t', header=None)
        columns = ['user', 'treatment', 'drug', 'view', 'date', 'time']
        records.columns = columns


        x = records.groupby('user')['drug'].agg('unique')
        medicines = []
        for i in range(0, len(x)):
            if len(x.iloc[i]) > min_items:
                user_medicine = f"{x.iloc[i][0]}"
                for drug in range(1,len(x.iloc[i])):
                    user_medicine += f' {x.iloc[i][drug]}'
                medicines.append(user_medicine)

        diagnosis = medicines
        procedures = medicines
    elif dataset == 'movielens':
        records = pd.read_csv('u.data',sep='\t',header = None)
        columns = ['user','drug','ranking','view']
        records.columns = columns
        medicines = []
        records = records[records['ranking'] > 0]
        records = records.drop('ranking', axis=1)

        x = records.groupby('user')['drug'].unique()
        for i in range(0, len(x)):
            if len(x.iloc[i]) > 20:
                user_medicine = f"{x.iloc[i][0]}"
                for drug in range(1,len(x.iloc[i])):
                    user_medicine += f' {x.iloc[i][drug]}'
                medicines.append(user_medicine)
        diagnosis = medicines
        procedures = medicines
    else:
        records = pd.read_csv('mimic.csv')
        columns = ['patient',  'drug']
        records= records[columns]

        x = records.groupby('patient')['drug'].agg('unique')
        medicines = []
        for i in range(0, len(x)):
            if len(x.iloc[i]) > 15:
                user_medicine = f"{x.iloc[i][0]}"
                for drug in range(1, len(x.iloc[i])):
                    user_medicine += f' {x.iloc[i][drug]}'
                medicines.append(user_medicine)
                
        diagnosis = medicines
        procedures = medicines
    # ID the code
    
    # Adding split records for taking few lines and not entire EHR dataset

    
    #diagnosis=[row[0] for row in records]
    #procedures=[row[1] for row in records]
    #medicines=[row[2] for row in records]
    
    # print ATC codes of drugs used on first 10 patients on record file
    print('medicines:',medicines[:10])
    # record with the most diagnosis and procedures values
    diagnosis_maxlen=max([len(line.split(' ')) for line in diagnosis])
    procedure_maxlen=max([len(line.split(' ')) for line in procedures])
    medicines_maxlen = max([len(line.split(' ')) for line in medicines])
    # Creating a bag-of-words for Diagnosis and Procedures sets
    # Each diagnosis/procedure code is taken from the EHR input file
    diagnosis_tokenizer = Tokenizer()
    diagnosis_tokenizer.fit_on_texts(diagnosis)
    sequences = diagnosis_tokenizer.texts_to_sequences(diagnosis)

    diagnosis_= pad_sequences(sequences, maxlen=diagnosis_maxlen, padding='post', truncating='post')

    procedure_tokenizer=Tokenizer()
    procedure_tokenizer.fit_on_texts(procedures)
    sequences=procedure_tokenizer.texts_to_sequences(procedures)

    procedure_=pad_sequences(sequences,maxlen=procedure_maxlen,padding='post',truncating='post')

    medicines_tokenizer = Tokenizer()
    medicines_tokenizer.fit_on_texts(medicines)
    sequences = medicines_tokenizer.texts_to_sequences(medicines)

    medicines_ = pad_sequences(sequences, maxlen=medicines_maxlen, padding='post', truncating='post')
    medicineSet=[]
    for row in medicines:
        for item in row.split(' '):
            if item not in medicineSet:
                medicineSet.append(item)

    drug2id={drug:id for id,drug in enumerate(medicineSet)}
    drug2id['END']=len(drug2id)
    drugIds=[]
    for line in medicines:
        line=line+' '+'END'
        drugIds.append([drug2id[item] for item in line.split(' ')])

    X,Y=[],[]
    for x,z in zip(medicines_,drugIds):
        train_number = int((len(z)-1)*0.7)
        x[train_number:] = 0
        X.append([x])
        Y.append(z[train_number:])
    print('drugIds:',drugIds[:5])
    return X,Y, drug2id

def init_ADJ(nodes_dict):
    adjacencies = []
    # n x n matrix (connected to each other node or not?)
    adj_shape = (len(nodes_dict), len(nodes_dict))

    edges = np.empty((len(nodes_dict), 2), dtype=np.int32)
    for j in range(len(nodes_dict)):
        edges[j] = np.array([j, j])
    row, col = np.transpose(edges)
    data = np.zeros(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)

    adjacencies.append(adj)
    adjacencies.append(adj)

    # Adjacency Matrix (initially all zeros)
    return adjacencies
def get_torch_sparse_matrix(A,dev):
    '''
    A : list of sparse adjacency matrices
    '''

    newA=[]
    for row in A:
        idx = torch.LongTensor([row.tocoo().row, row.tocoo().col])
        dat = torch.FloatTensor(row.tocoo().data)
        newA.append(torch.sparse.FloatTensor(idx, dat, torch.Size([row.shape[0], row.shape[1]])).to(dev))
    del idx,dat
    # gc.collect()
    return newA

class RGCN(nn.Module):
    def __init__(self,layer_sizes,total_ent,dev='cpu'):
        super(RGCN, self).__init__()

        self.layer_sizes = layer_sizes
        self.node_init = None
        self.layers = nn.ModuleList()
        self.device=dev

        _l=total_ent
        for l in self.layer_sizes:
            # Two layers, first of [139, 50] and second of [50, 50]
            self.layers.append(GCNLayer(_l, l))
            _l = l
        self.ent_emb = None

    def forward(self,adj_mat_list):
        '''
        inp: (|E| x d)
        adj_mat_list: (R x |E| x |E|)
        '''
        out = self.node_init
        final_rep=[]
        for i, layer in enumerate(self.layers):
            if i != 0:
                out = F.relu(out)
            out = layer(out, adj_mat_list)
            final_rep.append(out)
        self.ent_emb = out
        final_rep=torch.cat(final_rep,1) # EQUATION 11
        final_rep=torch.sum(final_rep,0).view(-1,final_rep.shape[1]) # EQUATION 10
        return out,final_rep

class GCNLayer(nn.Module):
    def __init__(self,in_size,out_size,total_rel=2,n_basis=2,dev='cuda:1'):
        super(GCNLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_basis=n_basis
        self.total_rel=total_rel
        self.dev=dev
        # Weight bases
        self.basis_weights = nn.Parameter(torch.FloatTensor(self.n_basis, self.in_size, self.out_size))
        # Linear combination coefficients (for basis regularization for num of parameters growth)
        self.basis_coeff = nn.Parameter(torch.FloatTensor(self.total_rel, self.n_basis))

        self.register_parameter('bias', None)
        self.reset_parameters() # Initialize all variables

    def forward(self,inp,adj_mat_list):
        '''
               inp: (|E| x in_size)
               adj_mat_list: (R x |E| x |E|)
               '''
        # Aggregation (no explicit separation of Concat step here since we are simply averaging over all)
        # self.basis_coeff:(95,2),self.basis_weights:(2,8285,16)
        # rel_weights:(95,8285,16)
        rel_weights = torch.einsum('ij,jmn->imn', [self.basis_coeff, self.basis_weights])
        # weights:(787075,16)
        weights = rel_weights.view(rel_weights.shape[0] * rel_weights.shape[1],
                                   rel_weights.shape[2]) # (in_size * R, out_size)

        emb_acc = []
        if inp is not None:
            for mat in adj_mat_list:
                emb_acc.append(torch.mm(mat, inp))  # (|E| x in_size)
            tmp = torch.cat(emb_acc, 1)
        else:
            tmp = torch.cat([item.to_dense() for item in adj_mat_list], 1)
        out = torch.matmul(tmp, weights)  # (|E| x out_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)  # EQUATION 9
        return out  # (|E| x out_size)

    def reset_parameters(self):
        # Initializing weights using xavier_uniform method
        nn.init.xavier_uniform_(self.basis_weights.data) #(2,8285,16)
        nn.init.xavier_uniform_(self.basis_coeff.data) #(95,2)

        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)


# Convolutional Neural Network (for Diagnosis and Procedure patient representations)
class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, num_channels, hidden_dim, dropout):
        super(CNN, self).__init__()
        # a lookup table that stores word embeddings of a fixed dictionary and size
        self.embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=emb_size,  # input num channels
                out_channels=num_channels,  # n_filters
                kernel_size=3,  # filter size
                stride=2,  # filter movement/step
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.Tanh(),  # EQUATION 16

            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.Tanh(),  # EQUATION 16

            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
        )
        self.dropout = dropout
        self.out = nn.Linear(num_channels, hidden_dim, bias=True)  # Linear transformation
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, x):
        x_emb = self.embedding(x).unsqueeze(0).permute(0, 2, 1)
        # print('x_emb:',x_emb.shape) #[1, 100, 30]
        x = self.conv(x_emb)
        # average and remove the extra dimension
        remaining_size = x.size(dim=2)
        features = F.max_pool1d(x, remaining_size).squeeze(dim=2)  # Max-Pooling (get max value)
        features = F.dropout(features, p=self.dropout)
        output = self.out(features)
        return output

class DQN(nn.Module):
    def __init__(self,state_size,action_size):
        super(DQN, self).__init__()
        # A simple three-layer perceptron network that makes decisions based on state
        # W, U are parameter matrices for calculating hidden state representation
        self.W=nn.Parameter(torch.FloatTensor(state_size,state_size))
        # Initializing weights with xavier_uniform method
        nn.init.xavier_uniform_(self.W.data)
        self.U=nn.Parameter(torch.FloatTensor(state_size,state_size))
        nn.init.xavier_uniform_(self.U.data)

        self.fc1=nn.Linear(state_size,512)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2=nn.Linear(512,action_size)
        nn.init.kaiming_normal_(self.fc2.weight) # relu function
        self.learn_step_counter=0 # Used to determine when to update the target network

    def forward(self,x_): # next state in the DQN
        x_t,h_t_1=x_[0],x_[1]
        h_t_1=h_t_1.to(device)

        state=F.sigmoid(torch.mm(self.W,x_t.t())+torch.mm(self.U,h_t_1.t())) # EQUATIOM 6

        fc1=F.relu(self.fc1(state.t())) # EQUATION 5
        output=self.fc2(fc1) # EQUATION 4
        # Outputs of network are used in act() Agent method

        return state.t(),output

class A2C(nn.Module):
    def __init__(self,state_size,action_size):
        super(A2C, self).__init__()
        # A simple three-layer perceptron network that makes decisions based on state
        # W, U are parameter matrices for calculating hidden state representation
        self.W=nn.Parameter(torch.FloatTensor(state_size,state_size))
        # Initializing weights with xavier_uniform method
        nn.init.xavier_uniform_(self.W.data)
        self.U=nn.Parameter(torch.FloatTensor(state_size,state_size))
        nn.init.xavier_uniform_(self.U.data)

        self.fc1=nn.Linear(state_size,512)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.actor=nn.Linear(512,action_size)
        nn.init.kaiming_normal_(self.actor.weight) # relu function
        self.critic=nn.Linear(512,1)
        nn.init.kaiming_normal_(self.critic.weight) # relu function
        self.learn_step_counter=0 # Used to determine when to update the target network

    def forward(self,x_): # next state in the DQN
        x_t,h_t_1=x_[0],x_[1]
        h_t_1=h_t_1.to(device)

        state=F.sigmoid(torch.mm(self.W,x_t.t())+torch.mm(self.U,h_t_1.t())) # EQUATIOM 6

        fc1=F.relu(self.fc1(state.t())) # EQUATION 5
        logits=self.actor(fc1) # EQUATION 4
        # Outputs of network are used in act() Agent method
        value = self.critic(fc1)
        return state.t(),logits,value
class Agent(object):
    def __init__(self, state_size, action_size, layer_sizes):
        super(Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9  # Discount Rate for future Bonus
        self.tau = 0.99 # Parameter for A2C and advantage
        self.epsilon = 0.9  # Discovery rate of Environment selection action
        self.epsilon_min = 0.05  # Agent controls the threshold for random exploration
        self.epsilon_decay = 0.995  # For agent to make better choices, reduce exploration rate (after episode)
        self.cnn_diagnosis = CNN(len(diagnosis_token.word_index) + 2, EMB_SIZE, 128, EMB_SIZE, 0.5).to(device)
        self.cnn_procedure = CNN(len(procedure_token.word_index) + 2, EMB_SIZE, 128, EMB_SIZE, 0.5).to(device)
        self.ddi_df = ddi_df
        # Node Representation with R-GCN
        # RGCN Model is composed of RGCN Layers
        # For each node, RGCN Layer computes outgoing message using node representation and weight matrix
        # associated with edge type. Then it aggregates incoming messages and generates new node representations
        self.rgcn = RGCN(layer_sizes, drug_vocab_size, dev=device).to(device)
        self.model = A2C(state_size, action_size).to(device)  # Policy Network
        #self.target_model = A2C(state_size, action_size).to(device)  # Target Network
        self.model_params = list(self.cnn_diagnosis.parameters()) + list(self.cnn_procedure.parameters()) + list(
            self.rgcn.parameters()) + list(self.model.parameters())
        self.optimizier = torch.optim.Adam(self.model_params, lr=LR, betas=(0.9, 0.999), weight_decay=5.0)
        self.loss = nn.MSELoss()
        # self.load_params()

    def load_params(self):
        if os.path.exists('MIMIC-III/agent.pkl'):
            # reload the params
            trainedModel = torch.load('MIMIC-III/agent.pkl')
            print('load trained model....')
            self.cnn_diagnosis.load_state_dict(trainedModel.cnn_diagnosis.state_dict())
            self.cnn_procedure.load_state_dict(trainedModel.cnn_procedure.state_dict())
            self.rgcn.load_state_dict(trainedModel.rgcn.state_dict())
            self.model.load_state_dict(trainedModel.model.state_dict())
            # self.target_model.load_state_dict(trainedModel.target_model.state_dict())

    def reset(self, x):  # multiple records for one patient
        # Get a representation of each electronic medical record data
        x0 = torch.LongTensor(x[0]).to(device)  # this gets diagnosis
        # x1 = torch.LongTensor(x[1]).to(device)  # this gets procedures
        diagnosis_f = self.cnn_diagnosis(x0)  # embedding - zd
        # procedure_f = self.cnn_procedure(x1)  # embedding - zp
        # Concatenation of embeddings
        # f = torch.cat((diagnosis_f, procedure_f), 0)  # Concatenation of zd and zp
        # print('g:',f.shape)
        return diagnosis_f

    def act(self, x, h, selectedAction):
        # Select action based on state
        # if np.random.rand() < self.epsilon:
        #     while True:
        #         # Exploration (select a random action/drug)
        #         action = random.randrange(self.action_size)
        #         if action not in selectedAction:
        #             return action,0,0, h
        #             # Exploitation
        # Forwards state input into DQN network
        next_h, logits, value = self.model((x, h))
        while True:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                action = probs.multinomial(1).data

                # Select maximum Q-value from table with current (state,action) pair

                if action not in selectedAction:
                    return action,value,logits, next_h
                else:
                    # if action already in selectedAction
                    logits[0][action] = -999999

    def new_state(self, f, g):
        # Non-linear activation function (prob values sum up to 1)
        # Element wise multiplication
        a = nn.functional.softmax(torch.mm(f, g.t()))  # EQUATION 14
        f_ = torch.mm(a.t(), f)  # matrix multiplcation
        # xt the state used to compute ht
        x = f_ + g  # EQUATION 7
        return x

    def step(self, action, selectedAction, y,drug2id):  # Transition
        action_size = len(drug2id)
        # First determine whether the action is the end of the flag
        if action == action_size - 1:
            # To determine whether the current end of the number of steps
            if len(selectedAction) == len(y) - 1:
                # if selectedNumOfDrugs set length is the same as Y_train set length
                # Correct num of selections, then reward of 2
                reward = 2
                return reward, 0
            else:  # There are two cases where the end number is exceeded or the end number is not reached
                reward = -2
                return reward, 0
        else:
            # Reward based on the action
            # Prediction of a drug in Y set per this record and is not a duplicated choice
            if int(action) in y and action not in selectedAction:
                reward = 3
            # Wrong prediction
            else:
                reward = -1

            # Update new drug map (adjacency matrix)
            # Use of a(t) recently obtained output
            adjacencies = RL.getADJ(action, selectedAction, drug2id, self.ddi_df)  # arrow on the graph for next state
            adjacencies = RL.get_torch_sparse_matrix(adjacencies, device)

            _, g = self.rgcn(adjacencies)  # g shape(1,100)
            del adjacencies
            gc.collect()
            return reward, g

    def process_rollout(self, steps, ):
        # bootstrap discounted returns with final value estimates
        _, _, _, _, last_values = steps[-1]
        returns = last_values.data

        advantages = torch.zeros(1, 1).to(device)

        # batch_actions = []
        # batch_policies = []
        # batch_values = []
        # batch_returns = []
        # batch_advantages = []
        out = [None] * (len(steps) - 1)

        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(len(steps) - 1)):
            rewards, masks, actions, policies, values = steps[t]
            _, _, _, _, next_values = steps[t + 1]

            returns = rewards + returns * self.gamma * masks

            deltas = rewards + next_values.data * self.gamma * masks - values.data
            advantages = advantages * self.gamma * self.tau* masks + deltas
            # batch_actions.append(actions)
            # batch_policies.append(policies)
            # batch_values.append(values)
            # batch_returns.append(returns)
            # batch_advantages.append(advantages)
            out[t] = actions, policies, values, returns, advantages

        # return data as batched Tensors, Variables

        return map(lambda x: torch.cat(x, 0), zip(*out))
    def replay(self, steps):
        print('learning_step_counter:{}'.format(self.model.learn_step_counter))
        # Determine whether we need to update the target network (every 10 iterations)


        self.model.learn_step_counter += 1

        # Selecting randomly BATCH_SIZE num of experiences from memory set
        actions_to_train, policies, values, returns, advantages = self.process_rollout( steps)

        # calculate action probabilities
        probs = F.softmax(policies, dim=-1)
        log_probs = F.log_softmax(policies, dim=-1)
        log_action_probs = log_probs.gather(1, actions_to_train)

        policy_loss = (-log_action_probs * advantages).sum()
        value_loss = (.5 * (values - returns) ** 2.).sum()
        entropy_loss = (log_probs * probs).sum()

        loss = policy_loss + value_loss * 0.5 - entropy_loss * 0.01

        self.optimizier.zero_grad()
        loss.backward(retain_graph=True)

        # nn.utils.clip_grad_norm_(net.parameters(), params.params['max_grad_norm'])
        self.optimizier.step()


        return loss

    def update_target_model(self):
        # Load the parameters into Target Network Model
        # Used for computing loss between target and policy networks
        self.target_model.load_state_dict(self.model.state_dict())


class Evaluate(object):
    # X_val and Y_val sets
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def evaluate(self):
        # Evaluate model performance on datasets
        Jaccard_list = []
        Recall_list = []
        Reward_list = []
        Precision_list = []
        F_list = []
        D_DList = []

        for x, y in zip(self.X, self.Y):
            if len(y) == 0:
                break
            sampleReward = 0
            # Because there are duplicate drugs in y, this translates into a collection of duplicate drugs
            y = set(y)
            # Get the initial state
            # Use the GCN and pass initialized adjacency matrix for node representation
            # if test:
            _, g = agent.rgcn(init_adj)
            # Get output of CNN (diagnose, procedure) that will be multiplied with 'g'
            f = agent.reset(x)
            selectedAction = []
            # Hidden layer h(t-2) values first layer where we initialize with 0
            h = np.zeros((1, state_size))  # （1,100）
            h = torch.FloatTensor(h)

            for step in range(max_step):  # length y_pred is 15 max
                x = agent.new_state(f, g)  # x shape:(100,1)

                # Fprward x,h into DQN and get state s(t-1)
                next_h, logits, value = agent.model((x, h))
                next_h = next_h.detach()  # Stop grad calculation
                logits = logits.detach()  # Stop grad calculation
                value = value.detach()  # Stop grad calculation

                probs = F.softmax(logits, dim=-1)
                action = probs.max(1)[1][0]
                if action == action_size - 1 and step == 0:
                    logits[0][action_size - 1] = -999999
                while True:
                    # Choose largest output from DQN output layer value as the action
                    probs = F.softmax(logits, dim=-1)
                    action = probs.max(1)[1][0]
                    if int(action) not in selectedAction:
                        break
                    else:
                        logits[0][action] = -999999
                # Execute the action to get the reward and update the state
                reward, _ = agent.step(action, selectedAction, y)
                if type(_) != int:  # Note that the prediction is not an end character
                    g = _
                    # Add selected action to list
                    selectedAction.append(int(action))
                    sampleReward += int(reward)
                    next_x = agent.new_state(f, g)  # Get a new moment of input xt
                    # Replace the original state with the new state of the moment
                    x = next_x
                    h = next_h

                else:  # Predicting end-of-run
                    selectedAction.append(int(action))
                    sampleReward += int(reward)
                    break

            # Get metrics per EHR record (row in X_val, Y_val)
            jaccard, recall, precision, f_measure = self.evaluate_sample(selectedAction, y)

            Jaccard_list.append(jaccard)
            Recall_list.append(recall)
            Reward_list.append(sampleReward)
            Precision_list.append(precision)
            F_list.append(f_measure)
            # Determine whether the resulting drug has an adverse DDI drug
            d_d, ddRate = self.evaluate_ddi(y_pred=selectedAction)
            D_DList.append(ddRate)

        # After all records in validation sets
        avg_jaccard = sum(Jaccard_list) * 1.0 / len(Jaccard_list)
        avg_recall = sum(Recall_list) * 1.0 / len(Recall_list)
        avg_reward = sum(Reward_list) * 1.0 / len(Reward_list)
        avg_precision = sum(Precision_list) * 1.0 / len(Precision_list)
        avg_f = sum(F_list) * 1.0 / len(F_list)
        avg_ddr = sum(D_DList) * 1.0 / len(D_DList)
        print('avg_jaccard:{},avg_recall:{},avg_precision:{},avg_f:{},avg_reward:{},avg_ddr:{}'.format(avg_jaccard,
                                                                                                       avg_recall,
                                                                                                       avg_precision,
                                                                                                       avg_f,
                                                                                                       avg_reward,
                                                                                                       avg_ddr))
        del Jaccard_list, Recall_list, Reward_list, Precision_list, F_list

        return avg_reward, avg_jaccard, avg_recall, avg_precision, avg_f, avg_ddr

    def evaluate_sample(self, y_pred, y_true):  # Predicted drug, Actual Drug
        # Evaluation results for three indicators of a single sample
        print('y_pred:', y_pred)
        print('y_true:', y_true)
        jiao_1 = [item for item in y_pred if item in y_true]  # Intersection of drug sets
        # Union of drug sets of predicted and true values
        bing_1 = [item for item in y_pred] + [item for item in y_true]
        bing_1 = list(set(bing_1))  # Unique values in union set
        # print('jiao:',jiao_1)
        # print('bing:',bing_1)
        recall = len(jiao_1) * 1.0 / len(y_true)
        precision = len(jiao_1) * 1.0 / len(y_pred)
        jaccard = len(jiao_1) * 1.0 / len(bing_1)

        if recall + precision == 0:
            f_measure = 0
        else:
            f_measure = 2 * recall * precision * 1.0 / (recall + precision)
        print('jaccard:%.3f,recall:%.3f,precision:%.3f,f_measure:%.3f' % (jaccard, recall, precision, f_measure))
        del jiao_1, bing_1
        return jaccard, recall, precision, f_measure

    def evaluate_ddi(self, y_pred):
        y_pred = list(set(y_pred))
        # Find the corresponding drug name according to the drug id
        # to determine whether there is a drug against these drugs

        # Combining the resulting drug
        D_D = []
        for i in range(len(y_pred) - 1):
            for j in range(i + 1, len(y_pred)):
                key1 = [y_pred[i], y_pred[j]]
                key2 = [y_pred[j], y_pred[i]]

                if key1 in ddi_df or key2 in ddi_df:
                    # Record the DDI data for analysis in the case Study section of the paper
                    D_D.append(key1)
        allNum = len(y_pred) * (len(y_pred) - 1) / 2
        if allNum > 0:
            return D_D, len(D_D) * 1.0 / allNum
        else:
            return D_D, 0

    def plot_result(self, total_reward, total_recall, total_jaccard):
        # Drawing
        import matplotlib.pyplot as plt
        import matplotlib
        # Start drawing
        plt.figure()
        ax = plt.gca()

        epochs = np.arange(len(total_reward))

        plt.subplot(1, 2, 1)
        # Set the display scale of the axis to a multiple of 50
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.plot(epochs, total_reward, label='Reward')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, total_recall, label='Recall', color='red')
        plt.plot(epochs, total_jaccard, label='Jaccard')
        # Set the display scale of the axis to a multiple of 50
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.show()



class EmbeddingNet(nn.Module):
    """
    Creates a dense network with embedding layers.
    
    Args:
    
        n_users:            
            Number of unique users in the dataset.
        n_movies: 
            Number of unique movies in the dataset.
        n_factors: 
            Number of columns in the embeddings matrix.
        embedding_dropout: 
            Dropout rate to apply right after embeddings layer.
        hidden:
            A single integer or a list of integers defining the number of 
            units in hidden layer(s).
        dropouts: 
            A single integer or a list of integers defining the dropout 
            layers rates applyied right after each of hidden layers.
            
    """
    def __init__(self, n_users, n_movies,
                 n_factors=50, embedding_dropout=0.02, 
                 hidden=10, dropouts=0.2):
        
        super().__init__()
        hidden = get_list(hidden)
        dropouts = get_list(dropouts)
        n_last = hidden[-1]
        
        def gen_layers(n_in):
            nonlocal hidden, dropouts
            assert len(dropouts) <= len(hidden)
            
            for n_out, rate in zip_longest(hidden, dropouts):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out
            
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(*list(gen_layers(n_factors * 2)))
        self.fc = nn.Linear(n_last, 1)
        self._init()
        
    def forward(self, users, movies, minmax=None):
        features = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = torch.sigmoid(self.fc(x))
        if minmax is not None:
            min_rating, max_rating = minmax
            out = out*(max_rating - min_rating + 1) + min_rating - 0.5
        return out.float()
    
    
    def _init(self):
        
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)
        init(self.fc)
class MatrixFactorization(torch.nn.Module):
    
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, 
                                               n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, 
                                               n_factors,
                                               sparse=True)
        
    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)  
    
def get_list(n):
    if isinstance(n, (int, float)):
        return [n]
    elif hasattr(n, '__iter__'):
        return list(n)
    raise TypeError('layers configuraiton should be a single number or a list of numbers')