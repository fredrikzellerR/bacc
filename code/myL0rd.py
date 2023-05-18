
import abba_settings
from gatel0rd import GateL0RD
import gatel0rd_model as GateL0RD
import rnn_model as RNNs
import torch
import torch.nn as nn
import data_utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import billiard_ball_dataloader as bb_dataloader
import my_experiments
import data_utils
import globals
import abba_midi_utils
from abba_dataloader import g_fromKategoriellerWahrscheinlichkeit_seq
from torch import Tensor
from testmodell import tModel as tModel
import math
# keine gute Idee' ?
class _trivial_loss(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, pre, tar):
        ctx.save_for_backward(pre,tar);
        differenz = torch.sub(pre,tar);
        # nur positive 1nsen, dann summe
        return torch.mul(differenz,differenz).sum(); 
    @staticmethod
    def backward(ctx, grad_output):
#        grad_output = gradient of the loss
        # "Part Ableitung nach y " = 2y_i - 2x_i
        #  x, y gleich -> 0
        # x= 0, y = 1 -> -2; x=1 , y = 0  -> 2
        #print(grad_output);
        y, x = ctx.saved_tensors;# y = pre, x = tar
        result = grad_output * (torch.sub(y,x) * -2);
        
        return result, None;# hier müssen 2 zurück weil in Foreward 2 rein
    
class _HeavisideST(torch.autograd.Function): 
    """
    Heaviside activation function with straight through estimator
    """

    @staticmethod
    def forward(ctx, input):
        
        #print('_HeavisideST:');
        #print(input);
        #print('============================================');
        sz = input.size();
        assert abba_settings.g_this_net is not None;
        if (abba_settings.g_this_net.training == True): 
            input_ = input - torch.rand(sz).to(abba_settings.device);
        else:
            input_ = input - 0.5;
        
       
        return torch.ceil(input_).clamp(min=0, max=1).to(abba_settings.device);

    @staticmethod
    def backward(ctx, grad_output):
        # durchwinken.... kommt was!
        #print(grad_output);
        #print(ctx);
        return grad_output.clone();

class _heavyside_step_wrapper(nn.Module):
    def __init__(self):
        super(_heavyside_step_wrapper,self).__init__();
        self.fkt = _HeavisideST.apply;
        
    def forward(self,input:Tensor)  -> Tensor:
        return self.fkt( input);#
    



def adapt_hs_shape(hs, newshape):
    if hs is None:
        return None;
    #print(hs.grad_fn);#grad_fn=<StackBackward0>) ???
    shape2 = hs.shape[2];
    adaptet_hs_vec = None;
    shape_1_hs = hs.shape[1];
    new_shape_1 = newshape[1];
    if shape_1_hs == new_shape_1:
        return hs;
    adapted_hs = [];
    hs = torch.squeeze(hs);
    li_hs = hs.tolist();
    #print(li_hs);
    # Fall 1 hs zu klein ... nullen anhängen
    if shape_1_hs < new_shape_1:
        adapted_hs = li_hs;
        while shape_1_hs < new_shape_1:
            h = [1] * shape2 ;
            adapted_hs.append(h  );
            shape_1_hs = shape_1_hs + 1;
        adaptet_hs_vec = torch.tensor(adapted_hs ,requires_grad = True, dtype = torch.float);
        adaptet_hs_vec = torch.unsqueeze(adaptet_hs_vec,0);
       # print(adaptet_hs_vec.shape);
        return adaptet_hs_vec;

    # Fall 2 hs zu gross von vorne löschen (hinten bleibt = LAST hs...)
    if shape_1_hs > new_shape_1:
        i = shape_1_hs - new_shape_1;
        while i < shape_1_hs:
            adapted_hs.append(li_hs[i] );
            i = i + 1;
       # print(len(adapted_hs))
        adaptet_hs_vec = torch.tensor(adapted_hs,requires_grad = True ,dtype = torch.float);
        adaptet_hs_vec = torch.unsqueeze(adaptet_hs_vec,0);
      #  print(adaptet_hs_vec.shape);
        return adaptet_hs_vec;
    return None;

def restore_dummies(beatvector, n_dummies):# LIST
    for i in range(n_dummies):
        beatvector.append([-1,-1,-1]);
    return beatvector;

def filter_dummies(beatvector):
    n_dummies = 0;
    clean_beat_vector = [];
    for i in range(len( beatvector)):
        n = beatvector[i];
        if n[0] + n[1] + n[2] == -3:
            n_dummies = n_dummies + 1;
            continue;
        clean_beat_vector.append(n.tolist());
    return clean_beat_vector,n_dummies;

def predict_one_step(model,sequenz,max_iter,seq_len,batch_size,data_size,factor_output,latent_dimension, use_warm_up = True):
    # NUR die vorhergesagten zurück
    
    assert sequenz.shape[0] > 2;
    assert seq_len > 2;
    t_start = 0;
    seq_pred = [];
    hidden_states = [];
    h_tlast = None;
    model.use_warm_up = False;
    if use_warm_up:
        model.use_warm_up = True;
        obs_init_batch = sequenz[0:2,:,:];# TODO soviel wie in params definiert
        h_tlast = model._GateL0RDModel__warm_up(obs_init_batch,0);
   
        h_tlast = torch.unsqueeze(h_tlast,0);#1. dimension Layer nummer
        t_start = 2;
    #test
    action_t = None;
    #action_t = torch.tensor( torch.zeros(1 * batch_size * data_size));
    #action_t = torch.reshape(action_t,(1 , batch_size , data_size));
    for t in range(t_start,seq_len):
        notes_in_beat = batch_size;
        beat_t = sequenz[t,:,:];
        beat_t_filtered ,n_dummies = filter_dummies(beat_t);
        if len(beat_t_filtered) == 0:
            continue;
        
        beat_t = torch.tensor(beat_t_filtered,dtype = torch.float);
        beat_t = torch.unsqueeze(beat_t,0);#1,batchsize,3
        inputshape = beat_t.shape;
        #print('inputshape = ' + str(inputshape));
        notes_in_beat = beat_t.shape[1];
        #------
        #inputshape = torch.Size([1, 21, 3])
        #h_sshape = torch.Size([1, 3, 8])
        # hsshshape muss in dim = 1 gleich sein, also 1,21,8
        h_tlast = adapt_hs_shape(h_tlast,inputshape);

        y, h_t,  g_regs, ys, delta_out = model.forward_one_step(beat_t, predict_deltas = False, factor_delta = factor_output, action_t = action_t, h_tminus1 = h_tlast);#
        h_tlast = h_t;
        #print('h_sshape = ' + str(h_t.shape));
        new_beat = torch.squeeze(y);
        #print(new_beat.size());
        #restore dummies
        new_beat_list = new_beat.tolist();
        new_beat_list = restore_dummies(new_beat_list,n_dummies);
        assert len(new_beat_list)==  batch_size;
        seq_pred.append(new_beat_list);

    return seq_pred,hidden_states
        
def vorhersage_n_steps(model,test_song, sequenz, max_iter, seq_len, batch_size,data_size,factor_output,latent_dimension, limit_data, params = None, prob2cat_treshold = 0.99):
    # testsong komplett wegen vergleich
    model.use_warm_up = False;
    if (params.get( "warm_up_layers" ) > 0) and (params.get( "warm_up_inputs" ) > 1):
        model.use_warm_up = True;
    # -> dieVORHERGESAGTEN im Format (seqlen, 128)
    val_ss = np.ones((seq_len, batch_size));#Wahrscheinlichkeit für echten input immer 1
    # print(pts.shape);
    seq_pred = np.array([], dtype = np.float);
    

    losses = [];
    hidden_states = [];
    gate_regs =[];
    out_hiddens = [];
    l_hs = None;
    

    for iter in range(0, max_iter):
        #target = test_song[ (iter + 1) : (seq_len + iter + 1),:,:];
        #assert(target.shape ==(seq_len,1,128));
        #target_tensor = torch.tensor(target,dtype = torch.float, requires_grad = True).to(abba_settings.device);
        #------------------
              
        #y, z, gate_reg,   out_hidden, deltas = model.forward_n_step_freddy(obs_batch = sequenz, train_schedule = val_ss, predict_deltas = False, factor_delta = factor_output, last_hidden_state = None);#
        y, z, gate_reg, out_hidden, deltas = model.forward_one_step_freddy(sequenz, False, 0, None,  None, None);
        # z = hiddenstates
        # loss im vergleich zum original (test_song) ? target_tensor
        # sonst loss zur eigenen vorhersage
        # assert(y.size() == target_tensor.size());
        
        LOSS = model.loss(y,sequenz, gate_reg).detach().item();

        losses.append(LOSS);
        
        #das sind die hidden states!
        if out_hidden is not None:
            sz = out_hidden.size();# [16,64] seqlen hiddensize vgl. gateL0rdmodel forward_one_step_freddy
            #etzten abknipsen
            l_outh = (out_hidden[(sz[0] -1),:]).cpu();# 
            #l_outh = torch.squeeze(l_outh);# letzter beat : für 8 Noten 11 gates8 * 11
            l_outh_np = l_outh.detach().numpy();#
            #print(l_outh_np);
            hidden_states.append(l_outh_np);
        #if z is not None: # z unused
        #    sz = z.size();
            #print(sz);#
       #     assert len(sz) == 4; # (num_layers , seq_len, batch_size, latent_dim)  [1, 4, 8, 11],[2, 12, 1, 32]
        #    if  params.get("network_name") =="GateL0RD":
                #nehme den vom letzten layer #batchsize bei predict immer 1 !?
         #       z = z[sz[0] - 1,:,:,:];
         #       z = torch.reshape(z, (seq_len,batch_size, latent_dimension));#  geht jetz wg. "num_layers = 1 "!
        #        l_hs = z[len(z) -1,:, :];# braucht size 1,32,8   
                #----hiddenstates zur testausgabe----
                    #nur den letzten z 
        #        lhs_np = l_hs.detach().cpu().numpy();#
        #        hidden_states.append(lhs_np);
        #    else:# für Testnetz
        #        l_hs =  torch.flatten(z.detach().clone());
        
        
        #--------------------------------
        if gate_reg is not None:
        #von gateregs nur LETZTEN zur Ausgabe
            sz = gate_reg.size();# Layers, seqlen, batchsize, hiddensize 
            l_gr = (gate_reg[  0 , (sz[1] -1)  , :, :] ).cpu();# geht wg. nur letzter layer!.... normalerweise num_layers = 1, 4 Beats x für 8 noten 11 gates
            l_gr = torch.squeeze(l_gr);# letzter beat : für 8 Noten 11 gates8 * 11
            #print(l_gr.size());
            lgr_np = l_gr.detach().cpu().numpy();#
            gate_regs.append(lgr_np);
        
   
        #-----------------------------------------
        #letzen punkt ... vorhersage ans ergebnis anhängen
        y_tplus1_np = y[ len(y) - 1].cpu().detach().numpy();
        lossfkt = params.get("lossfunction");
        

        if( lossfkt == "CNN"):
         #   print(y_tplus1_np)
            y_tplus1_np = g_fromKategoriellerWahrscheinlichkeit_seq(y_tplus1_np, prob2cat_treshold);# tresh = ? abba_dataloader.
        
        #print(y_tplus1_np);
        seq_pred = np.append(seq_pred, y_tplus1_np[0],axis = 0);
        
        #--------------------------------------------
        #neue inputsequenz bauen
        #ausschnitt bleibt gleich 
        old_shape = sequenz.size();
        np_seq = sequenz.cpu().detach().numpy();
        np_seq = np.append(np_seq,y_tplus1_np);
        # 1ns länger...
        np_seq = np.reshape( np_seq , (old_shape[0] + 1, data_size));# 2Dim
        # ausschneiden ...
        np_seq = np_seq[1 : np_seq.shape[0],:];
        # tensor
        
        sequenz = torch.tensor(np_seq, dtype = torch.float , device = abba_settings.device, requires_grad = True).to(abba_settings.device);
        sequenz = torch.reshape(sequenz, (seq_len, batch_size , data_size)).to(abba_settings.device);
        
        #-------------------------------------------------------------
        
    seq_pred = np.reshape(seq_pred,( max_iter,data_size)).astype( np.float);
    return seq_pred, hidden_states,gate_regs,out_hiddens,losses;


def load_model_from_checkpoint(pathname_checkpoint,params, input_dim):
    
    # model mus genauso konstruiert werden, wie es gespeichert wurde
    
    #input_dim = 128;
    seq_len = params.get("abba_seq_beats_train_len");#soviele beats - abbasettings 10  

    output_dim = input_dim;

    network_name = params.get("network_name");
    factor_output = params.get("factor_output");
    latent_dimension = params.get("latent_dim");
    feature_dim = params.get("feature_dim");
    num_layers = params.get("layer_num");
    num_layers_internal = params.get("num_layers_internal");
    reg_lambda = params.get("reg_lambda");
    f_pre_layers = params.get("preprocessing_layers");
    f_post_layers = params.get("postprocessing_layers");
    f_init_layers = params.get("warm_up_layers");
    f_init_inputs = params.get("warm_up_inputs");
    stochastic_gates = params.get("stochastic_gates");
    gate_noise_level = params.get("gate_noise_level");
    gate_type = params.get("gate_type");
    learning_rate = params.get("lr");
    lossfunc = params.get("lossfunction");
    modelnummer = params.get("my_number");

    if network_name == "GateL0RD":
        model = GateL0RD.GateL0RDModel(input_dim = input_dim, output_dim = output_dim, latent_dim = latent_dimension,
                                             feature_dim = feature_dim,                                                    # featuredim ?
                                             num_layers = num_layers,                                        
                                             num_layers_internal = num_layers_internal, 
                                             reg_lambda = reg_lambda,
                                             f_pre_layers = f_pre_layers,
                                             f_post_layers = f_post_layers,
                                             f_init_layers = f_init_layers, 
                                             f_init_inputs = f_init_inputs,
                                             stochastic_gates = stochastic_gates,
                                             gate_noise_level = gate_noise_level,
                                             gate_type = gate_type,
                                             slossfct = lossfunc);
    else:
        # copy from abba_net01.py
        #infostr = network_name + " - " + str(modelnummer) + " - " + lossfunc;
        #print(' loading ', infostr);
        if network_name == 'LSTM':
            model = tModel(input_dim, output_dim, params.get("latent_dim"), params.get("layer_num") ,seq_len , slossfct = lossfunc);
        else:
            if network_name == 'LSTMFAN':
                rnn_Type = "LSTM";
            elif network_name == 'RNNFAN':
                rnn_Type = "ElmanRNN";
            elif network_name == 'GRUFAN':
                rnn_Type = "GRU";

            model = RNNs.RNNModel(input_dim = input_dim, output_dim = output_dim, latent_dim = params.get("latent_dim"),
                               feature_dim = params.get("feature_dim"), num_layers =params.get("layer_num"),
                               rnn_type  = rnn_Type,   f_pre_layers = params.get("preprocessing_layers"),
                               f_post_layers = params.get("postprocessing_layers"), f_init_layers = params.get("warm_up_layers"),
                               f_init_inputs = params.get("warm_up_inputs"),slossfct = lossfunc);
        ##
        


    #learning_rate = 0.01;
    #optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate); # optimizer of your choice
    dic_checkpoint = torch.load(pathname_checkpoint);# dictionary 
    model_state_dic = dic_checkpoint['model_state_dict'];
    model.load_state_dict(model_state_dic);#OrderedDict #
    optimizer = model.get_optimizer(learning_rate);
    optimizer.load_state_dict(dic_checkpoint['optimizer_state_dict']);
    epoch = dic_checkpoint["epoch"];
    #loss = dic_checkpoint["loss"];# nicht gespeichert
    
    
    return model, optimizer, epoch;

def _is_torch_loss(criterion):# -> bool
    type_ = str(type(criterion)).split("'")[1];
    parent = type_.rsplit(".",1)[0];
    if parent == "torch.nn.modules.loss":
        return True;
    return False;

def loss_all_loop(y_all,tar_all, func):
    assert(y_all.size() == tar_all.size()); # [n, 128]
    n = y_all.size()[0];
    datasize = y_all.size()[1]; #= 128
    loss = torch.tensor( 0.0,  dtype= torch.float64, requires_grad = True).to(abba_settings.device);# ZERO
    for i in range(n):
        y = y_all[i];
        tar = tar_all[i];
        l = func( y, tar);
        loss = torch.add(loss,l);#<AddBackward0 object at 0x0000020429507E08>
    return loss / n;


def _nnCE(y, tar):
    f = nn.CrossEntropyLoss();#weight = abba_settings.g_CE_weigths
    return f(y,tar);
def _myCrossEntropy(y, tar):# kommen beide y entweder [0,... 1] (sigmoid) oder [0 | 1] heavyside tar immer [0 | 1] 
    ############################################################
    #  H(p,q) = Sum_x_in X  p(x) * log( q(x))
    ############################################################
    #print(y.sum());
    f_softmax = nn.Softmax(dim = 0);
    #y_p =   y; # kommt schon von softmax
    
    
   
    p_tar = f_softmax(tar);
    p_y = f_softmax(y);# falls predict heavyside
   
    #print(m);
    n = torch.mul(torch.mul(p_tar, torch.log(p_y )),abba_settings.g_CE_weigths);
    #print(n);
    l = -n.sum();# gewichtete summe

    #
    #f = nn.MSELoss();
    
    ##l = f(y_p,tar_p);
    assert math.isnan(l.item()) == False;
       
    #print(y);
    #print(tar);
    #print( f(y,tar));
   # l = -torch.mul(tar_p,torch.log(y_p)).sum();# = ln
    return l;

def _mse(y,tar):
    n_y = y.sum();
    n_tar = tar.sum();

    if (n_tar == 0) and (n_y == 0):
        return 0;
            
    if (n_tar != 0):
        tar = tar / n_tar;

    if (n_y != 0):
        
        y = y / n_y;

    return nn.MSELoss()(y,tar);

def _loss_trivial(y,tar):# tensor 1dim,tensor 1dim -> tensor, number
    # für jeden der nicht gleich war gibts abzug
    differenz = torch.sub(y,tar);
    # nur positive 1nsen, dann summe
    return torch.mul(differenz,differenz).sum(); # ? * 100 / len(y) ?  in Prozent bei kürzeren Vektoren sind "2" fehler schlimmer als bei längeren. Für ableitung egal: Konstante

def _loss_angle(y,tar):# tensor 1dim,tensor 1dim -> tensor, number
    #versuch mit vec Winkel.

    y_norm = torch.norm(y);
    tar_norm = torch.norm(tar);
    _1minus_theta_cos = 0;
    if tar_norm == 0.0:#pause!
            if y_norm != 0.0:# falls auc 0 gut
                _1minus_theta_cos = 1;#  nullvec sekrecht zu allem
    else:
            _1minus_theta_cos = 1.0 - (torch.dot( y ,tar  ) /  ( y_norm * tar_norm  ));

    assert  math.isnan(_1minus_theta_cos) == False;
        
    return _1minus_theta_cos;
#-------------------------------------------------------------------
def loss_(y, target, lossfunction):# y x lossfkt:string
    #fkt =  nn.CrossEntropyLoss() <-------------- comes from gateL0rd_model.loss  BCE loss
    # es interessiert nur der letzte Akkord vom y, der muss gleich sein wie letzter von Target
    #       A  B  C  D  E   F
    # y_in  --------------|
    #y_out     |----------- ?|
    #  tar     |-------------|
    #

    #assert( _is_torch_loss(loss_fkt) );# braucht man dann nich mehr... loss fkt hier direkt
    #print(y.size());

    assert len(y.size()) == 3;
    seq_len, batchsize, datalen = y.size();
   
    loss_all = torch.tensor( 0.0,  dtype= torch.float64, requires_grad = True).to(abba_settings.device);# ZERO
   
    for i_batch in range(batchsize):
        
        y_chords = y[:,i_batch,:]; # = alle akkorde aus einer batch
        
        target_chords = target[:,i_batch,:];
        #print(target_chords);
        assert y_chords.size() == target_chords.size();
        func = None;
        if lossfunction == "BCE":
            func = nn.BCELoss();
            
            #l = nn.BCELoss()(y_chords[seq_len - 1,:] , y_chords[seq_len - 1,:]);
        elif lossfunction == "CNN":
            func = nn.CrossEntropyLoss();
            # y: [0.5008, 0.5145, 0.4812,  ...,  Wahrscheinlichkeiten
            #target[0,....1.0, 0. ] die 1nsen sind 100% Wahrscheinlicjhkeit
            l = loss_all_loop(y_chords,target_chords,nn.CrossEntropyLoss());
            
            #l = _nnCE( y_chords[seq_len - 1,:], target_chords[seq_len - 1,:]);
        elif lossfunction == "TRI":
            func = _loss_trivial;# _trivial_loss.apply; #_loss_trivial;#
            #l = _loss_trivial( y_chords[seq_len - 1,:], target_chords[seq_len - 1,:]);# grad_fn=<SumBackward0>)
        assert( not func is None);
        l = loss_all_loop(y_chords,target_chords,func);
        loss_all = torch.add(loss_all,l);#<AddBackward0 object at 0x0000020429507E08>
        
    return loss_all / batchsize;#<DivBackward0 object at 0x0000020439BB7588>
        
        
   

   




def ini_module_weights(_linear: nn.Module):#nn.Module
    #print(_linear.weight); macht immo nix
    
    sz = _linear.weight.size();
    assert len(sz) == 2;
    
    #w = torch.rand(sz[0] * sz[1]);# uniform distribution
    #w = torch.zeros(sz[0] * sz[1]);
    #rand, ones
    #w = torch.randint(0,2,(sz[0]*sz[1],));
    w = torch.reshape(torch.ones(sz[0] * sz[1]),sz);
    #w = torch.reshape(w,sz);
    torch.nn.init.xavier_uniform_(w);
    #print(w);
   
    with torch.no_grad():
        _linear.weight.copy_(w);

    #print(_linear.weight);
    return _linear;
    #----------------
