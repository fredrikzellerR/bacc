import abba_settings
import data_utils
import os
import numpy as np
import torch
import math



class data_feed():
    def __init__(self):
        self.datenpakete = None;
        self.feed_order = None; # array mit zufallszahlen
        self.i_data_feed = -1;# ptr in feed order
        self.tmp_i_seq = -1;# ptr ins paket...
        self.i_data_feed_max = -1;

    def ini(self,datenpakete,feed_order,i_data_feed_0,i_data_feed_max):# initial:datenpakete, array mit zufallszahlen, startindex dadrin, endindex, dadrin
        self.datenpakete = datenpakete;
        self.feed_order = feed_order;
        self.i_data_feed = i_data_feed_0;
        self.i_data_feed_max = i_data_feed_max;
        self.tmp_i_seq = 0;# set
        #self.i_validation_data_feed = self.n_train_pakete; in get_ready
        #print('len(self.feed_order)= ' + str(len(self.feed_order)));# 1462
    def feed_constant(self, num_seq):
        # für testmode gibt immer das selbe zurück
        #self.i_data_feed; #nach ini zb 84 rand
        if self.tmp_i_seq == -1:
            return None, None;
        i_paket = 0;
        assert i_paket < len(self.datenpakete);
        paket = self.datenpakete[i_paket];
        if self.tmp_i_seq >= len(paket):
            raise ValueError("feed_constant paket zu ende");
        seq_tar = paket[self.tmp_i_seq];# sequenz innerhalb des paketes

        beatseq = seq_tar[0];
        beatseq_target = seq_tar[1];
        self.tmp_i_seq = self.tmp_i_seq  + 1;
        if self.tmp_i_seq >= num_seq:
            self.tmp_i_seq = -1;
        return beatseq,beatseq_target;

    def feed(self):# batchsize normalerweise 1
        # nacheinander: alle paare seq_targ im paket. 
        # wenn paket zu ende: next paket, wenn keine pakete mehr da: None
        #print('len(self.feed_order)= ' + str(len(self.feed_order)));# 1462
        #print('self.i_data_feed ' + str(self.i_data_feed));# 1097 
        if self.i_data_feed == -1:
            return None,None;

        i_paket = self.feed_order[self.i_data_feed];

        paket = self.datenpakete[i_paket];
        #print('paket len:' + str(len(paket)));
        
        assert self.tmp_i_seq != -1;

        seq_ = paket[self.tmp_i_seq];# sequenz innerhalb des paketes
        
        beatseq = seq_[0]; #(seq_len, 32, 3)
        beatseq_target = seq_[1];
        
        #increment
        self.tmp_i_seq = self.tmp_i_seq + 1;

        if self.tmp_i_seq >= len(paket):
            self.tmp_i_seq = 0;
            # next paket ?
            self.i_data_feed = self.i_data_feed + 1;

            if self.i_data_feed > self.i_data_feed_max: # i_data_feed_max = (self.n_train_pakete + self.n_verifaction_pakete -1)
                self.i_data_feed = -1; # over
                self.tmp_i_seq = -1;# zur sicherheit


        return beatseq,beatseq_target;#batches (67,1,128)
# end class


class abba_loader():

    def __init__(self,seq_len):
        self.i_train_data_feed = -1;#temp ptr
        self.i_validation_data_feed = -1;# temp ptr
        self.all_songs = [];# normale liste, weil verschieden lange songs
        self.seq_len = seq_len;
        self.feed_order =[];
        self.n_validation_pakete = -1;# in getready
        self.n_train_pakete = -1;
        self.datenpakete = [];
        #####################################################################
        self.num_seq_in_paket = 8;# maximalwert...phantasie lieber mehr pakete
        ######################################################################
        self.bTestmode = True;# dadurch batschsize = 1
        self.num_testseq = 1;
        self.mk_paket_step = 1;
        #
        self.bShuffle = False;
        self.tmp_i_seq_validation = -1;
        self.validation_feeder = data_feed();
        self.train_feeder = data_feed();
        #
        self.model_dir = "";
        self.epoch_start = -1;
        
    def _mk_paket(self,song,read_index,seq_len):
        paket =[];
        step = self.mk_paket_step;#int(self.seq_len);#i; #int(self.seq_len/2);#int(1);#self.seq_len / 2
        #print(song.shape);
        len_song = song.shape[0];
        p1 = read_index + seq_len;# endposition
        if( (p1 + 1 ) >= len_song):
            return None, -1;
        c = 0;
        while ((p1 + 1) < len_song) and (c < self.num_seq_in_paket):
            beatseq = song[ read_index : p1 , :, :];
            beatseq_target = song[ (read_index + 1) : (p1 + 1),: ,: ];
            sh = beatseq.shape;
            assert(len(sh) == 3);# zb: 64,1,128
            assert(sh[1] == 1);
            sh = beatseq_target.shape;
            assert(len(sh) == 3);
            assert(sh[1] == 1);
            paket.append([beatseq,beatseq_target]);
            read_index = read_index + step;
            p1 = read_index +  seq_len;
            c = c + 1;

        return paket,read_index;
    #------------------------------------------------
    def shuffle(self):
        if len(self.feed_order) == 0:
            raise ValueError('len feed order = 0');
        if self.bShuffle == True:
            rngenerator = np.random.default_rng();
            rngenerator.shuffle(self.feed_order);
    #------------------------------------
    def load_one_song(self,filepathname):
        song = data_utils.open_data(filepathname);
        return  song;

    def _load(self, scrc_dir, protokoll = True):
        srcfiles_vec = os.listdir(scrc_dir);
        #dump protokoll
        if protokoll == True:
            with  open(self.model_dir + "loaded_train_data_" + str(self.epoch_start)+ ".txt",'w') as file:
                for name in  srcfiles_vec:
                    file.write(name);
                    file.write("\n");
                file.close();
        #----------------------------------------------------------------------------------------------------
        for fl in srcfiles_vec:
            filename = os.path.basename(fl);
            song = self.load_one_song(scrc_dir + filename);
            self.all_songs.append(song);

        return len(self.all_songs);
        
    #end load------------------------------------    
    def variate_sequence_length(self):# uses self.seq_len als Mittel
        m = int(self.seq_len / 4);
        s = np.random.randint(m / 2, m * 2 + 1);
        return s * 4;

    def load_data(self,scrc_dir, model_dir,epoch_start):
        self.model_dir = model_dir;
        self.epoch_start = epoch_start;
        #---
        #scrc_dir = abba_settings.g_dir_beatvecs_net;
        # TEST_DATEN!
        len_all_songs = self._load(scrc_dir);
        
        #datenpakete = [];#[datenpaket,datenpaket,...]
        #datenpaket = [[seq,target], [seq,target],...]
        
        # 10 (seqlen)ner packungen    beatseq, beatseq_target
        # jede datei leseindex
        sum_paketlen = 0;
        read_indexe = [0] * len_all_songs;
        for i in range(0,len_all_songs):
            if read_indexe[i] == -1:# der is fertig
                continue;
            song = self.all_songs[i];
            while(read_indexe[i] != -1):
                seq_len = self.seq_len;#self.variate_sequence_length();## hier seq_len variieren ?
                paket,read_indexe[i] = self._mk_paket(song,read_indexe[i],seq_len);
                if paket is not None:
                    self.datenpakete.append(paket);# [[beatseq,beatseq_target],[beatseq,beatseq_target],[beatseq,beatseq_target]...]
                    sum_paketlen = sum_paketlen + len(paket);
        
        len_datenpakete =  len(self.datenpakete);           
        #zufallszahlen von 0 bis einschl. len(datenpakete) - 1 ist die fütterreihenfolge feed_order
        print("abba loader load_data: " + str(len_datenpakete) + " datenpakete a av. " + str(sum_paketlen / len_datenpakete ) + " sequenzen");

        # zunächst geordnet ... fkt shuffle
        self.feed_order = np.arange(0,len_datenpakete, dtype = np.int);
        #print(self.feed_order);
        assert len_datenpakete > 5;#<-------------------------
        if len_datenpakete > 5:
            # davon 30%verifcation data
            self.n_validation_pakete = int(len_datenpakete * 0.35);
            self.n_train_pakete = len_datenpakete - self.n_validation_pakete;
            self.i_validation_data_feed = self.n_train_pakete;# pointer NACH trainpakete.# kann sein dass noch was kommt TRAIN, VERIFICATION, TEST ?
       
        
        self.i_train_data_feed = 0;
        # in feedorder 1. n_train_pakete 2.n_verifaction_pakete
        # end get ready
    
    #-------------------------------------------------------------------------------
    def get_data_train_ini(self):# muss beim 1. mal kommen
        self.train_feeder.ini( self.datenpakete, self.feed_order,0,(self.n_train_pakete  -1));

    def get_data_validation_ini(self):# muss beim 1. mal kommen
        #datenpakete,feed_order,i_data_feed_0,i_data_feed_max
        self.validation_feeder.ini(self.datenpakete, self.feed_order, self.i_validation_data_feed,(self.i_validation_data_feed + self.n_validation_pakete -1));
    
    def mk_batch(self, feeder,batchsize):
        seqs = np.array([]);
        targs = np.array([]);
        for i in range(batchsize):
            if self.bTestmode:
                seq, tar = feeder.feed_constant(self.num_testseq);
            else:
                seq, tar = feeder.feed();
            if (seq is None) or (tar is None):
                return None, None;
            seqs = np.append(seqs,seq);
            targs = np.append(targs,tar);
        sh = (self.seq_len,batchsize, 128 );
        seqs = np.reshape(seqs, sh);
        targs = np.reshape(targs, sh);
        #print(seqs.shape);
        return seqs, targs;
        
    def get_data_train(self, batchsize):
        #if self.bTestmode == True:
        #    return self.train_feeder.feed_constant(self.num_testseq);
        
        return self.mk_batch(self.train_feeder,batchsize);

    def get_data_validation(self, batchsize): # 
        #if self.bTestmode == True:
        #    return self.validation_feeder.feed_constant(self.num_testseq);
        return self.mk_batch(self.validation_feeder,batchsize);

    def fromKategoriellerWahrscheinlichkeit(self,beatsequenz = None, tresh = 0.9):#np array (len, 128) -> same
        #das mus keine memberfkt sein!
        assert beatsequenz is not None;
        for akk in beatsequenz:
            for i in range(len(akk)):
                #print(akk[i]);
                if akk[i] > tresh:
                     akk[i] = 1;
                else:
                     akk[i] = 0;
        return beatsequenz;


    def zuKategoriellerWahrscheinlichkeit(self,beatseq):
        #obsolet!
        assert False;
        assert beatseq is not None;
        beatseq_p = np.array([]);
        sh = beatseq.shape;
        l = sh[0];
        for j in range(0,l):
            chord = beatseq[j];
            #print(chord);
            one_hot = np.reshape(chord,(128));
            #print(one_hot.shape);
            assert len(one_hot) == 128;
            sum = 0;
            for i in range(0,128):
                sum = sum + one_hot[i];
            if sum > 0:
                p = 1 / sum; 
                for i in range(0,128):
                    if one_hot[i] == 1:
                        one_hot[i] = p;
            #print(one_hot);
            one_hot = np.reshape(one_hot,(1,128));
            beatseq_p = np.append(beatseq_p,one_hot);

        beatseq_p = np.reshape(beatseq_p,sh);
        #print(beatseq_p.shape);
        return beatseq_p;
    def calc_averagevector_train_data(self):#-> np array (128)
        if self.n_train_pakete == -1:
            raise ValueError("calc_averagevector_train_data: keine daten");
        average = np.array( [0] * 128 , dtype = np.float);# datazize = 128 ; fix!
        #print(len(self.datenpakete));
        c = 0;
        for i in range(self.n_train_pakete):
            paket = self.datenpakete[i]; #datenpaket = [[seq,target], [seq,target],...]
            for j in range(len(paket)):
                seq_target = paket[j];
                seq = seq_target[0];
                assert len(seq) == self.seq_len;
                for k in range(self.seq_len):
                    average = average + np.reshape(seq[k],(128));
                    c = c + 1;
          
        
        return average / c;
    #----------------------------------------------------
    def get_all_songs(self,dir):
        if len(self.all_songs) == 0:
            num_songs = self._load(dir,protokoll = False);
        return self.all_songs;
    #----------------------------------------------------
    def allsongs_to_chordslist(self,dir,num_songs = None):
        if dir  is not None:
            num_songs = self._load(dir,protokoll = False);
            songs = self.all_songs;
        else:
            songs = self.all_songs;
            num_songs = len(songs);

        all_chords = [];
        for i in range(num_songs):
              song = songs[i];
              for j in range(len(song)):
                  all_chords.append(song[j][0]);
        del songs;
        return all_chords;
##end abba loader


def g_fromKategoriellerWahrscheinlichkeit_seq(beatsequenz = None, tresh = 0.9):#np array (len, 128) -> np array
        # Format seq_len, 128 .... keine batch
        # kommazahlen zwischn 0 und 1 zu 0 oder 1
        assert beatsequenz is not None;
        #print(beatsequenz);
        for akk in beatsequenz:
            for i in range(len(akk)):
                #print(akk[i]);
                if akk[i] > tresh:
                     akk[i] = 1;
                else:
                     akk[i] = 0;
        return beatsequenz;

def g_zuKategoriellerWahrscheinlichkeit_akkord(akkord):#np -> np
        assert akkord is not None;
        sum = akkord.sum();
        if sum == 0:
            return akkord;
        return  akkord / sum;# broadcast
        

def g_zuKategoriellerWahrscheinlichkeit_seq_batch(beatseq):# np -> np
        assert(len(beatseq.shape) == 3);# MIT BATCH! (4,1,128)
        assert beatseq is not None;
        seq_len, batchsize, datasize = beatseq.shape;
        beatseq_p = np.array([]);
        for i in range(seq_len):# von links nach rechts arbeiten
            chords = beatseq[i,:,:];
            for j in range(0,batchsize):
                chd = chords[j,:];
                beatseq_p = np.append(beatseq_p, g_zuKategoriellerWahrscheinlichkeit_akkord(chd)); 

        return np.reshape(beatseq_p,(seq_len, batchsize, datasize));

def g_NumberToBits(n :int, bits = 16):# used beim Versuch chordvec in Zahl umzurechnen
    v = [0]*bits;
    # von rechts nach links msb <-------  lsb
    for i in range(bits):
        b = n & 1;
        v[bits - i - 1] = b;
        n = n >> 1;
    return v;

def g_bitsToNumber(a_bits):# array -> int
    l = len(a_bits);
    n:int = 0;
    _2_hoch : int = 1;
    for i in range(l-1, - 1, - 1):# von rechts nach links msb<-------------lsb
        bit = a_bits[i];
        n +=  (_2_hoch * bit);
        _2_hoch = _2_hoch << 1;
    return n;

def g_toNumberSequenz_chord(chord,input_dim):#np ->  list 
    # 128 bit wird zb. zerlegt in 8 * 16 bit dann clamp (-1, 1)
    
    assert len(chord) == 128;
    bits = int(128 / input_dim);
    # max number
    assert bits == 16;
    max = 65535.0;# input dim = 8,bits = 16
    num_nums = int(128 / bits);# = 8
    d_array = [None]*num_nums;

    for i in range(0, num_nums):
        j = i * bits;
        n_bits = chord[j:j + bits];
        #bits2number
        n = g_bitsToNumber(n_bits);
        if n == 0:
            d_array[i] = 0.0;
            continue;
        #clamp [-1,1]
        d =((n * 2.0) / max ) - 1.0;
        d_array[i] = d;
        
   
    return d_array;

def g_toNumberSequenz_seg_batch(chordseq, input_dim):#np -> np
    assert(len(chordseq.shape) == 3);# MIT BATCH! (4,1,128)
    assert chordseq is not None;
    seq_len, batchsize, datasize = chordseq.shape;
    chprdseq_num = np.array([]);
    for i in range(seq_len):# von links nach rechts arbeiten
        chords = chordseq[i,:,:];
        for j in range(0,batchsize):
            chd = chords[j,:];
            #print(chd);
            chd_n = g_toNumberSequenz_chord(chd,input_dim);
            #print(chd_n);
            #test backward!
            #print(np.asarray(g_nChord_to_multiple_hot128(chd_n)));
            chprdseq_num = np.append(chprdseq_num, chd_n); 

    return np.reshape(chprdseq_num,(seq_len, batchsize, input_dim));# 8 * 16 bit clamped to[-1,1]
    

def g_nChord_to_multiple_hot128(nChord):
    vec = [0.0] * 128;
    l = len(nChord);
    bits = int (128 / l);
    assert bits == 16;
    max = 65535.0;# input dim = 8,bits = 16
    for i in range(l):
        j = i * bits; # schreibpointer in vec
        d = nChord[i];
        if d == 0:
            continue;
        n =  int( round( (d + 1.0) * max / 2.0  ));
        
        # zu bits
        v = g_NumberToBits(n ,bits);
        for k in range(bits):
            vec[j + k] = v[k];
        
    #entclampen
    return vec;
def g_Numberseqenz_to_multiple_hot128(n_seq):# umkehrfkt
    l = n_seq.shape[0];
    mh_vecs = [None] * l;
    for i in range(l):
       mh_vecs[i] = g_nChord_to_multiple_hot128(n_seq[i]);
        
    mh_vecs = np.asarray(mh_vecs).reshape(l, 128);
    return mh_vecs;