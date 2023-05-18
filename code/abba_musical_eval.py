import abba_settings # IMMER BEI DIESEM PROJECT
import pickle
import numpy as np;
import torch
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
import data_utils
from  abba_dataloader import abba_loader # konflict mit abba_net01
from myL0rd import _loss_trivial #???
import data_analys
import pythonUtils.List.myListUtils as myListUtils # in abba settings: sys.path zu runtime

# konstruiere nachträglich klassen (quasi Datentypen) , damit das besser lesbar ist
class ModulationsData():
    def __init__(self,md):
        self.md = md;
        self.iterator_hinmod = 0;
        self.iterator_rueckmod = 0;
    def get_hinmodulationen(self):
        return self.md[0];
    def get_rueckmodulationen(self):
        return self.md[1];
    def get_from_beat(self, modlists, i_data, i_beat):# rekursiv ;
        # moddata = listoflists [[24,43,44...],[...] ] entweder Liste der hina oder Liste der Ruecks
        # returnt liste >= geforderter beat ab i_data
        if modlists is None:
            return -1,i_data;
        if i_data >= len(modlists):
            return -1,i_data;
        li = modlists[i_data];
        if li is None:
            return  -1,i_data;
        if len(li) == 0:
            return  -1,i_data;
        #hier.....
        if (li[0] >= i_beat):
            return li[0],i_data;
        return  self.get_from_beat(modlists, i_data + 1, i_beat );

    def nextHinmod(self):
        hinmods = self.get_hinmodulationen();
        if len(hinmods) <= self.iterator_hinmod :
            return None;
        else:
            hm = hinmods[self.iterator_hinmod];
            self.iterator_hinmod += 1;
            return hm;
    def nextRueckmod(self):
        rueckmods = self.get_rueckmodulationen();
        if len(rueckmods) <= self.iterator_rueckmod :
            return None;
        else:
            rm = rueckmods[self.iterator_rueckmod];
            self.iterator_rueckmod += 1;
            return rm;


class AkkordBassData():
    def __init__(self,ab_data):
        self.ab_data = np.asarray(ab_data);#-> Akkorde, Bässe #shape  84,2
        
    def split(self):
        return self.ab_data[:,0], self.ab_data[:,1];
    def get_Akkorde(self):
        return self.ab_data[:,0];
    def get_Baesse(self):
        return self.ab_data[:,1];
    


class musicEvalUtils():
    def __init__(self):
        self.txt_notes_oct =["C","C#","D","Eb","E","F","F#","G","Ab","A","B","H"];# spelling ?

    def chord_bassnote_mid(self,chord):
        # Suche Bass im Akkord = unterste note wenn < Treshold
        ch_mid = np.array(self.categorical_to_midnotes(chord));
        if len(ch_mid) == 0:# pause
            return -1;
        i_min = np.argmin(ch_mid);
        bassnote = ch_mid[i_min];
        if bassnote > 60:# treshold
            bassnote = -1;
        return bassnote;

    def extract_basses(self,song):#pro song
        basses = [-1 ] * len(song);
        for i in range(len(song)):
            chord = myListUtils.np_squeeze_or_not(song[i]);# 
            if np.sum(chord) > 0:
               basses[i] = self.chord_bassnote_mid(chord);
        return basses;

    def extract_akkordBass_data(self, listofSongs):# -> [[Akk-oktave],bass] for all songs
        ab_data = [];
        for song in listofSongs:#song = numpy ? (x,1,128)
            len_song = len(song);
            data = [None] * len_song;
            #array i [Akkord-in-Oktave, bass] exzerpieren...
            for i in range(len_song):
                chord = myListUtils.np_squeeze_or_not(song[i]);# nicht immer numpy ?
                cord_octave = self.clamp_chord_to_octave_categorical(chord);
                bass = -1;
                if np.sum(chord) > 0:
                    bass = self.chord_bassnote_mid(chord);
                data[i] = [cord_octave,bass]
            ab_data.append(data);
        return ab_data;
    
    def clamp_chord_to_octave_categorical(self, chord):# np -> np(int) ???
        octave = np.array([0] * 12, dtype = np.int);
        for i in range(len(chord)): #should be 128
            if chord[i] == 1:
                i_oct = i % 12;
                octave[i_oct] = 1;
        return octave;

    def empty_chord_categorical(self,size):
        return np.array([0] * size);

    def midnote_to_text(self, n):
        if n == -1:
            return "-";
        return self.txt_notes_oct[ n % 12];

    def categorical_to_midnotes(self, chord):
        midnotes = [];
        for i in range(len(chord)):
            if chord[i] == 1:
                midnotes.append(i);
        return midnotes;

    def midinotes_to_categorial(self, ch, size):
        chord = self.empty_chord_categorical(size);
        for n in ch:
            assert n < size;
            chord[n] = 1;
        return chord;

    def sum_song(self,song):#
        sum = 0;
        for i in range(len(song)):
            sum = sum + np.sum(song[i]);
        return sum;

    def next_valid_bass(self, basses, iStart):# bassarray -> index != -1 , iStart inclusive
        for i in range(iStart,len(basses)):
            if basses[i] != -1:
                return i;
        return -1;

    def next_valid_bass_skip_wh(self, basses, iStart, note2skip):# bassarray -> != -1 , iStart inclusive
        
        k = self.next_valid_bass(basses, iStart);
        while (k != -1) and (note2skip == basses[k]) and (k < len(basses)):
               k = self.next_valid_bass(basses, k + 1);
                    
        return k;

    def midi_array_2_octave(self, a):
        a_oct = [None] * len(a);
        for i in range(len(a)):
            a_oct[i] = a[i] % 12;
        return a_oct;

    def find_bass_line(self, basses, line):# l-> list of Index
        #Ton wiederholungen werden geskippt
        
        l_bass = len(basses);
        l_line = len(line);
        matches = [];
        ibass = 0;
        while ibass < (l_bass - l_line):
           k = ibass;
           match = True;
           for j in range(l_line):
               
               k = self.next_valid_bass(basses, k);
               if k == -1:
                  match = False;
                  break;

               bassnote = basses[k] % 12;
               if bassnote != (line[j] %12):
                   match = False;
                   break;
               # vorspulen , über WH drüber
               while (bassnote == basses[k] % 12) and (k < (l_bass - l_line)):
                     k += 1;
                     k = self.next_valid_bass(basses, k);
                     if k == -1:
                         break;

               if (k == -1) or (k >= (l_bass - l_line)):
                   match = False;
                   break;

               
               
           #efoj
           if match == True:
              matches.append(ibass);

           # ibass vorspulen
           bassnote = basses[ibass];
           ibass += 1; # auf jeden fall...
           while (bassnote == basses[ibass]) and (ibass < (l_bass - l_line)):
                    ibass += 1;
        #wend   

        
        
       
        return matches;
##################################################

   
class triadfinder():

    def __init__(self):
        self.aMidivalues = [];
        
        self.musicEvalUtils = musicEvalUtils();
        self.a_chordnames = ["ma","mi" ,"7-ma","v","ma46"];# auch 46 Akkord! diese Liste ausbaubar...

    def triadinfo(self,chord_categorical_octave, bass, sName):
        return chord_categorical_octave,bass,sName;#list ?...

    
    
    def calc_possible_chords(self, chord, compare_chords, bass, a_names):#chord ist der fragliche-categorical, comparechord die auswahl ->liste mit TriadInfos, loss
        
        assert(len(compare_chords) == len(a_names));# in a_names stehen die Namen der 'comparechords'

        losses = self._list_match_triad_trivial_loss(compare_chords, chord);
        #ohne konstante... nur minimales loss zählt
        treshold =  losses[ np.argmin(losses)];# self.TriadFinder.const_min_treshold_trivial_loss(12);#
        i_possible_triads = myListUtils.np_leq_indizi(losses,treshold);# 
        if(len(i_possible_triads) == 0 ):
            return [], None;

        #TODO: nach loss aufsteigend sortieren !
        # entfällt weil alle losses minimal

        TriadInfos = [];# Liste der möglichen... ev mehrere ... konstruieren
        loss = 0.0;
        for  i in i_possible_triads:
             ch = self.musicEvalUtils.categorical_to_midnotes(compare_chords[i]);
             TriadInfos.append( self.triadinfo(ch , bass, self.midnote_to_text(bass) + "\n" + a_names[ i ])  );# akkord im midinoten oktave format!
             loss = loss + losses[i]; 

        loss = loss * len(i_possible_triads);# je mehr desto schlecht ?
        return TriadInfos, loss;

    def empty_chord_categorical(self,size):return self.musicEvalUtils.empty_chord_categorical(size);

    def midnote_to_text(self, n):
        return self.musicEvalUtils.midnote_to_text(n);
        

    def categorical_to_midnotes(self, chord):
        return self.musicEvalUtils.categorical_to_midnotes(chord);
        

    def midinotes_to_categorial(self, ch, size):
        return self.musicEvalUtils.midinotes_to_categorial(ch, size);
        
    ##################################################################################
   
    def _find_interval_or_complement(self,chord, lo, interval):# suche intervall drüber oder complementärintervall unter lo. chord: Midi in lowest octave, lo ebso. -> midi :int (nix = -1)
        # intervall 7 = Qiunte 8 =kl.Sexte etc.
        # drüber ?
        for n in chord:
            if (n - lo)  == interval:
                return n;
        intervall_comp = 12 - interval;
        # komplement drunter
        for n in chord:
            if (lo - n ) == intervall_comp:
                return n;

        return -1;
    def _chord2tensor(self,chord):
        return torch.tensor(chord, dtype = torch.float).to(abba_settings.device);

    def const_min_treshold_trivial_loss(self, lenarray):## zum vergleichen!
        return 1.0 / lenarray;

    def _chord_trivial_loss(self, x, y):# x= predict y = target
        #trivial loss
        assert(len(x) == len(y));
        x_np = np.asarray(x);
        y_np = np.asarray(y);
        s = np.subtract(x_np,y_np);#[0 , -1, 1, 0 ... alle die gleich waren = 0]
        
        return np.sum( np.multiply(s,s) ) / len(x);# 'im quadrat' neg weg diffbar
        

    def _list_match_triad_trivial_loss(self,list_triads, triadseek):# alle categorical...
        
        loss = np.array([None] * len(list_triads), dtype = np.float);
        for i in range(len(list_triads)): 
            loss[i] =  self._chord_trivial_loss(triadseek,list_triads[i]);
        return loss;

    def make_scale_mid(self,fund_key,flag):
        scale =  None;
        if flag == "ma":
            scale = [0] * 7;
            scale[0] = fund_key;    #c
            scale[1] = fund_key + 2;#d
            scale[2] = fund_key + 4;#e
            scale[3] = fund_key + 5;#f
            scale[4] = fund_key + 7; # g
            scale[5] = fund_key + 9; # a
            scale[6] = fund_key + 11; # h
        
        elif flag == "mi_harm":
            scale = [0] * 7;
            scale[0] = fund_key;    #c
            scale[1] = fund_key + 2;#d
            scale[2] = fund_key + 3;#es
            scale[3] = fund_key + 5;#f
            scale[4] = fund_key + 7; # g
            scale[5] = fund_key + 8; # as
            scale[6] = fund_key + 11; # h
        else:
            print("make_scale_mid flag nicht definiert: ", flag);
            scale =  None;
        return scale;
    def chord_flag_from_intervalls(self, chordMid):# chord midi in oktave -> "name" Grundton
        #
        intervalle = [-1] * (len(chordMid) -1 );
        for i in range(len(chordMid) -1):
            intervalle[i] = chordMid[i+1] - chordMid[i];
        #A septakkorde
        if len(chordMid) == 4:
            if intervalle == [4,3,3]:# sept akk
                return "7-ma", chordMid[0];
            elif intervalle == [3,3,2]:# quint sext
                return "567-ma",chordMid[3];
            elif intervalle == [3,2,4]:# terzquart
                return "347-ma",chordMid[2];
            elif intervalle == [2,4,3]:# sekund
                return "2ma",chordMid[1];

        print("chord_flag_from_intervalls TODO")

        return None, -1;

    def make_chord_categorical(self,lowest,flag ):# lowest: midinote "unterste note, grundton" ->in catagorical - oktave
        # diese flags == chord_flag_from_intervalls ! 
        if flag == "7-ma":# septakkord über dur!  
            chord = [0] * 4;
            chord[0] = lowest % 12;
            chord[1] = (lowest + 4) %12;
            chord[2] = (lowest + 7) % 12;
            chord[3] = (lowest + 10) % 12;
            return self.midinotes_to_categorial(chord, 12);
        elif flag == "567-ma": # quintsextakkord über Dur
            chord = [0] * 4;
            chord[0] = lowest % 12;
            chord[1] = (lowest + 3) %12;
            chord[2] = (lowest + 6) % 12;
            chord[3] = (lowest + 8) % 12;
            return self.midinotes_to_categorial(chord, 12);
        elif flag == "347-ma": # terzquart über Dur
            chord = [0] * 4;
            chord[0] = lowest % 12;
            chord[1] = (lowest + 3) %12;
            chord[2] = (lowest + 5) % 12;
            chord[3] = (lowest + 9) % 12;
            return self.midinotes_to_categorial(chord, 12);
        elif flag == "2ma": # sekundakkord über Dur
            chord = [0] * 4;
            chord[0] = lowest % 12;
            chord[1] = (lowest + 2) %12;
            chord[2] = (lowest + 6) % 12;
            chord[3] = (lowest + 9) % 12;
            return self.midinotes_to_categorial(chord, 12);
        elif  flag == "v":
            chord = [0] * 4;
            chord[0] = lowest % 12;
            chord[1] = (lowest + 3) %12;
            chord[2] = (lowest + 6) % 12;
            chord[3] = (lowest + 9) % 12;
            return self.midinotes_to_categorial(chord, 12);
        elif  flag == "ma":
            chord = [0] * 3;
            chord[0] = lowest % 12;
            chord[1] = (lowest + 4) %12;
            chord[2] = (lowest + 7) % 12;
            return self.midinotes_to_categorial(chord, 12);
        elif  flag == "mi":
            chord = [0] * 3;
            chord[0] = lowest % 12;
            chord[1] = (lowest + 3) %12;
            chord[2] = (lowest + 7) % 12;
            return self.midinotes_to_categorial(chord, 12);
        elif flag == "ma46":# quartsextakk
            chord = [0] * 3;
            chord[0] = lowest % 12;
            chord[1] = (lowest + 5) %12;
            chord[2] = (lowest + 9) % 12;
            return self.midinotes_to_categorial(chord, 12);
        else:
            return None; ### noch nix gemacht

    def make_chord_categorical_list(self, lowest, liFlags): #liFlags= li String
        liChords =[];
        for flag in liFlags:
            liChords.append(self.make_chord_categorical(lowest,flag));
        return liChords;
######################################################################################################################
class modulationsInterpreter():
    def __init__(self):
        self.TriadFinder = triadfinder();
        self.MeUtils = musicEvalUtils();
    #------------------------------------------------------------------------------------
    def _get_modulations(self, modakk, zielakkord, zielbass, chords_categorical,basses):# modakk = Dominante zb.
        #findet modulationen in given array of chords_categorical_octave
        #bassarray mus dazu passen
        #A finde modulationsakkord(e)
        loss = self.TriadFinder._list_match_triad_trivial_loss(chords_categorical, modakk );
        treshold =  self.TriadFinder.const_min_treshold_trivial_loss(12); #loss[np.argmin(loss)] ?
        indizi_d7 = myListUtils.np_leq_indizi(loss,  treshold); #8.333334 ist zb A-dur ohne 7
        # A7 A7 A7 A7  der nächste muss "d-moll" ODER "D-dur" sein
        mod = [];
        indizi_ziel = [];
        if len(indizi_d7) > 0:
           
            d7_chains = myListUtils.List_to_succeder_lists(indizi_d7);#[4, 5, 6, 7, 12, 13, 14, 15]
            #B finde zielakkord(e)
            for chain in d7_chains:
                indizi_ziel = self.find_zielakkord( chords_categorical, basses, zielbass, chain, zielakkord);# zielbass bei 46
                #wenn es keine zielakkorde gibt, gilt die Mod nicht..
                if len(indizi_ziel) > 0:
                    #der 1. zielakkord muss nachfolger des letzten d7 sein
                    if (indizi_ziel[0] - 1 == chain[-1]):
                        mod.append(chain);
                        mod[-1].append(indizi_ziel[0]);
        return mod;

    
    def _modulations_data_check_harmony_zielakkord_song(self,mods,triadinf_song):
        li =[];
        for i in range(len(mods)):
                mod = mods[i];
                index_ziel = mod[-1];# last
                TI = triadinf_song[index_ziel];
                if TI is None:
                    li.append(index_ziel);
                else:
                    if len(TI) == 0:
                        li.append(index_ziel);
                    else:
                        TI = TI[0];# die erste
                        ch, b, name = TI;
                        if (len(ch) == 0) or (b == -1) or (name.split("\n")[0] == "-"):
                            print(TI);
                            li.append(index_ziel);
        return li;

    def modulations_data_check_harmony(self, modulationsdata, Triadinfos):
        #check ob indizi der mod data- insbes. der Zielakkord mit Triadinf besetzt ist... -> return List mit Fehlenden
        assert( len(modulationsdata) == len(Triadinfos));
        li = [];
        for j in range(len(modulationsdata)):
            triadinf_song = Triadinfos[j];
            moddata_song = modulationsdata[j];
            mods_hin =  ModulationsData(moddata_song).get_hinmodulationen();
            li.append(self._modulations_data_check_harmony_zielakkord_song(mods_hin,triadinf_song));
            mods_rueck = ModulationsData(moddata_song).get_rueckmodulationen();
            li.append(self._modulations_data_check_harmony_zielakkord_song(mods_rueck,triadinf_song));
        return li;
    def modulations_data_check_format(self,modulationsdata ):
        # TESTFUNKTION
        # Listentiefe [ [ [] ],  [ [] ] ]
        #  regel für mod data
        #  interpoliert "auf Hin folgt rück"
        #  Hin          Hin
        #       Rueck         Rueck
        
        mh :list  = ModulationsData(modulationsdata).get_hinmodulationen();#[0];# list of lists müssen nicht gleich lang sein! 
        mr :list = ModulationsData(modulationsdata).get_rueckmodulationen();
        l_h = len(mh);
        l_r = len(mr);
        if (l_h == 0) and (l_r > 0): # keine Hin ABER ruek
            return False;

        if(  l_h > 0 and type(mh[0]) != list):
            return False;
        if(  l_r > 0 and type(mr[0]) != list):
            return False;
        # längen dürfen sich nur um 1 unterscheiden  auf hin folgt rueck ?
        # möglich: h - r = 1   "h...r...h "   oder h-r = 0   "h...r "  
        if abs(l_r - l_h) > 1:
            return False;

        m_zip = [None] * (l_h + l_r);

        #  muss "gezippt" aufsteigende (>= ?) Folge ergeben  1 2 3 4 8 9 10   10 11 12
        # 0  2  4 ... even = hin
        #   1  3   ... odd = rueck
        i_z = 0;
        for i in range(l_h):
            m_zip[i_z] = mh[i];
            i_z += 2;

        i_z = 1;
        for i in range(l_r):
            m_zip[i_z] = mr[i];
            i_z += 2;

        #flatten
        m_zip = [e for sublist in m_zip for e in sublist ];
        # raufzählen...
        for i in range (len(m_zip) - 1):
            if m_zip[i + 1] - m_zip[i ] < 0:
                return False;
        return True;

    def korrigiere_modulationsdata(self,modulationsdata):
        #A es gibt keine hinmod => alle rueckmods ungültig. (Wurden hinmods ev nicht erkannt) (Arpeggio ?)
        if len(modulationsdata[0]) == 0:
            return [[],[]];
        
        new_hinmods =[];
        new_rueckmods = [];
        hinmods = ModulationsData(modulationsdata).get_hinmodulationen();
        rueckmods = ModulationsData(modulationsdata).get_rueckmodulationen()
        # Folge hin... rueck, ... hin
        # Neubau: solange hinmodulation existiert, nächste rüeckmod bestätigen
        #B
        #wenn eine Hinmod existiert, werden alle weiteren Hinmods bis zur nächsten Rueckmod gekillt.
        # besser direkte intepolation ? siehe check.
        ih = 0;
        ir = 0;
        i_beat_hin = 0;
        i_beat_hin , ih =  ModulationsData(None).get_from_beat(hinmods,ih, i_beat_hin);#
        
        while i_beat_hin != -1: # es existiert eine hinmod
            # neubau :
            new_hinmods.append(hinmods[ih]);
            #suche nächste rueckmod >  i_beat_hin
            i_beat_rueck, ir = ModulationsData(None).get_from_beat(rueckmods,ir,i_beat_hin);# rueckmod_index
            
            if i_beat_rueck == -1: # gibts nicht => fertig
                break;
            else:
                # neubau
                new_rueckmods.append(rueckmods[ir]);

            #iter : next hinmod > ibeat rueck
            i_beat_hin , ih = ModulationsData(None).get_from_beat(hinmods,ih,i_beat_rueck); #
        return [new_hinmods,new_rueckmods ];

    def run_song_on_categorical_chords(self, chords_categorical,basses):# chords_categorical in einer Oktave
        modulationsdata = [None, None];#[0] indizis auf denen die Hinmodulation stattfindet (d7 gefolgt von Sp) [1] = rüeckmod DDv gefolgt von T(4/6)
        #-------------------------------------------
        fund_key = basses[0]; # das ist so gemacht 1. Bassnote ist grundtonart
        fund_key_D = (fund_key + 7) % 12;# dominante
        fund_key_S = (fund_key + 5) % 12;# Subdominante
        fund_key_sp = (fund_key + 2) % 12;# subdominant-paralelle
        fund_key_sp_d = (fund_key_sp + 7) % 12;# die dominante dazu
        fund_key_v = (fund_key + 1) % 12;# alternativ vermindert über Tonika-bass-alt#
        # Modulationsakkord - zielakkord
        
        # hinmodulation: finde D7 ODER v-über-Bass
        d7 = self.TriadFinder.make_chord_categorical(fund_key_sp_d, "7-ma");
        v =  self.TriadFinder.make_chord_categorical(fund_key_v, "v");
        SDp = self.TriadFinder.make_chord_categorical(fund_key_sp,"mi");# subdominantparalellen moll oder Dur !
        SDp_ma = self.TriadFinder.make_chord_categorical(fund_key_sp,"ma");
        # händisch 4 Möglickeiten
        modulationsdata[0] = self._get_modulations( d7,SDp, fund_key_sp,chords_categorical ,basses);
        hinmod_alternate = self._get_modulations( d7,SDp_ma, fund_key_sp, chords_categorical ,basses);
        for i in range(len(hinmod_alternate)):
            if not hinmod_alternate[i] in modulationsdata[0]:
                modulationsdata[0].append(hinmod_alternate[i]);

        hinmod_alternate = self._get_modulations( v,SDp, fund_key_sp,chords_categorical ,basses);
        for i in range(len(hinmod_alternate)):
            if not hinmod_alternate[i] in modulationsdata[0]:
                modulationsdata[0].append(hinmod_alternate[i]);

        hinmod_alternate = self._get_modulations(v, SDp_ma, fund_key_sp, chords_categorical ,basses);
        for i in range(len(hinmod_alternate)):
            if not hinmod_alternate[i] in modulationsdata[0]:
                modulationsdata[0].append(hinmod_alternate[i]);

        modulationsdata[0].sort(key = lambda e : e[0]);
        
        # rueckmodulation: 
        # A: Finde 'DDv mit nachfolgendem 46'
        tritone = (fund_key + 6) % 12;# 
        DDv = self.TriadFinder.make_chord_categorical(tritone, "v"); # in kleinster oktave
        ch_fund_46 = self.TriadFinder.make_chord_categorical(fund_key_D,"ma46");# quartSext über Dominantbass in A   E,- E A C#
        modulationsdata[1] = self._get_modulations(DDv,ch_fund_46, fund_key_D, chords_categorical ,basses);# modAkk, ZielAkk, zielbass...
        #  B: falls A schief ging..
        #  finde chromatische Bass sequenz und (interpretiere Akkorde = side effect ?)
        if len(modulationsdata[1]) == 0:
            #chromatik im Bass  SD DD D46 = 2,3 ,4
            bassline = [fund_key_S,tritone, fund_key_D];
            line_indexe = self.MeUtils.find_bass_line(basses,bassline );
            for i in(line_indexe):
                #modulation bei Basston 2 (lt. Def) "v v v v 46 46 46 46"
                i_v = self.MeUtils.next_valid_bass_skip_wh(basses, i, basses[i]);
                i_46 = self.MeUtils.next_valid_bass_skip_wh(basses,i_v, basses[i_v]);
                #print(basses[i],  basses[i_v],basses[i_46]);
                mod_range = np.arange(i_v,i_46 + 1).tolist();
                modulationsdata[1].append(mod_range);

        modulationsdata[1].sort(key = lambda e : e[0]);
        #------------------------------------------------------------------------------------------------------------------------
        #interpolate data
        modulationsdata = self.korrigiere_modulationsdata(modulationsdata);#[[40, 41, 42, 43, 44], [59, 60]], [[]]
        #if self.modulations_data_check_format(modulationsdata) == False:# TESTFUNKTION...
        #    print(modulationsdata);
        #    raise ValueError("modulationsdata format falsch");
        return modulationsdata;

    
    def find_zielakkord(self,chords_categorical,basses,fund_key, indizi_Dominante, zielakk):
        # bass testen
        indizi_ziel =[];
        if len(indizi_Dominante) == 0:
            return [];
        
        min_loss = self.TriadFinder.const_min_treshold_trivial_loss(12);

        for i in range(len(indizi_Dominante) - 1):
            #wenn indizis nicht fortlaufend oder der letzte, mach das für 1 Akkord...
            i0 = indizi_Dominante[i];
            i1 = indizi_Dominante[i+1];
            if i1 - i0 > 1:
                i_seek = i0 + 1;
                if (basses[i_seek] % 12) == (fund_key %12):
                    ch_seek =  chords_categorical[i_seek];
                    loss =  self.TriadFinder._chord_trivial_loss(ch_seek,zielakk);
                    if loss <= min_loss:
                        indizi_ziel.append(i_seek);
        #für den letzten (oder es war der einzige)
        i = indizi_Dominante[len(indizi_Dominante) -1];
        i_seek = i + 1;
        if(i_seek >= len(basses)):
            return [];
        if (basses[i_seek] % 12) == (fund_key %12):
            ch_seek =  chords_categorical[i_seek];
            loss =  self.TriadFinder._chord_trivial_loss(ch_seek,zielakk);
            if loss <= min_loss:
                    indizi_ziel.append(i_seek);
        return indizi_ziel;
        
######################################################################################################################
class melodieInterpreter():
    def __init__(self):
        self.TriadFinder = triadfinder();
        self.musicEvalUtil = musicEvalUtils();
    def run(self, all_songs , all_basses = None):# raw format
        #A was ist eine 'einstimmige- melodie' passage ?
        all_passagen = self.find_non_homophon_passages( all_songs );#für jeden song diese passagen [anfangsindex, endindex exclusive]
        assert(len(all_passagen) == len(all_songs));
        # prinzip: ungültige wegfiltern...
        # diese passagen müssen eine gewisse Notendichte aufweisen - sonst gilts nicht
        all_passagen = self.filter_passagen_rhytmische_dichte(all_passagen,all_songs,0.5);
        #sind die melodietöne haronmisch sinnvoll = teil eines arpeggios?
        triadinfos = self.interpret_harmonik(all_passagen, all_songs, all_basses) ;
        
        return triadinfos,  all_passagen;

    def _eval_arp_next_solo_bass(self, ab_data, iStart):
        for i in range(iStart,len(ab_data)):
            akkord, bass = ab_data[i];
            if (np.sum(akkord) < 2) and (bass != -1):
                return i;
        return -1;
    def __next_valid_bass(self, basses_song, iStart):# bassarray -> != -1
        return self.musicEvalUtil.next_valid_bass(basses_song, iStart);
        

    def _eval_arp_collect_chords(self,chords, iStart, iEnd, size_chord = 12):      #categorical iEnd EXclusiv, size 
        common_cord = [0] * size_chord;
        for i in range(iStart, iEnd):
            ch = chords[i];
            for j in range(size_chord):
                common_cord[j] = common_cord[j] | ch[j]; #oder operation
        return common_cord;
    def _index_after_last_valid_chord(self, chords):# array chords in catagorical ->int
        # gibt den index HINTER dem letzten akk != 0 zurück falls alle valid, len
        for i in range( len(chords) - 1, -1,-1):
            ch = chords[i];
            if np.sum(ch) == 0:
                return i;
        return len(chords);
    def _eval_arp_segment(self,basses, i_start ):
        if i_start >= len(basses):
            return None;
        #A teste ob der bass weiterläuft.... ende ist dann der erste unterschiedliche
        cur_bass = basses[i_start];
        i = i_start + 1;
        while (i < len(basses)) and (basses[i] == cur_bass):
            i = i + 1;
            
        i_end = i;
        if i_end - i_start > 3:
            return [i_start, i_end];#mutable
        #sonst hier weiter...
        i_end = self.__next_valid_bass(basses,i_start + 1);# darf sich nicht selber finden
        if i_end == -1:
            i_end = len(basses);
        return [i_start, i_end];#mutable
    
    def ___eval_arpeggien(self,i_passage_start, i_passage_end, song_octave, basses):
        triadinfo_song = [None] * len(song_octave);
        ##ab hier sub mit chords in Octave und bass
        b_passage_fertig = False;
        # im Abschnitt bis nächsten bass chords zusammenziehen
        i_start = i_passage_start;
        segment = self._eval_arp_segment(basses,i_start);# bis nächster gültiger bass
        while not segment is None:
            _, i_end = segment;
            # gemeinsamer Akkord über diese Strecke
            common_chord = self._eval_arp_collect_chords(song_octave,i_start, i_end  ) ;
            #generiere vergleichsakkorde über dem bass
            
            bass = basses[i_start];
            # diese 2 zeile wh...
            compare_chords = self.TriadFinder.make_chord_categorical_list(bass,self.TriadFinder.a_chordnames);
            TriadInfos ,loss = self.TriadFinder.calc_possible_chords(common_chord, compare_chords,  bass, self.TriadFinder.a_chordnames); # wie wahrscheinlich ist einer davon ?

            # results 'broadcasten' (das gilt dann für das gesamte segment)
            for i in range(i_start,i_end):
                triadinfo_song[i] = TriadInfos;
                
            #nächstes Segment------------------------------------
            i_start = i_end;
            if(i_start >= i_passage_end):
                break;
            segment = self._eval_arp_segment(basses,i_start);
            if segment is None:
                break;
            if(segment[1] > i_passage_end):
               segment[1] = i_passage_end
            #segmentlänge == 0 ? break
            if(segment[1] - segment[0] == 0):
                break;
        #------wend -------------------------------------------------------
            
        return triadinfo_song;

    def _eval_arpeggien(self,i_passage_start, i_passage_end,  song):# -> arrays (komplette Länge song) mit TriadInf, -ungültige = None
        #---------------------------------------------------------------------------------------------------------------
        # methode für "1stimmige" arpeggien
        # Abschnittsweise. untersuchung nach harmonischem Sinn
        # span von Bass to Bass - dort pitches zu 'sinnvollem Akkord im vgl. zur bassnote' zusammenziehen - akkord loss
        # 1. Finde 'alleinstehenden bass' (+ Akk < 3)

        ab_data = self.musicEvalUtil.extract_akkordBass_data([song])[0]; # chords categorical in Oktave
        chords_song, basses_song = AkkordBassData(ab_data).split();
        return self.___eval_arpeggien(i_passage_start, i_passage_end, chords_song, basses_song);
        
    def interpret_harmonik_melodiepassagen(self, passagen, song, bass = None):
        
        triadinfo_song = [None] * len(song);
        #für alle passagen
        for passage in passagen:
            if len(passage) == 0:
                continue;
            # falls song schon octavgeclampt ist, dort keine gültigen Bässe mehr. In diesem fall übergebenen bass nehmen...
            if len(song[0]) == 12:
                assert(not bass is None);
                triadinfos_arp = self.___eval_arpeggien(passage[0],passage[1], song, bass);
            else:
                triadinfos_arp = self._eval_arpeggien(passage[0],passage[1], song);
           
            #triadinfos_arp = array über ganze Länge, 
            for i in range(len(triadinfos_arp)):
                if not triadinfos_arp[i] is None:
                    triadinfo_song[i] = triadinfos_arp[i]
              
        return triadinfo_song;

    def interpret_harmonik(self,all_passagen,all_songs, all_basses = None):
        assert(len(all_passagen) == len(all_songs));
        triadinfo_all = [];
        
        for i in range(len(all_passagen)):# pro song
            basses = None;
            if not all_basses is None:
                basses = all_basses[i];
            triadinfo = self.interpret_harmonik_melodiepassagen(all_passagen[i], all_songs[i],basses);
            
            triadinfo_all.append(triadinfo);
            
        return triadinfo_all;

    def chordcount(self,song, i_start, i_end):
        c = 0;
        for i in range(i_start,i_end):
            if np.sum(song[i]) > 0:
                c += 1;
        return c;
    def filter_passagen_rhytmische_dichte_song(self,passagen,song,treshold = 0.5):
        filtered_passagen = [];
        # n - Noten /länge der passage
        for passage in passagen:
            i_start, i_End = passage;
            n = self.chordcount(song,i_start, i_End);
            q = n / (i_End - i_start);
            if q > treshold:
                filtered_passagen.append(passage);
        return filtered_passagen;

    def filter_passagen_rhytmische_dichte(self,all_passagen,all_songs,treshold = 0.5):
        assert(len(all_passagen) == len(all_songs));
        all_passagen_filtered = [];
        for i in range(len(all_passagen)):
            all_passagen_filtered.append(self.filter_passagen_rhytmische_dichte_song(all_passagen[i], all_songs[i],treshold) ) ;
        return all_passagen_filtered;

    def find_non_homophon_passages(self,all_songs):
        allpassagen = [];
        for song in all_songs:
            passagen_song = self.find_non_homophon_passages_song(song);#[all_triads, all_bass, all_losses]; __interpret_einstimmige_passagen_find_song
            passagen_song = self.passagen_trim_left(passagen_song,song);
            passagen_song = self.passagen_filter_length(passagen_song,3);                 
            allpassagen.append(passagen_song);# Akk_song, bass_song, loss_song
        return allpassagen;

    def passagen_filter_length(self,passagen,min_lenght = 3):
        filtert_passagen = [];
        for passage in passagen:
            i_start, i_end = passage;
            if (i_end - i_start) >= min_lenght:
                filtert_passagen.append(passage);
        return filtert_passagen;

    def passagen_trim_left(self, passagen, song):
        # linksbündig. pausen links weg
        if(len(passagen) == 0):
            return passagen;
        for i in range(len(passagen)):
            
            i_start, i_end = passagen[i];
            # 1. nichtleerer akkord
            j =  i_start;
            while j < i_end:
                if np.sum(song[j]) > 0:
                    break;
                j += 1;
            passagen[i][0] = j;
        return passagen;
    def voices(self,song):#-> array mit stimmenanzahl pro Akkord
        vcs = np.asarray([0] * len(song));
        for i in range(len(song)):
            vcs[i] = np.sum(song[i]);
        return vcs;
    def find_non_homophon_passages_song(self, song, treshold = 3):
        passagen = [];
        # chordinfo, basses, losses = songdata;
        # range iStart, iEnd exclusive - wird ev len(song)
        istart_written = False;
        for i in range(len(song)):
            chord = song[i];
            if (istart_written == False) and (np.sum(chord) < treshold):# aber > 0 ???
                passagen.append([i,-1]);# neue passage
                istart_written = True;
            
            elif (istart_written == True) and (np.sum(chord) >= treshold):#
                 passagen[ len(passagen) -1 ] [1] = i;
                 istart_written = False;
        # ende ?
        if istart_written == True:# fehlt noch was ... "Klammer zu"...
            passagen[ len(passagen) -1][1] = len(song);

        
        return passagen;
        
######################################################################################################################
class evaluations_data(): # pro song!
    def __init__(self):
        self.TriadInfos = None;
        self.melodie_passagen = None;
        self.modulations_data = None;
        self.basses = None;
        self.voices = None;
        self.tonart_coverage = None;
        self.akkordvarianz = 0;
        self.rh_bassvarianz = 0;
        self.konvergenz = 0;
        self.title = "";
        #records
        self.average_anteil_mel_passagen = 0;
        self.akkord_hits = 0;# treffer im verhältnis zur länge
        self.tonart_hits = 0;# grundton definiert .. 
        self.modulation_hits = 0;
        #
        self.overall_loss = 0;

    def write_makeLine(self,sep):# -> tabbed string
        #title| mel_passagen | bassvarianz ... | loss<- header
        
        return (self.title + sep + str(self.average_anteil_mel_passagen) + sep + str(self.akkord_hits) + sep + str(self.tonart_coverage) + sep + str(self.modulation_hits) 
                + sep +  str(self.rh_bassvarianz) + sep + str(self.akkordvarianz) + sep + str(self.overall_loss) );
class tonartInterpreter():
    def __init__(self):
        self.dumm = 0;
    def _mk_abschnitte(self,basses,modulationsdata):#-> #list of [istart, iEnd, I ]  I:= 1.Stufe = Anfangston-bass
        abschnitte  = []; 
        length_song = len(basses);# stimmt immer
        md =  ModulationsData(modulationsdata);
        
        a = 0;
        hinmod = md.nextHinmod();
        while not hinmod is None:
            abschnitte.append([a,hinmod[0 ], basses[a]]);
            a = hinmod[-1]; # len(hinmod)-1 : letzter ist Zielakkord
            rueckmod = md.nextRueckmod();       
            if not rueckmod is None:
                abschnitte.append([a,rueckmod[0], basses[a] + 5]);# quarte rauf = grundton  (das ist quartsektvorhalt auf dominante)
                a = rueckmod[-1];
            hinmod = md.nextHinmod();
        #wend hinmod
        #letzer Abschnitt
        if(a < length_song):
            abschnitte.append([a,length_song, basses[a]]);
        return abschnitte;
    def _kadenz_from_bassnote(self, b):# -> List I IV IV V  octaveclamped
        # b kommt als midi
        kadenz =[0] * 4; # I IV IV V
        kadenz[0] = b % 12;
        kadenz[1] = (b + 9) % 12;# gosse sexte
        kadenz[2] = (b + 5) % 12;# quarte
        kadenz[3] = (b + 7) % 12;# quinte
        return kadenz;
    def coverage_in_abschnitt(self,abschnitt,basses):
        
        # alle bässe dieses Abschnitts sollten in der "kadenz" liegen...
        coverage = 0.0;
        iStart , iEnd, midi_bass_I =  abschnitt; # unpack
        
        kadenz = self._kadenz_from_bassnote(midi_bass_I);
        
        defined = 0;
        for i in range(iStart, iEnd):
            if basses[i] == -1:
                continue;
            b = basses[i] % 12;
            if b in kadenz:
                defined += 1;
        coverage = float(defined ) / float(iEnd - iStart); # definierte im Verhältnis zur Länge dieses Abschnits

        return coverage;
    def interpret_tonart_song(self, triadinfos, basses, modulationsdata):# triadinfo nicht unbedingt notwendig
        # berechnet tonart coverage  definiert / length  in abschnitten
        abschnitte = self._mk_abschnitte(basses,modulationsdata);#-> #list of [istart, iEnd, I ]  I:= 1.Stufe = Anfangston-bass
        coverage = 0.0;
        for a in abschnitte:
            coverage += self.coverage_in_abschnitt(a,basses);

        return coverage / len(abschnitte);
#end class tonartinterpreter
###################################################################
class music_eval():
    def __init__(self,filepathname_histo_grounddata):
        self.max_song_len = 0;# fürs plotten ,changed in sample
        self.all_eval_data = []; # wird Liste der Klasse evaluations_data !
        self.all_eval_ground_data = []; # genauso
        self.ground_bassvarianz = -1;
        self.ground_akkordvarianz = -1;
        self.ground_average_anteil_mel = -1;
        self.ground_Tonart_coverage = -1;
        self.histo_grounddata = None;
        if len(filepathname_histo_grounddata) > 0:
            with open(filepathname_histo_grounddata + '.pkl', 'rb') as file:
                 self.histo_grounddata : data_utils.histo = pickle.load(file) ;
                         
        self.samples = [];# list of songs
        self.songtitles =[];
        self.len_vorgabe = 0;# ini bei eval
        self.TriadFinder = triadfinder();
        self.modulationsInterpreter = modulationsInterpreter();
        self.melodieInterpreter = melodieInterpreter();
        self.musicEvalUtil = musicEvalUtils();
        self.print_loss_details = True;

    def _find_Triad_over_bass(self, chord_octave, bass):#clamped chord-categorical, bass midi
        assert(bass != -1);
        # findet das in einem categorical chord
        #-> (dreiklang: [midinoten octaveclamped , unterster (ist grundton)], Bezeichner:str)
        chord_midi = self.musicEvalUtil.categorical_to_midnotes(chord_octave);
        terz_gr = -1;
        terz_kl = -1;
        sept_kl = -1;
        b = bass % 12;
        sKey = self.TriadFinder.midnote_to_text(b);# spelling ? Akkord- Quintenzirlkei ?
        results = [];
        #A) Akkord in Grundstellung möglicherweise mehrere Ergebnisse...
        #suche Quinte relativ zum Bass...
        quinte = self.TriadFinder._find_interval_or_complement(chord_midi,b, 7);
        if quinte != -1:
            #suche 3+ wg. Dur
            terz_gr = self.TriadFinder._find_interval_or_complement(chord_midi,b, 4);
            #suche 3- wg. moll
            terz_kl = self.TriadFinder._find_interval_or_complement(chord_midi,b, 3);
            #suche 7- wg. D7
            sept_kl = self.TriadFinder._find_interval_or_complement(chord_midi,b, 10);
            if terz_gr != -1:
                if sept_kl != -1:# zuerst den komlizierteren
                    sSex = "\n7-ma";
                    results.append( self.TriadFinder.triadinfo( [b, terz_gr, quinte, sept_kl], bass, sKey + sSex) );# 
                else:
                    sSex = "\nma";
                    results.append( self.TriadFinder.triadinfo( [b, terz_gr, quinte], bass, sKey + sSex) );# 

            if terz_kl != -1:
                sSex = "\nmi";
                results.append( self.TriadFinder.triadinfo([b, terz_kl, quinte], bass, sKey + sSex) );
        
                 
           

        #B
        #verminderter akkord mindestens 2 von 3 treffern nötig 3 | 6| 9
        terz_kl = -1;
        trit = -1;
        sext_gr = -1;
        terz_kl = self.TriadFinder._find_interval_or_complement(chord_midi,b, 3);# (WH ?)
        trit = self.TriadFinder._find_interval_or_complement(chord_midi,b, 6);
        sext_gr = self.TriadFinder._find_interval_or_complement(chord_midi,b, 9);
        # 2von3 -1 ?
        li = myListUtils.np_leq_indizi([terz_kl,trit,sext_gr],0);
        if len(li) < 2:
            v = [b];
            if terz_kl != -1:
                v.append(terz_kl);
            if trit != -1:
                v.append(trit);
            if sext_gr != -1:
                v.append(sext_gr);
            
            results.append( self.TriadFinder.triadinfo(v, bass, "?" + "\nv") ); # ist eig nicht Key
        #C wahrscheinlichkeitsmethode
        if len(results) == 0:
            compare_chords = self.TriadFinder.make_chord_categorical_list(b,self.TriadFinder.a_chordnames);
            TriadInfos ,loss = self.TriadFinder.calc_possible_chords(chord_octave, compare_chords,  b, self.TriadFinder.a_chordnames); # wie wahrscheinlich ist einer davon ?
            results.extend(TriadInfos);
        #D methode umkehrungen table
        if len(results) == 0:
            if (b in chord_midi):# bass im Akkord enthalten
                chordflag , GT =  self.TriadFinder.chord_flag_from_intervalls(chord_midi);
                if(chordflag != None):
                    sKey = self.TriadFinder.midnote_to_text(GT);
                    results.append( self.TriadFinder.triadinfo(chord_midi, bass, sKey  + "\n" + chordflag) );
        return results;
        
    def clamp_chord_to_octave_mid(self, chord):
        for i in range(len(chord)):
            chord[i] = chord[i] % 12;
        return chord;

    
    def clamp_song_to_octave_categorical(self, song):#np -> List
        song_octave = [];
        for i in range( len(song)):
            #print(song[i].shape) 1,128
            ch = self.musicEvalUtil.clamp_chord_to_octave_categorical(np.squeeze(song[i]));
            song_octave.append(ch.tolist())
        return song_octave;
        
    def sample(self, song, title):# sammelt die songs eines gesamtdurch laufs kriegt np
        # callby "Aussen" ist Schnittstelle
        self.samples.append(song);
        self.songtitles.append(title.split(".")[0]);
        if len(song) > self.max_song_len:
            self.max_song_len = len(song);
        
    #-----------------------------------------
    def _interpret_chords_song(self,chords_song, basses_song):#song, bässe ->List of TriadInfos
        
        assert(len(chords_song) == len(basses_song));

        triadinfo_song = [None] * len(chords_song);# Liste-mit Listen: indizes müssen erhalten bleiben wegen darstellung
        
        
        for i in range(0, len(chords_song)):# self.len_vorgabe
            chord = chords_song[i];# schon geclamped
            triad_info = [];
            if np.sum(chord) == 0:
                #print(i, " _interpret_chords_song : chord empty")
                continue;

            
            if (np.sum(chord) > 2) and (basses_song[i] != -1):
                
                triad_info = self._find_Triad_over_bass(chord, basses_song[i]);#-> List[dreiklang-midi, name:string]
                if len(triad_info) > 0:
                    loss_tmp = np.asarray([0] * len(triad_info),dtype = np.float);
                    for j in range(len(triad_info)):
                       ch, _ , s   = triad_info[j];
                       #print(s);
                       chord_tar =  self.TriadFinder.midinotes_to_categorial(ch,12);
                       chord_pre = self.musicEvalUtil.clamp_chord_to_octave_categorical(chord);
                       loss_tmp[j] =  self.TriadFinder._chord_trivial_loss(chord_pre,chord_tar);
                    #efo j ------------------------------------------------------------------ 
                    l_min = np.argmin(loss_tmp);
                    #print(l_tmp);
                    a = myListUtils.np_leq_indizi(loss_tmp , loss_tmp[l_min]);
                    # alle Akkorde die da nicht drin sind werden aus traidinfo gelöscht. e.g leere Liste!
                    for j in range(len(triad_info)):
                        if not j in a:
                            triad_info[j] = None;
                    triad_info = myListUtils.NoneList_trim(triad_info);
                    # alle ohne Bass (? wieso haben die keinen bass ?) raus
                    for j in range(len(triad_info)):
                        _ , b , _   = triad_info[j];
                        if b == -1:
                           triad_info[j] = None;

                    triad_info = myListUtils.NoneList_trim(triad_info);
                    #if len(triad_info) > 1:
                    #       print("_interpret_predicted_chords_song - akk nicht eindeutig. " + str(len(triad_info) ) + "möglichkeiten " + str(triad_info));
                           #[([2, 5, 9], 50, 'D\nmi'), ([2, 5, 11], -1, '?\nv')]
            triadinfo_song[i] = triad_info;
            
        #----------------------------------------------------------
        return triadinfo_song;
    #-------------------------------------------------------

    def __merge_triadinf_lists(self,list_target,list_to_merge):# pro song ....
        # beschreibt nur unbesetzte Targets
        assert(len(list_target) == len(list_to_merge));
        for i in range(len(list_target)):
            if (list_target[i] is None) or (len(list_target[i]) == 0):
               if list_to_merge[i] is not None:
                    list_target[i] = list_to_merge[i];

        return list_target;

    def interpret_melodies(self, songs, basses):#songs  -> neue Songs mit ev. neuen Akkorden, ohne tonartdata
        #
        # stellt weniger-stimmige passagen fest, und versucht diese zusammenfassend (arpeggio angenommen)  in chords umzuwandeln
        # 
        # songs zur oktave 
        for i in range(len(songs)):
             songs[i]  = self.clamp_song_to_octave_categorical(songs[i] );
        
        Triadinfos,  all_passages = self.melodieInterpreter.run( songs, basses);
        
        assert(len(all_passages) == len(songs));
        assert(len(Triadinfos) == len(songs));
        # schreibe neuen songs, in dem die passagen mit dem erkannten Akkord ausgefüllt sind.
        for i  in range(len(songs)):
            #song = songs[i];
            passagen = all_passages[i];
            if( passagen is None):
                continue;
            
            for passage in passagen:
                songs[i], _  = self.notate_triadinfo_basses(songs[i],None, passage,Triadinfos[i] );
                
        return songs, Triadinfos, all_passages;
    
    #---------------------------------------------
    def calc_Akkordvarianz(self,all_songs):
        # forallLOOP
        all_akk_var = [];
        for song in all_songs:
            all_akk_var.append(self.calc_Akkordvarianz_song(song))
        return all_akk_var;

    def calc_Akkordvarianz_song(self,song, histo = None):
        #length = len(song);
        if histo is None:
            histo = data_utils.histo("");
            histo.make(song, False);
            
        histo.calc_Wahrscheilichkeiten_Akk();# wie wahrsch. dass dieser oder jener akkord
        #data_analys.plot_Balkenvert_1darray(histo.Wks,"Wahrscheinlichkeiten akks",'log');
        E = histo.calc_Erwartungswert(); #idr kommt akkord mit diesem index
        var = histo.calc_Varianz( E );# abweichung davon
        return  var / len(song);# noch mal relativ Länge des songs  (ist schon per def varianz  Sum (E-p)^2) / (lenWk -1) relativ zur menge der wks = Anzahl Akkore != länge song
        
    #---------------------------------------------
    def calc_rhythmic_bassvarianz_song(self,song):
        ab_data = self.musicEvalUtil.extract_akkordBass_data([song])[0];# func ist fuer Liste gemacht...
        #zählen. wann (nach wieviel Einheiten) sich die bassnote ändert kein bass (-1) skip
        baesse = AkkordBassData(ab_data).get_Baesse();#._split_ab_data(data);
        length = len(baesse);
        last_bass = -42;
        changes_bass = [];
        #basswechsel periode
        for i in range(len(baesse)):
            b = baesse[i];
            if b == -1: # skip invalid
                continue;
            if b != last_bass:
                last_bass = b;
                changes_bass.append(i);
        
        # dauern davon
        l = (len(changes_bass) - 1);
        if (l < 2):
            return 0;
        dauern = np.array([0] * l, dtype = np.int);
        for i in range(l):
            dauern[i] = changes_bass[i+1] - changes_bass[i];
        summe_dauern = dauern.sum();
        reste = np.array([0] * l, dtype = np.int);
        mode = data_utils._modus(dauern);
        
        for i in range(0,l):
            reste[i] = dauern[i] % mode;
        summe_reste = reste.sum();
        rbvar:float = float(summe_reste) / summe_dauern;
        return rbvar;#  zwangsläufig zwischen 0 und 1 
        
    def calc_rhytmic_bassvarianz(self, all_songs):
        all_b_varianz = [];
        for song in all_songs:
            all_b_varianz.append(self.calc_rhythmic_bassvarianz_song(song))
        return all_b_varianz;
    
    #-------------------------------------------------------------------
    def _to_allchords(self,all_songs):
        all_chords = [];
        for song in all_songs:
            for chord in song:
                all_chords.append(chord);
        return all_chords;
    #-------------------------------------------------------------------

    def interpret_Tonart(self,all_TraidInfo, all_basses, all_modulationsdata):# -> LISTE mit coverage der eizelnen sobgs
        assert(len(all_TraidInfo) == len(all_basses));
        assert(len(all_TraidInfo) == len(all_modulationsdata));
        all_tonart_coverage = [];
        for i in range(len(all_basses)):
            all_tonart_coverage.append( tonartInterpreter().interpret_tonart_song(all_TraidInfo[i],all_basses[i], all_modulationsdata[i] ));

        return all_tonart_coverage;

    def interpret_chords(self, songs, basses = None):# entweder song != oktave oder song-clamped UND bässe
        # beurteilt die plausibilität der Akkorde - jeden einzeln (vertikal), ohne Zusammenhang
        # findet Akkordnamen und Bass
        allTriadInfos = [];
        #i_name = 0;
        for i in range(len(songs)):
            song = songs[i];
            #print("interpret_predicted_chords: ",self.songtitles[i_name]);
            #i_name += 1;
           
            if basses is None:# hier ein nicht geclampter original (size 128 Song)
                ab_data = self.musicEvalUtil.extract_akkordBass_data([song])[0];
                chords_song, basses_song = AkkordBassData( ab_data).split();
            else:
                chords_song = song;
                basses_song = basses[i];

            triadinfo_song = self._interpret_chords_song(chords_song,basses_song);#->triadinfo_song
            allTriadInfos.append(triadinfo_song);
        return allTriadInfos;

    def interpret_modulation(self,songs, basses):
        assert(len(songs) == len(basses));
        mod_data = [];
        for i in range(len(songs)):
            mod_data.append(self.modulationsInterpreter.run_song_on_categorical_chords(songs[i],basses[i]))
        return  mod_data;

    def notate_triadinfo_basses(self, song_octave, basses, _range, TriadInfo_song):# schreibt 1- Akkord + bass aus Triadinfo in octave-song , range[a,b]
        for i in range(_range[0],_range[1] ):
            assert( not TriadInfo_song[i] is None);
            chord_octave, bass, sName = TriadInfo_song[i][0];#immer den ersten. TODO versuche zb. im apeggio finder Mehrdeutigkeit aufzulösen (Kontext)
            ch = self.musicEvalUtil.midinotes_to_categorial(chord_octave,12);
            song_octave[i] = ch;
            if not basses is None:
                basses[i] = bass;
        return song_octave,basses;

    def notate_bass_from_Triadinfo(self,basses, TriadInfo):
        assert(len(basses) == len(TriadInfo));
        for i  in range(len(TriadInfo)):
            TI = TriadInfo[i];
            if TI is None:
                continue;
            if len(TI) == 0:
                continue;
            chord_octave, bass, sName = TI[0];
            if basses[i] == -1:
                basses[i] = bass;

        return basses;

    #"EXTRA KLASSE! TODO----------------------------------------------------
    def __examine_gateregs_conv_mask(self,a, minsize = 5 ):
        l = int(len(a) / 6.5);# macht conv. mask frei Schnauze
        if l < minsize:
            l = minsize;

        m = np.asarray([0.3] * l);
        m[0] = 0.1;
        m[1] = 0.2;
        m[-1] = 0.1;
        m[-2] = 0.2;
        return m;

    def examine_gatregs(self, song, li_gate_regs, name):# zeitl.zusammenhang zwischen gr und A) einst.passagen B) fundKey 
        #gatereg data zum Zeitpunkt bla...
        if li_gate_regs is None:
            return None;
        l_offset = len(song) - len(li_gate_regs);#gate regs ohne anfangssequenz
        gr_sums = np.asarray([0.0] * len(song));
       
        for i in range(len(li_gate_regs) ):
            gr_sums[i + l_offset]= np.sum(li_gate_regs[i]);
        
        if np.sum(gr_sums) == 0:
            return 1;
        
        #print(gr_sums)
        #überlagere die Kurven von Gate reg und Stimmenzahl
        gr_mean = np.mean(gr_sums[l_offset: :]);
        gr_varianz = data_utils.normalize((gr_sums - gr_mean)[l_offset : :]);
        #gr_std = np.std(gr_sums[l_offset: :]);
        voices = self.melodieInterpreter.voices(song[l_offset: :]);

        vcs = data_utils.normalize(voices);

        #m = self.__examine_gateregs_conv_mask(vcs);
        #vcs_c = data_utils.convolution_1d(vcs, m);
        #vcs_c = data_utils.normalize(vcs_c);
        #gr_c = data_utils.convolution_1d(gr_varianz, m);
        #gr_c = data_utils.normalize(gr_c);
        gr_c = data_utils.normalize(gr_varianz);
        #--------------
       
        data_analys.plot_1dimarrays( [vcs, gr_c], name + " correlation gates - voices",["voices","gates"]);

       

        
        return 1;

    #-----------------------------------------------------------------
    def update_evaluations_data(self,evaldata : list , param, data):# for all loop mit payload
        assert( not evaldata is None);
        
        for i in range(len(evaldata)):
                if param == "melodie_passagen":
                    evaldata[i].melodie_passagen = data[i];
                elif param == "TriadInfos":
                    evaldata[i].TriadInfos = data[i];  
                elif param == "modulations_data":
                    evaldata[i].modulations_data = data[i];  #
                elif param == "basses":
                    evaldata[i].basses = data[i]; 
                elif param == "voices":
                    evaldata[i].voices = data[i];
                elif param == "akkordvarianz":
                    evaldata[i].akkordvarianz = data[i];#<- KEIN array... 
                elif param == "rh_bassvarianz":
                    evaldata[i].rh_bassvarianz = data[i]; 
                elif param == "title":
                    evaldata[i].title = data[i]; #ist array!
                elif param == "anteil_mel_passagen":
                    evaldata[i].average_anteil_mel_passagen = data[i]; 
                elif param == "overall_loss":
                    evaldata[i].overall_loss = data[i]; #Karray!
                elif param == "konvergenz":
                    evaldata[i].konvergenz = data[i]; #Karray!
                elif param == "tonart_coverage":
                    evaldata[i].tonart_coverage = data[i];
            
            
        return 1;
    def evaluate_eval_data_prozent_melodie_passagen(self, data :list):# listof evaluations_data# wid in liste eingetragen 
        #berechnet den  Anteil an Melodiepassgen auf die Länge des songs....
        # braucht BÄSSE
        # verändert data
        for ed   in data: #pro song
            len_song = len(ed.basses);
            passagen = ed.melodie_passagen;
            d = 0;
            for p in passagen:
                assert(len(p) == 2);
                if (data ==  self.all_eval_data ) and (p[0] >= self.len_vorgabe):
                    d = d + (p[1] - p[0]);
                else:
                    d = d + (p[1] - p[0]);
            value = d / len_song;
            ed.average_anteil_mel_passagen = value;
        return data;

    
    def ed_write_header(self,sep):#-> str
        return ("title" + sep + "av-anteil-mel-passagen" +  sep + "akkord_hits" + sep + "tonart_cover" + sep +"modulation_hits" + sep + "rh-bassvarianz" + sep + "akkordvarianz" + sep + "overall-loss");
    def ed_print_to_screen(self, data:list):# obsolet
        # header
        print(self.ed_write_header("\t"));#mel_passagen | bassvarianz ... | loss<- header
        # zeilen drunter
        for ed in data:
            print(ed.write_makeLine("\t"));
    #1
    def ed_update_loss_Tonart(self, w = 1.0):# 
        # vieviele chords indexe sind Tonartmässig definiert wert := 0<=  tonart_coverage  <= 1 
        # nur all_eval_data
        average_coverage = 0.0;
        for i in range(len(self.all_eval_data)):
            ed_song = self.all_eval_data[i];
            average_coverage = ed_song.tonart_coverage;
            loss = 1.0 -  average_coverage;# alles gecovered => loss = 0
            if self.print_loss_details == True:
               print("loss key coverage:" , loss);
            self.all_eval_data[i].overall_loss += loss * w;
    #2
    def ed_update_loss_modulation(self,w = 1.0):# for jede Modulation 1 punkt. punktwert relativ zur länge des song ... zb 4/l
        #  Es gibt in grounddata jeweils 1 hin und eine Rückmod pro file
        #  es wären in predict allerdings auch mehrere modulationen ok... ?...
        #  eine mod dauert 4 beats also maximale Anzahl mods length / 4
        # nur all_eval_data
        for i in range(len(self.all_eval_data)):# for all songs
            ed_song = self.all_eval_data[i];
            length = len(ed_song.basses);
            max_mods : float = float(length / 4);
            modhin :float =   float( len(ModulationsData( ed_song.modulations_data).get_hinmodulationen()));
            modrueck :float = float( len( ModulationsData( ed_song.modulations_data).get_rueckmodulationen()));
            mod_hits : int = int(modhin + modrueck);
            loss = 1.0 - (modhin + modrueck) / max_mods;# wenn modhin + modrueck == maxmods; loss = 0
            if self.print_loss_details == True:
                print("loss modulation:" , loss);
            self.all_eval_data[i].modulation_hits = mod_hits;
            self.all_eval_data[i].overall_loss += loss * w;
    #3
    def ed_update_loss_Akkordvarianz(self, w = 1.0):
        # bestraft u.A konvergierendes verhalten (gleichförmigkeit)
        for i in range(len(self.all_eval_data)):
            ed = self.all_eval_data[i];
            assert(self.ground_akkordvarianz != -1);
            loss = abs(self.ground_akkordvarianz - ed.akkordvarianz) * w;
            if self.print_loss_details == True:
                print("loss variance chords:" , loss);
            self.all_eval_data[i].overall_loss += loss # 
        return None;
    #4
    def ed_update_loss_rh_bassvarianz(self,w = 1.0):
        # all_eval_data - groundunddata (abs) = loss
        for i in range(len(self.all_eval_data)):
            ed = self.all_eval_data[i];
            #self.ground_bassvarianz ground bassvarianz immer 0
            loss = abs(ed.rh_bassvarianz - 0);
            if self.print_loss_details == True:
                print("loss bassvarianz:" , loss);
            self.all_eval_data[i].overall_loss += loss * w;# von modell zu modell denken...
        return None;
    #5
    def ed_update_loss_Akkordhits(self, w = 1.0):
        # wieviele Akkorde im verhältniss zur länge sinnvoll ?
        for i in range(len(self.all_eval_data)):
            ed_song = self.all_eval_data[i];
            #print(ed_song.title); #--------------------
            length = len(ed_song.basses);# length = song length
            if (len(ed_song.TriadInfos) != length):
                raise ValueError("ed_updateloss_Akkordhits ERR len TriadInfos !=  len basses");
            hits = 0.0;
            for j in range(length):
                TI = ed_song.TriadInfos[j];
                if TI is None :
                    continue;
                valid_chords = 0;
                for k in range(len(TI)):
                    ch, b, name = TI[k];
                    if (b != -1) and (len(ch) > 0):# nur die mit Bass,(vollständig)
                        valid_chords = valid_chords + 1;
                # --efok---
                if valid_chords > 0:
                    hits = hits + 1 / valid_chords;# mehrere möglichkeiten nicht "volle punktzahl"
            self.all_eval_data[i].akkord_hits = hits;

            assert(float(length) >= hits);
            # alle SOLLTEN erkannt sein. Loss ist das, was übrig bleibt
            loss = 0;
            if (hits < float(length)):
                loss = (length - hits  ) / length; # zahl zwischen 0 und 1
            if self.print_loss_details == True:
                print("loss chordhits:" , loss);
            self.all_eval_data[i].overall_loss += loss * w;
    #6
    def ed_update_loss_melpassagen(self, w = 1.0):
        #overall loss mit melodieanteil modifizieren
        for i in range(len(self.all_eval_data)):
            loss = abs(self.ground_average_anteil_mel - self.all_eval_data[i].average_anteil_mel_passagen) ; #(0 =< zahl =< 1) - (0 =< zahl <= 1 )
            if self.print_loss_details == True:
                print("loss melpassagen:" , loss);
            assert(loss <= 1.0);# 0 <= loss <= 1  ?
            self.all_eval_data[i].overall_loss += loss * w;
        
    def ed_overall_loss(self, nEinzellosses = 6):
        # über ALLE songs
        numberofSongs = len(self.all_eval_data)
        assert(numberofSongs > 0);
        overall_loss = 0.0;
        # all_eval_data -
        for i in range(len(self.all_eval_data)):
            ed = self.all_eval_data[i];
            loss = self.all_eval_data[i].overall_loss;
            overall_loss += loss;

        return overall_loss  / (numberofSongs * nEinzellosses);

    def ed_write_to_csv(self, title, out_path):
        lines = [];
        lines.append(self.ed_write_header("\t") + "\n" );
        for ed in self.all_eval_data:
           lines.append( ed.write_makeLine("\t") + "\n" );

        lines.append(str(self.overall_loss));

        out_file_path = out_path + title + "_evaldata" + ".csv";
        print("write file: ", out_file_path);
        with open(out_file_path,'w') as file:
            file.writelines(lines);
            file.close();

        return None;
    # hier immer zum vergleich die grounddata ! 
    def calc_losses_from_eval_data(self):
        nfuncs = 6;
        # 1 MELODIEANTEILE
        self.ed_update_loss_melpassagen(1.0);
        # 2 AKKORDE erkannt.?
        self.ed_update_loss_Akkordhits(1.0);
        # 3 TONART- Grundtöne ?
        self.ed_update_loss_Tonart(1.0);
        # 4 MODULATION ?
        self.ed_update_loss_modulation(1.0);
        # 5 RH-BASSVARIANZ
        self.ed_update_loss_rh_bassvarianz(1.0);
        # 6
        self.ed_update_loss_Akkordvarianz(1.0);
        
        return nfuncs
        
        
    def Luecken_in_Traidinfos_find(self, Triadinfos):
        li_all = [];
        for TIs_song in Triadinfos:
            
            li = [];
            for i in range(len(TIs_song)):
                TIs = TIs_song[i];
                if (TIs is None) or len(TIs) == 0:
                    li.append(i);
                    
            li_all.append(li);
        return li_all;
    def Luecken_in_Traidinfos__stopf(self,lists_luecken,all_TriadInfos,all_songs, all_basses):
        #irgendwie ein Notfall ?
        for i in range(len(lists_luecken)):
                li = lists_luecken[i];
                if(len(li) == 0):
                    continue;
                else:
                    print("Luecken ", li);

                TIs = all_TriadInfos[i];
                sng = all_songs[i];
                bass = all_basses[i];
                
                for j in range(len(li)):
                    index = li[j];
                    #print(sng[index], bass[index]);
                    ch = self.musicEvalUtil.clamp_chord_to_octave_categorical( myListUtils.np_squeeze_or_not(sng[index])  );
                    if np.sum(ch) == 0:# leer / pause
                        li[j] = None;
                        continue;
                    else:
                        raise ValueError("try_stopfe_luecken_in_triadinfo- new dataflow! " + str(li));
                        TIs = self._find_Triad_over_bass(ch, bass[index]); 
                        print(TIs[index]);
                li = myListUtils.NoneList_trim(li);
                if(len(li)> 0):
                    raise ValueError("immer noch lücken! " + str(li));
        return all_TriadInfos;

    def eval_grounddata(self, datafldr):#"
        # arbeitet auf membervariablen sachen über alle songs zusammengefasst
        print("music eval... ground-data ... das kann dauern ...");
        all_songs_ground  = abba_loader(0).get_all_songs(datafldr);#list of songs
        number_of_songs = len(all_songs_ground);
        # test mit 1nem file
        #all_songs_ground  = [abba_loader(0).load_one_song(datafldr + "0_ModArp_27.pkl")];#list of songs

        # --- main  diese fkt auch für predicted --------
        self._eval(all_songs_ground, None, self.all_eval_ground_data); # <- created speicher  all_eval_ground_data
        # - - - - - - - - - - - - - - - - - - - - - - - - -
        assert(len(self.all_eval_ground_data) == number_of_songs);
        #bassvarianz , akkordvarianz auf grounddata für alle berechnen
        print("varianzen ground...");
        all_chords = self._to_allchords(all_songs_ground); #Trick: mache 1nen song aus allen
        self.ground_bassvarianz = self.calc_rhythmic_bassvarianz_song(all_chords);
        
        self.ground_akkordvarianz = 0;
        for sng in all_songs_ground:
                 self.ground_akkordvarianz += self.calc_Akkordvarianz_song(sng,None);# histo self.histo_grounddata grounddata im Konstruktor nütz nix, da das über ALLE daten ist
        self.ground_akkordvarianz = self.ground_akkordvarianz / number_of_songs;

        # Melodieanteil
        print("average melodieanteil - ground")
        sum = 0.0;
        for ed in self.all_eval_ground_data:
            sum += ed.average_anteil_mel_passagen;
        self.ground_average_anteil_mel = sum / number_of_songs;

        # Mittelwert der Tonart coverage (sollte 1 sein) wert wird eig. nicht gebraucht weil bei loss 0 und 1 absolute grenzen sind
        sum = 0.0;
        for ed in self.all_eval_ground_data:
            sum += ed.tonart_coverage;
        self.ground_Tonart_coverage = sum / number_of_songs;
        print("eval_grounddata: ready");
    #-----------------------------------------------
                
    def _eval(self, all_songs, songtitles, evaldata : list): # für samples UND für grounddata
        #call: grounddata: songtitles = None (als Flag missbraucht)

        assert ( len(all_songs) > 0); #!
        #Ergebnislisten createn...
        for i in range(len(all_songs) ):
            evaldata.append( evaluations_data() ) ;

        if not songtitles is None:# => es sind die samples
            self.update_evaluations_data( evaldata , 'title' ,songtitles);

        all_basses = [];
        all_voices = [];
        ## Akkorde und Melodiepassagen
        _songs = all_songs.copy();

        for i in range(len(_songs)):
            # bässe ....
            _bass =  self.musicEvalUtil.extract_basses(_songs[i]) ;
            all_basses.append(_bass);
            #Stimmenzahl 
            all_voices.append( self.melodieInterpreter.voices(_songs[i])); 
           
        #----------------------------------------------------------------------------------
        __songs_extended , _Triadinfos1, _all_passages = self.interpret_melodies( _songs, all_basses );# ->neue songs mit melodieteilen in chords umgewandelt (Oktavgeclampt)
        
        # bässe updaten - nach Triadinfo - falls melodies bass "weitergezogen" hat. 
        # Es sollten im Idealfall keine Basslöcher mehr drin sein (= -1)
        for i in range(len(_Triadinfos1)):
            all_basses[i] = self.notate_bass_from_Triadinfo( all_basses[i], _Triadinfos1[i]);# -> womöglich bässe verändert !
        
        #'einstimmige passagen' zu eval data. "speichern"
        assert(len(evaldata) == len(_all_passages));
        self.update_evaluations_data(evaldata, 'melodie_passagen' , _all_passages);

        # Akkodinterpretation läuft auf __songs_extended wg. möglicher Zusammenhänge - (vorher , ? , nachher )
        _Triadinfos2 = self.interpret_chords( __songs_extended, all_basses );# all_songs (braucht bässe, dann != clamped to octave) 
        
        all_TriadInfos = _Triadinfos1;
        # Triadinfos joinen.
        assert(len(_Triadinfos1) == len(_Triadinfos2));
        assert(len(_Triadinfos1) == len(   evaldata ));
        all_TriadInfos = [None] * len(_Triadinfos1);
        for i in range(len(_Triadinfos1)):
            all_TriadInfos[i] = self.__merge_triadinf_lists(_Triadinfos1[i],  _Triadinfos2[i]);
            

        # check lücken in traidinfo - (nur für grounddata) . Es sollte keine Lücken geben
        if songtitles is None:# => grounddata
            lists_luecken = self.Luecken_in_Traidinfos_find(all_TriadInfos);
            all_TriadInfos = self.Luecken_in_Traidinfos__stopf(lists_luecken,all_TriadInfos,all_songs, all_basses);# ??? notfallfunktion
            
            
        # und zu evaldata
        self.update_evaluations_data(evaldata, 'TriadInfos' ,all_TriadInfos);
        self.update_evaluations_data(evaldata, 'basses' ,all_basses);
        self.update_evaluations_data(evaldata, 'voices' ,all_voices);
        
        #dies braucht bässe!
        evaldata = self.evaluate_eval_data_prozent_melodie_passagen(evaldata);

        # modulation ..........
        all_modulationsdata = self.interpret_modulation( __songs_extended, all_basses );
        # test - in grounddata muss genau 1 hin und 1 rueck rauskommen
        if songtitles is None:#=> grounddata
            self.TEST_modData(all_modulationsdata);
        #check modulationsdata vs Triad info ?
        #lists_luecken = self.modulationsInterpreter.modulations_data_check_harmony(all_modulationsdata, all_TriadInfos);
        self.update_evaluations_data(evaldata, 'modulations_data' ,all_modulationsdata);
       
        
        #muss nach modulation kommen, weil mod.data benötigt
        all_tonart_coverage = self.interpret_Tonart(all_TriadInfos, all_basses,all_modulationsdata)#
        #für grounddata wird hier gleich der mittelwert berechneht
        
        self.update_evaluations_data(evaldata, 'tonart_coverage' ,all_tonart_coverage);

        # NUR samples  != Grounddata (wird in eval_grounddata gerechnet)
        if not songtitles is None:
            all_bassvarianz = self.calc_rhytmic_bassvarianz(all_songs);
            self.update_evaluations_data( evaldata ,'rh_bassvarianz' , all_bassvarianz);
            all_akkordvarianz = self.calc_Akkordvarianz(all_songs);
            self.update_evaluations_data( evaldata, 'akkordvarianz' , all_akkordvarianz);
            #all_konvergenz = self.calc_Konvergenz(all_songs); # try calc_Konvergenz_lambdatest
            #self.update_evaluations_data(evaldata, 'konvergenz' ,all_konvergenz);
        return 1;
    #---------------------------------------------------------------------------
    def TEST_modData(self,all_modulationsdata):
        for md in all_modulationsdata:
                hinmods = ModulationsData(md).get_hinmodulationen();
                rueckmods = ModulationsData(md).get_rueckmodulationen();
                if (len(hinmods) != 1):
                    print("TEST_modData failed: n-Hinmod ", len(hinmods))
                if (len(rueckmods) != 1):
                    print("TEST_modData failed: n-Rueckmod ", len(rueckmods))
    ####################################################################
    # startfunktion evaluation
    ###############################################################
    def eval(self, title, path_model, len_vorgabe, print_loss_details = True, bCsv = True, bShowPlot = True):# 
        self.len_vorgabe = len_vorgabe;
        self.print_loss_details = print_loss_details;
        
        #------------------------------------------------------
        # 1. grounddata
        #wenn grounddata schon berechnet ist, wird sie geladen
        filepath_gd = abba_settings.g_base_dir_other + "ground_data/grounddata";
        if os.path.isfile(filepath_gd + ".pkl") == True:
            data = data_utils.open_data(filepath_gd + ".pkl");
            self.all_eval_ground_data, self.ground_bassvarianz, self.ground_akkordvarianz , self.ground_average_anteil_mel,self.ground_Tonart_coverage = data;
            print("grounddata from file ", filepath_gd);
        else:
            self.eval_grounddata(abba_settings.g_dir_beatvecs_net_other);
            data_utils.dump_data_list( [self.all_eval_ground_data, self.ground_bassvarianz, self.ground_akkordvarianz, self.ground_average_anteil_mel,self.ground_Tonart_coverage], filepath_gd);
            print("grounddata saved to file ", filepath_gd);
        #-------------------------------------------------------------
        #2.  die predicted songs...
        #print("music eval... prediced");
        self._eval(self.samples, self.songtitles, self.all_eval_data); #  
        
        if print_loss_details == True:
            print(self.songtitles);
        #----Auswertung... Losses berechnen etc.
        nfuncs = self.calc_losses_from_eval_data();
        self.overall_loss = self.ed_overall_loss(nfuncs);
        if self.print_loss_details == True:
            print("overall-loss: " , self.overall_loss);

        
        
        # zu csv file
        if bCsv == True:
            self.ed_write_to_csv(title,path_model);

        if bShowPlot == True:
            self._plot_data(title,path_model,bShowPlot);

        return self.overall_loss;
    #------------------------------------------
    ############################
    # plots
    #############################
    def _plot_data(self,title,path_model,bShow):
        #manager = plt.get_current_fig_manager();
        #fig_size = manager.window.maxsize();
        #manager.resize(*manager.window.maxsize());

        len_data = len(self.all_eval_data);
        if len_data < 2:
            print("abba_music_eval._plot_data : len data < 2. mindestens 2 songs gefordert");
            return 0;

        #xse = längster song
        xse = np.arange(0,self.max_song_len);
        fig, axs = plt.subplots(len_data, 1,sharex = True);
        
        #fig.tight_layout();# abstände zwischen den subplots
        fig.suptitle(title, fontsize = 10);
        fig.subplots_adjust(hspace = 0.4);
        for ax in axs:
            ax.tick_params(axis = 'x', labelsize = 6);
            ax.tick_params(axis = 'y', labelsize = 6);
            ax.set_ylim([0,2]);
            ax.set_xlim([0,xse[- 1]]);
            ax.set_yticks([]);# keine beszeichner an y
           
        
        for i in range(len_data):
            ax = axs[i];
            ax.title.set_text(self.songtitles[i]);
            ax.title.set_size(6);
            songdata : evaluations_data  = self.all_eval_data[i];
            self.__plot_data_modulation(ax,xse,songdata.modulations_data);
            self.__plot_data_chordbass(ax,xse,songdata.TriadInfos , songdata.basses ,  songdata.melodie_passagen);
        if bShow == True:
            #fig.set_size_inches(1924,1061); klappt nicht - saved bild weiss
            #plt.savefig( path_model + "music-eval-" + title + ".pdf",bbox_inches = 'tight' );## zu klein 
            plt.show();
            #plt.close();
        return 1;
    #----end _plot
    def __plot_data_chordbass_get_color(self, index, einst_passagen):
        #entscheidet Farbe für diesen Index (Dreiklang, Bass)
        std_color =(0.1,0.1,0.9); #blue
        passagen_color =(0.6,0.1,0.6); #lila, dunkel
        for passage in einst_passagen:
            if len(passage) == 0:
                continue;
            iStart_passage , iEnd_passage = passage;
            if (index >= iStart_passage) and (index < iEnd_passage):
                return passagen_color;
        return std_color;
    def __plot_data_chordbass(self, ax, xse, triads,  basses ,einst_passagen):# 
        # schreibt Akkord + Bass BLAU, wenn in "einstimmiger Passage" LILA
        # alle listen gleich lang!
        xse = xse[0 : len(triads)];
        for i in range(len(triads)):
            triad_inf = triads[i];
            if triad_inf is None:
                continue;
            if len(triad_inf) == 0:
                bass = basses[i];
                sChord = "-";
            else:
                tri, bass, sChord = triad_inf[0];#([0, 4, 7], 48, 'C\nma')
                if len(tri) == 0:
                    sChord = "-";
            
            sBass = self.TriadFinder.midnote_to_text(bass);
            #farbe für einstimmige passage anders (LILA)
            color = self.__plot_data_chordbass_get_color(i,einst_passagen);
            ax.text(xse[i] + 0.2,0,sBass, fontfamily = 'Tahoma', fontsize = 6, fontstretch = 'ultra-condensed',color = color);
            ax.text(xse[i],0.3,sChord, fontfamily = 'Tahoma', fontsize = 8, fontstretch = 'ultra-condensed', color = color);
        return 1;
    def __plot_data_modulation(self,ax,xse,modulation):
        # vertical bar grün hinmod ymax = 2
        # vertikal bar lila rückmod
        hin_mods = ModulationsData(modulation).get_hinmodulationen();
        rueck_mods =ModulationsData(modulation).get_rueckmodulationen();
        if not hin_mods is None:
            for hm in hin_mods:
                if len( hm) > 0:
                    ypse = [2] * len(hm);
                    ax.bar(hm, ypse , align = 'edge',width = 1.0 , color = (0.1, 0.9, 0.1) );
        if not rueck_mods is None:
            for rm in rueck_mods:
                if len( rm) > 0:
                    ypse = [2] * len(rm);
                    ax.bar(rm, ypse , align = 'edge',width = 1.0 , color = (0.7, 0.1, 0.9) );
        
        return 1;
    
    

      
