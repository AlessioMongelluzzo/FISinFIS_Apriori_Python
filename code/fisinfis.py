import os
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
import numpy as np
from scipy.sparse import coo_matrix
import seaborn as sns
from itertools import combinations
from scipy.special import binom
import argparse
import matplotlib
import networkx as nx

from ar_metrics import *


class FISinFIS:
    """
    wrapper for sparse implementation of FISinFIS Apriori
    """
    def __init__(self, topN, min_supp, min_conf,
                 max_IDF_percentile, min_lift):
        """
        topN: top n% selection from IDF filtering for itemset
        min_supp: minimum support threshold
        min_conf: minimum confidence threshold
        max_IDF_percentile: defines maximum IDF percentile to keep 
                            higher IDFs will be discarded since too rare items -> typos etc
        min_lift: minimum lift for rules filtering
        """
        self.topN = topN
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.max_IDF_percentile = max_IDF_percentile
        self.min_lift = min_lift

    def initialize(self, data_folder, csv_name, verbose_cols):
        """
        """
        self.csv_name = csv_name
        self.aux_folder = "../aux_fisinfis_{}".format(csv_name)
        if not os.path.exists(self.aux_folder):
            os.makedirs(self.aux_folder)
        self.imgs_folder = os.path.join(self.aux_folder, "imgs")
        if not os.path.exists(self.imgs_folder):
            os.makedirs(self.imgs_folder)

        self.df = pd.read_csv(os.path.join(data_folder, csv_name))

        #reduce df for test here
        self.df = self.df[:100] # to be removed

        print("df shape:\t{}".format(self.df.shape))
        self.df.reset_index(drop=True, inplace=True)
        descr_dict_path = os.path.join(self.aux_folder, "{}_description_dict.pkl".format(csv_name))
        if not os.path.exists(descr_dict_path):
            # indexing merged transactions
            self.descriptions_dict = dict()
            for i in tqdm(range(self.df.shape[0]), desc='transactions processed'):
                r = self.df.iloc[i]
                self.descriptions_dict[i] = ""
                for c in verbose_cols:
                    self.descriptions_dict[i]+="{} ".format(r[c]).lower()
            with open(descr_dict_path, "wb") as f:
                pickle.dump(self.descriptions_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(descr_dict_path, "rb") as f:
                self.descriptions_dict = pickle.load(f)

    def cleanse(self, lang):
        """
        performs stop words removal and stemming with snowball stemming
        """
        snowball = SnowballStemmer(language=lang)
        tokenizer = RegexpTokenizer('\w+')
        stop_words = list(set(stopwords.words(lang)))
        if lang == 'italian':
            stop_words+= ["il", "lo", "la", "i", "gli", "le"] #exta stop words to remove
        
        self.source_descriptions_dict = self.descriptions_dict.copy()
        descr_dict_path_mod = os.path.join(self.aux_folder, "{}_description_dict_mod.pkl".format(self.csv_name))
        stem_dict_path = os.path.join(self.aux_folder, "{}_stem_dict.pkl".format(self.csv_name))
        self.stem_word_dict = defaultdict(set)
        descriptions_dict_mod = defaultdict(list)
        if (not os.path.exists(descr_dict_path_mod)) or (not os.path.exists(stem_dict_path)):
            # stemming and stop words removal
            for k in tqdm(self.descriptions_dict.keys(), desc='transactions stemmed and cleaned'):
                #[snowball.stem(x) for x in tokenizer.tokenize(descriptions_dict[k]) if x not in stop_words]
                for x in tokenizer.tokenize(self.descriptions_dict[k]):
                    if x not in stop_words:
                        stemmed = snowball.stem(x)
                        descriptions_dict_mod[k].append(stemmed)
                        self.stem_word_dict[stemmed].add(x)
            
            # reset key indices in case of empty descriptions found
            ni = 0
            self.descriptions_dict = dict()
            for k, v in descriptions_dict_mod.items():
                self.descriptions_dict[ni] = v
                ni+=1
            
            with open(descr_dict_path_mod, "wb") as f:
                pickle.dump(self.descriptions_dict, f, pickle.HIGHEST_PROTOCOL)
            with open(stem_dict_path, "wb") as f:
                pickle.dump(self.stem_word_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(descr_dict_path_mod, "rb") as f:
                self.descriptions_dict = pickle.load(f)
            with open(stem_dict_path, "rb") as f:
                self.stem_word_dict = pickle.load(f)

        # top topn items
        topn=30
        tl = [i for k, v in self.descriptions_dict.items() for i in v]
        mctl = Counter(tl).most_common(len(tl))
        plt.figure(figsize=(25,25))
        plt.barh([x[0] for x in mctl][:topn], [x[1] for x in mctl][:topn])
        plt.yticks(fontsize=25);
        plt.savefig(os.path.join(self.imgs_folder, "top_{}_items_occs_bar.pdf".format(topn)), bbox_inches='tight')

        items_list_path = os.path.join(self.aux_folder, "{}_items_list.pkl".format(self.csv_name))
        if not os.path.exists(items_list_path):
            self.I = list(set([x for k, v in self.descriptions_dict.items() for x in v]))
            with open(items_list_path, "wb") as f:
                pickle.dump(self.I, f)
        else:
            with open(items_list_path, "rb") as f:
                self.I = pickle.load(f)
        print("itemset I contains {} items.".format(len(self.I)))


    def move_to_sparse(self):
        """
        create coo and csc matrices for efficient metrics computation
        """
        if not os.path.exists(os.path.join(self.aux_folder, "{}_coo_matrix.pkl".format(self.csv_name))):
            # boolean matrix |T| x |I| representing occurrence of items in transactions
            dense_occs = np.zeros((len(self.descriptions_dict), len(self.I)), dtype = int)
            for k in tqdm(self.descriptions_dict.keys(), desc='transactions processed'):
                for i in self.descriptions_dict[k]:
                    dense_occs[k][self.I.index(i)]=1
                    
            # build coo sparse matrix
            self.coo_occs = coo_matrix(dense_occs)
            
            # dump sparse matrix
            with open(os.path.join(self.aux_folder, "{}_coo_matrix.pkl".format(self.csv_name)), "wb") as f:
                pickle.dump(self.coo_occs, f)
        else:
            with open(os.path.join(self.aux_folder, "{}_coo_matrix.pkl".format(self.csv_name)), "rb") as f:
                self.coo_occs = pickle.load(f)
        
        print("Percentage of matrix population (density):\t{}%".format(\
            self.coo_occs.getnnz()/np.prod(self.coo_occs.shape)*100))

        # to csc
        self.csc_occs = self.coo_occs.tocsc()
        self.item_index_dict = {self.I[x]:x for x in range(len(self.I))}
        self.index_item_dict = {x:self.I[x] for x in range(len(self.I))}

    def algorithm1(self):
        """
        FISinFIS algorithm 1
        """
        IDF_dict_path = os.path.join(self.aux_folder, "{}_IDF_dict.pkl".format(self.csv_name))
        if not os.path.exists(IDF_dict_path):
            self.items_IDF_dict = dict()
            for i in tqdm(self.I, desc = "items processed"):
                self.items_IDF_dict[i] = IDF([i], self.csc_occs, self.item_index_dict)
            
            # serialize idf dict to disk
            with open(IDF_dict_path, "wb") as f:
                pickle.dump(self.items_IDF_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(IDF_dict_path, "rb") as f:
                self.items_IDF_dict = pickle.load(f)

        IDF_sorted_list = [(k, v) for k, v in self.items_IDF_dict.items()]
        IDF_sorted_list.sort(key = lambda x : x[1], reverse = True)

        IDF_values_list = [x[1] for x in IDF_sorted_list]

        # select topn IDF values
        IDF_values_unique = list(set(IDF_values_list))
        IDF_values_unique.sort(reverse=True)
        lowest_idf = IDF_values_unique[:int(len(IDF_values_unique)*self.topN)][-1]
        highest_idf = IDF_values_unique[0]

        print("Discarded {}/{} items with IDF lower than minimum.".format(len([x for x in IDF_values_list if x<lowest_idf]), len(self.I)))

        IDF_counter = Counter(IDF_values_list).most_common(len(IDF_values_list))
        plt.figure(figsize=(20, 10))
        plt.title("IDF distribution kde - topN = {}%".format(self.topN*100))
        sns.kdeplot(IDF_values_list, label = 'IDF kde');
        plt.vlines(lowest_idf, min(plt.yticks()[0]), max(plt.yticks()[0]), color = 'red', label = 'IDF threshold')
        plt.axvspan(min(plt.xticks()[0]), lowest_idf, facecolor = 'red', alpha = 0.1, label = 'discarded region')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self.imgs_folder, "idf_kde_topN_{}.pdf".format(self.topN)), bbox_inches='tight')

        # single items support plot
        support_values = [supp([x], None, self.csc_occs, self.item_index_dict) for x in self.I]
        support_values.sort()
        high_supp_items = [x for x in self.I if supp([x], None, self.csc_occs, self.item_index_dict)>=self.min_supp]
        num_discarded_items = len([x for x in support_values if x<self.min_supp])
        print("Discarding {}/{} items with support lower than min_supp={}\nLeft items:\t{}".format(
            num_discarded_items, len(self.I), self.min_supp, len(self.I)-num_discarded_items))

        surviving_items = [x for x in high_supp_items if IDF([x], self.csc_occs, self.item_index_dict)>=lowest_idf]
        print("{}/{} left items with IDF > minimum IDF_threshold ({} = lowest IDF value to select topN = top {}%)".format(
            len(surviving_items), len(self.I)-num_discarded_items, lowest_idf, self.topN*100))


        plt.figure(figsize=(20, 10))
        plt.title("Support distribution kde - min_supp={}".format(self.min_supp))
        sns.kdeplot(support_values, label = 'items support');
        plt.vlines(self.min_supp, min(plt.yticks()[0]), max(plt.yticks()[0]), color = 'red', label = 'min_supp')
        plt.axvspan(min(plt.xticks()[0]), self.min_supp, facecolor = 'red', alpha = 0.1, label = 'discarded region')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.imgs_folder, "support_plot_min_supp_{}.pdf".format(self.min_supp)), bbox_inches='tight')

        # filter too high IDF = terms occurring too rarely
        max_IDF = np.percentile(IDF_values_unique, self.max_IDF_percentile)

        # filtered items by IDF topN selection
        tempk = []
        tempk =  [k for k, v in self.items_IDF_dict.items() if v >=lowest_idf and v <= max_IDF]# top N % items based on IDF
        print("selected {}/{} items with topN={} leading to minimum IDF of {}.\
        \nDiscarded {} elements with IDF value higher than maximum.".format(
            len(tempk), len(self.I), self.topN, lowest_idf, len([x for x in IDF_values_list if x>max_IDF])))

        # actual algo 1
        k=1
        # INITIALIZE
        # create FIS and inFIS sets filtering over support 
        self.FIS = []
        self.inFIS = []
        for i in tqdm(tempk, desc="filtered items processed"):
            if supp([i], None, self.csc_occs, self.item_index_dict)>=self.min_supp:
                self.FIS.append([i])
            else:
                self.inFIS.append([i])
        print("FIS[1] len: {}".format(len(self.FIS)))
        print("inFIS[1] len: {}".format(len(self.inFIS)))

        # builds dict like
        # {k: {(i1, i2): supp_value>0}}
        # for k >= 2
        #
        k_comb_FIS_dict = defaultdict(dict)
        k_comb_inFIS_dict = defaultdict(dict)

        for ci in tqdm(range(self.csc_occs.shape[1]-1), desc = 'main column'):
            multarr = self.csc_occs[:, ci]
            ma_supp = multarr.getnnz()/self.csc_occs.shape[0]
            k=2
            oci = ci+1
            while(ma_supp>0 and oci<self.csc_occs.shape[1]):
                multarr = multarr.multiply(self.csc_occs[:, oci])
                ma_supp = multarr.getnnz()/self.csc_occs.shape[0]
                if ma_supp>0: # minsup = 0 according to paper notation in Algorithm 1
                    if ma_supp > self.min_supp:
                        k_comb_FIS_dict[k][tuple([self.index_item_dict[x] for x in range(ci, oci+1)])] = ma_supp
                    else:
                        k_comb_inFIS_dict[k][tuple([self.index_item_dict[x] for x in range(ci, oci+1)])] = ma_supp
                k+=1
                oci+=1

        for k in k_comb_inFIS_dict.keys():
            for comb, csupp in k_comb_inFIS_dict[k].items():
                self.inFIS.append(comb)

        for k in k_comb_FIS_dict.keys():
            for comb, csupp in k_comb_FIS_dict[k].items():
                self.FIS.append(comb)

        print("FIS len:\t{}".format(len(self.FIS)))
        print("inFIS len:\t{}".format(len(self.inFIS)))

        # serialize FIS e inFIS per params
        FIS_apriori_path = os.path.join(self.aux_folder, "FIS_apriori_topN_{}_minsupp_{}_max_idf_perc_{}_min_lift_{}".format(
            self.topN, self.min_supp, self.max_IDF_percentile, self.min_lift))
        inFIS_apriori_path = os.path.join(self.aux_folder, "inFIS_apriori_topN_{}_minsupp_{}_max_idf_perc_{}_min_lift_{}".format(
            self.topN, self.min_supp, self.max_IDF_percentile, self.min_lift))
        if not os.path.exists(FIS_apriori_path):
            with open(FIS_apriori_path, "wb") as f:
                pickle.dump(self.FIS, f, pickle.HIGHEST_PROTOCOL)
        if not os.path.exists(inFIS_apriori_path):
            with open(inFIS_apriori_path, "wb") as f:
                pickle.dump(self.inFIS, f, pickle.HIGHEST_PROTOCOL)

    def generateARs(self, FIS, inFIS, min_conf, min_lift):
        """
        aux: Algorithm 2 part to generate ARs
        """
        nPARFIS = 0
        nNARFIS = 0
        nPARinFIS = 0
        nNARinFIS = 0
        PAR = []
        NAR_neg1 = []
        NAR_neg2 = []
        NAR_neg12 = []
        # ARs from FIS
        for fiscomb in tqdm(combinations(FIS, 2), desc="ARs processed from FIS", total = int(binom(len(FIS), 2))):
            if fiscomb[0] != fiscomb[1]:
                # PARs
                rconf = conf(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                rlift = lift(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                if (rconf > min_conf) and (rlift > min_lift):
                    PAR.append([fiscomb, rconf, rlift])
                    nPARFIS+=1
                # NARs
                rconf = conf_neg1(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                rlift = lift_neg1(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                if (rconf > min_conf) and (rlift > min_lift):
                    NAR_neg1.append([fiscomb, rconf, rlift])
                    nNARFIS+=1
                rconf = conf_neg2(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                rlift = lift_neg2(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                if (rconf > min_conf) and (rlift > min_lift):
                    NAR_neg2.append([fiscomb, rconf, rlift])
                    nNARFIS+=1

        # ARs from inFIS
        for fiscomb in tqdm(combinations(inFIS, 2), desc="ARs processed from inFIS", total = int(binom(len(inFIS), 2))):
            if fiscomb[0] != fiscomb[1]:
                # PARs
                rconf = conf(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                rlift = lift(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                if (rconf > min_conf) and (rlift > min_lift):
                    PAR.append([fiscomb, rconf, rlift])
                    nPARinFIS+=1
                # NARs
                rconf = conf_neg1(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                rlift = lift_neg1(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                if (rconf > min_conf) and (rlift > min_lift):
                    NAR_neg1.append([fiscomb, rconf, rlift])
                    nNARinFIS+=1
                rconf = conf_neg2(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                rlift = lift_neg2(fiscomb[0], fiscomb[1], self.csc_occs, self.item_index_dict)
                if (rconf > min_conf) and (rlift > min_lift):
                    NAR_neg2.append([fiscomb, rconf, rlift])
                    nNARinFIS+=1

        print("generated {} PARs and {} NARs.\n{} PAR from FIS\n{} NAR from FIS\n{}\
        PAR from inFIS\n{} NAR from inFIS".format(len(PAR), len(NAR_neg1+NAR_neg2+NAR_neg12),
                                                  nPARFIS, nNARFIS, nPARinFIS, nNARinFIS))
           
        # (items, confidence, lift)
        return (PAR, NAR_neg1, NAR_neg2, NAR_neg12)

    def negate_consequent(self, cons_conf_lift_list):
        """
        aux
        """
        return ("! "+str(cons_conf_lift_list[0]), cons_conf_lift_list[1], cons_conf_lift_list[2])

    def algorithm2(self):
        """
        FISinFIS algorithm2
        """

        # compute PARs and NARs if not on disk already for same parameters
        PAR_path = os.path.join(self.aux_folder, "PAR_topN_{}_minsupp_{}_max_idf_perc_{}_minconf_{}_min_lift_{}.pkl".format(
            self.topN, self.min_supp, self.max_IDF_percentile, self.min_conf, self.min_lift))
        NAR_neg1_path = os.path.join(self.aux_folder, "NAR_neg1_topN_{}_minsupp_{}_max_idf_perc_{}_minconf_{}_min_lift_{}.pkl".format(
            self.topN, self.min_supp, self.max_IDF_percentile, self.min_conf, self.min_lift))
        NAR_neg2_path = os.path.join(self.aux_folder, "NAR_neg2_topN_{}_minsupp_{}_max_idf_perc_{}_minconf_{}_min_lift_{}.pkl".format(
            self.topN, self.min_supp, self.max_IDF_percentile, self.min_conf, self.min_lift))
        NAR_neg12_path = os.path.join(self.aux_folder, "NAR_neg12_topN_{}_minsupp_{}_max_idf_perc_{}_minconf_{}_min_lift_{}.pkl".format(
            self.topN, self.min_supp, self.max_IDF_percentile, self.min_conf, self.min_lift))

        if not os.path.exists(PAR_path) or not os.path.exists(NAR_neg1_path) or not os.path.exists(NAR_neg2_path) or not os.path.exists(NAR_neg12_path): 
            self.PAR, self.NAR_neg1, self.NAR_neg2, self.NAR_neg12 = self.generateARs(self.FIS, self.inFIS, self.min_conf, self.min_lift)
            with open(PAR_path, "wb") as f:
                pickle.dump(self.PAR, f, pickle.HIGHEST_PROTOCOL)
                
            with open(NAR_neg1_path, "wb") as f:
                pickle.dump(self.NAR_neg1, f, pickle.HIGHEST_PROTOCOL)
            
            with open(NAR_neg2_path, "wb") as f:
                pickle.dump(self.NAR_neg2, f, pickle.HIGHEST_PROTOCOL)
                
            with open(NAR_neg12_path, "wb") as f:
                pickle.dump(self.NAR_neg12, f, pickle.HIGHEST_PROTOCOL)

        else:
            print("Already computed PARs and NARs for parameters topN {}, minsupp {}, max_idf_perc {}, minconf {}!".format(
                self.topN, self.min_supp, self.max_IDF_percentile, self.min_conf))
            with open(PAR_path, "rb") as f:
                self.PAR = pickle.load(f)
            with open(NAR_neg1_path, "rb") as f:
                self.NAR_neg1 = pickle.load(f)
            with open(NAR_neg2_path, "rb") as f:
                self.NAR_neg2 = pickle.load(f)
            with open(NAR_neg12_path, "rb") as f:
                self.NAR_neg12 = pickle.load(f)

        # for rules A -> B with confidence c1 and lift l1, A -> C with c2 and l2, build dict like
        # {A : [(B, c1, l1), (C, c2, l2), ...], ...}
        # 
        print("Building antecedent-consequent dicts...")
        self.PAR_antecedent_consequents_dict = defaultdict(list)
        for parlist in tqdm(self.PAR, desc="PARs"):
            antec = parlist[0][0]
            consec = parlist[0][1]
            if type(antec) == list: # list if single element
                antec = antec[0]
            if type(consec) == list: # list if single element
                consec = consec[0]
            self.PAR_antecedent_consequents_dict[antec].append((consec, parlist[1], parlist[2]))

        self.NAR_neg1_antecedent_consequents_dict = defaultdict(list)
        for narlist in tqdm(self.NAR_neg1, desc="NARs neg1"):
            antec = narlist[0][0]
            consec = narlist[0][1]
            if type(antec) == list: # list if single element
                antec = antec[0]
            if type(consec) == list: # list if single element
                consec = consec[0]
            self.NAR_neg1_antecedent_consequents_dict[antec].append((consec, narlist[1], narlist[2]))

        self.NAR_neg2_antecedent_consequents_dict = defaultdict(list)
        for narlist in tqdm(self.NAR_neg2, desc="NARs neg2"):
            antec = narlist[0][0]
            consec = narlist[0][1]
            if type(antec) == list: # list if single element
                antec = antec[0]
            if type(consec) == list: # list if single element
                consec = consec[0]
            self.NAR_neg2_antecedent_consequents_dict[antec].append((consec, narlist[1], narlist[2]))
            
        self.NAR_neg12_antecedent_consequents_dict = defaultdict(list)
        for narlist in tqdm(self.NAR_neg12, desc="NARs neg12"):
            antec = narlist[0][0]
            consec = narlist[0][1]
            if type(antec) == list: # list if single element
                antec = antec[0]
            if type(consec) == list: # list if single element
                consec = consec[0]
            self.NAR_neg12_antecedent_consequents_dict[antec].append((consec, narlist[1], narlist[2]))

        # negate antecedent in neg1 rules
        for k in tqdm(self.NAR_neg1_antecedent_consequents_dict.keys(), desc = "neg1 rules processed"):
            v = self.NAR_neg1_antecedent_consequents_dict[k]
            self.NAR_neg1_antecedent_consequents_dict["! "+str(k)] = v

        # negate consequent in neg2 rules
        for k in tqdm(self.NAR_neg2_antecedent_consequents_dict.keys(), desc = "neg2 rules processed"):
            v = self.NAR_neg2_antecedent_consequents_dict[k]
            self.NAR_neg2_antecedent_consequents_dict[k] = list(map(lambda x: self.negate_consequent(x), v))
            
        # negate antecedent and consequent in neg12 rules
        for k in tqdm(self.NAR_neg12_antecedent_consequents_dict.keys(), desc = "neg12 rules processed"):
            v = self.NAR_neg12_antecedent_consequents_dict[k]
            self.NAR_neg12_antecedent_consequents_dict["! "+str(k)] = list(map(lambda x: self.negate_consequent(x), v))

        # create single ar dict
        self.AR_antecedent_consequent_dict = dict()

        for k in tqdm(self.PAR_antecedent_consequents_dict.keys(), desc="PARs processed"):
            v = self.PAR_antecedent_consequents_dict[k]
            if k in self.NAR_neg2_antecedent_consequents_dict.keys():
                self.AR_antecedent_consequent_dict[k] = v+self.NAR_neg2_antecedent_consequents_dict[k]
            else:
                self.AR_antecedent_consequent_dict[k] = v

        for k in tqdm(self.NAR_neg1_antecedent_consequents_dict.keys(), desc = "NARs neg1 processed"):
            v = self.NAR_neg1_antecedent_consequents_dict[k]
            if k in self.NAR_neg12_antecedent_consequents_dict.keys():
                self.AR_antecedent_consequent_dict[k] = v+self.NAR_neg12_antecedent_consequents_dict[k]
            else:
                self.AR_antecedent_consequent_dict[k] = v

        for k in tqdm(self.NAR_neg2_antecedent_consequents_dict.keys(), desc = "NARs neg2 processed"):
            v = self.NAR_neg2_antecedent_consequents_dict[k]
            if k in self.PAR_antecedent_consequents_dict.keys():
                self.AR_antecedent_consequent_dict[k] = v+self.PAR_antecedent_consequents_dict[k]
            else:
                self.AR_antecedent_consequent_dict[k] = v
            
        for k in tqdm(self.NAR_neg12_antecedent_consequents_dict.keys(), desc = "NARs neg12 processed"):
            v = self.NAR_neg12_antecedent_consequents_dict[k]
            if k in self.NAR_neg1_antecedent_consequents_dict.keys():
                self.AR_antecedent_consequent_dict[k] = v+self.NAR_neg1_antecedent_consequents_dict[k]
            else:
                self.AR_antecedent_consequent_dict[k] = v

    def plot_ARs_graphs(self):
        """
        plots and serialize ARs as graphs
        """
        matplotlib.use('Agg')
        AR_graphs_dir = os.path.join(self.imgs_folder, "ARs_graph_topN_{}_minsupp_{}_max_idf_perc_{}_minconf_{}_min_lift_{}".format(
            self.topN, self.min_supp, self.max_IDF_percentile, self.min_conf, self.min_lift))

        if not os.path.exists(AR_graphs_dir):
            os.makedirs(AR_graphs_dir)

        for k in tqdm(self.AR_antecedent_consequent_dict.keys(), desc="antecedent"):
            v = self.AR_antecedent_consequent_dict[k]
            plt.figure(figsize=(20, 20));
            G = nx.DiGraph();
            edge_labels = dict()
            for conslist in v:
                node1 = k
                node2 = conslist[0]
                weight = "c={:.2f} l={:.2f}".format(conslist[1], conslist[2])
                G.add_edge(node1, node2, label=str(weight), weight = conslist[1]*conslist[2]/2);
                edge_labels[(node1, node2)] = weight;
            edges = G.edges()
            weights = [G[u][v]['weight'] for u,v in edges]
            pos = nx.spring_layout(G);
            nx.draw_networkx(G, pos=pos, node_color='#92f0eb', font_size=15, width = weights); 
            nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=15);
            plt.savefig(os.path.join(AR_graphs_dir, "{}.pdf".format(k)), bbox_inches='tight')
            plt.clf();

        # visualize PARs only
        matplotlib.use('Agg')
        PAR_graphs_dir = os.path.join(self.imgs_folder, "PARs_graph_topN_{}_minsupp_{}_max_idf_perc_{}_minconf_{}_min_lift_{}".format(
            self.topN, self.min_supp, self.max_IDF_percentile, self.min_conf, self.min_lift))

        if not os.path.exists(PAR_graphs_dir):
            os.makedirs(PAR_graphs_dir)

        for k in tqdm(self.PAR_antecedent_consequents_dict.keys(), desc="antecedent"):
            v = self.PAR_antecedent_consequents_dict[k]
            plt.figure(figsize=(20, 20));
            G = nx.DiGraph();
            edge_labels = dict()
            for conslist in v:
                node1 = k
                node2 = conslist[0]
                weight = "c={:.2f} l={:.2f}".format(conslist[1], conslist[2])
                G.add_edge(node1, node2, label=str(weight), weight = conslist[1]*conslist[2]/2);
                edge_labels[(node1, node2)] = weight;
            edges = G.edges()
            weights = [G[u][v]['weight'] for u,v in edges]
            pos = nx.spring_layout(G);
            nx.draw_networkx(G, pos=pos, node_color='#92f0eb', font_size=15, width = weights); 
            nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=15);
            plt.savefig(os.path.join(PAR_graphs_dir, "{}.pdf".format(k)), bbox_inches='tight')
            plt.clf();



    def show_stem_transactions(self, selected_stem):
        """
        shows info about stem item and transactions containg the corresponding items to the selected stem
        input:
            - selected_stem: string or list of strings
        """
        if type(selected_stem) == list:
            corresponding_words = [self.stem_word_dict[x] for x in selected_stem]
            islist = True
        elif type(selected_stem) == str:
            corresponding_words = self.stem_word_dict[selected_stem]
            islist = False
        else:
            print("Unrecognized type for selected_item {}".format(type(selected_stem)))
            return None
        
        num_matches = 0
        if islist:
            for ei in range(len(selected_stem)):
                cw = len(corresponding_words[ei])
                print("Found {} items corresponding to stem {}.".format(cw, selected_stem[ei]))
                if cw == 0:
                    print("(Check the stem spelling.)")
                num_matches+=cw
        else:
            cw = len(corresponding_words)
            print("Found {} corresponding items to stem {}.".format(cw, selected_stem))
            num_matches=cw
        if num_matches>0:
            see_words_choice = None
        else:
            print("Check the stem(s) spelling.")
            see_words_choice = "no"
        while see_words_choice not in ["yes", "no"]:
            see_words_choice = input("Do you want to see the items?\t[yes/no]")
        if see_words_choice=="yes":
            print(corresponding_words)
        
        if islist:
            selected_desc_indices = [k for k, v in self.descriptions_dict.items() if all(s in v for s in selected_stem)]
        else:
            selected_desc_indices = [k for k, v in self.descriptions_dict.items() if selected_stem in v]
            
        print("Found {} transactions containing stemmed item {}.".format(len(selected_desc_indices), selected_stem))
        
        if len(selected_desc_indices) > 0:
            see_sent_choice = None
        else:
            see_sent_choice = "no"
        while see_sent_choice not in ["yes", "no"]:
            see_sent_choice = input("Do you want to see the transactions?\t[yes/no]")
        if see_sent_choice=="yes":
            print()
            tabs_len = 10
            tabs_num = np.ceil(len(selected_desc_indices)/tabs_len)
            tr_it = 0
            continue_transactions = "yes"
            shown_t = 0
            while continue_transactions == "yes" and shown_t<len(selected_desc_indices):
                filtered_transactions_indices = selected_desc_indices[tabs_len*tr_it:tabs_len*(tr_it+1)]
                for k in filtered_transactions_indices:
                    print("- {}\n".format(self.source_descriptions_dict[k]))

                print("Shown {}/{} transactions.".format(shown_t+len(filtered_transactions_indices), len(selected_desc_indices)))
                tr_it+=1
                shown_t+=len(filtered_transactions_indices)
                if shown_t < len(selected_desc_indices):
                    continue_transactions = None
                else:
                    continue_transactions = "no"
                while continue_transactions not in ["yes", "no"]:
                    continue_transactions = input("Continue?\t[yes/no]")


def main(topN, min_supp, min_conf, max_IDF_percentile, min_lift,
         data_folder, csv_name, verbose_cols, lang):
        
    fisinfis = FISinFIS(topN, min_supp, min_conf,
                        max_IDF_percentile, min_lift)

    fisinfis.initialize(data_folder, csv_name, verbose_cols)
    fisinfis.cleanse(lang)  
    fisinfis.move_to_sparse()
    fisinfis.algorithm1()
    fisinfis.algorithm2()
    fisinfis.plot_ARs_graphs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topN', '-topN', help='topN')
    parser.add_argument('--min_supp', '-ms', help='minimum support')
    parser.add_argument('--min_conf', '-mc', help='minimum confidence', default = 0.6)
    parser.add_argument('--max_IDF_percentile', '-MIDF', help='maximum idf percentile', default = 95)
    parser.add_argument('--min_lift', '-ml', help='minimum lift', default = 1.01)
    parser.add_argument('--data_folder', '-df', help='data folder where csv is stored')
    parser.add_argument('--csv_name', '-csv', help='csv file name relative to data folder')
    parser.add_argument('--verbose_cols', '-vc', help='verbose columns', nargs = "+")
    parser.add_argument('--language', '-lang', help='language from nltk language')
    args = parser.parse_args()

    topN = float(args.topN)
    min_supp = float(args.min_supp )
    min_conf = float(args.min_conf)
    max_IDF_percentile = float(args.max_IDF_percentile )
    min_lift = float(args.min_lift)
    data_folder = str(args.data_folder)
    csv_name = str(args.csv_name)
    verbose_cols = list(args.verbose_cols)
    lang = str(args.language)

    main(topN, min_supp, min_conf, max_IDF_percentile, min_lift,
         data_folder, csv_name, verbose_cols, lang)