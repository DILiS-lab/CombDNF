### Description: CLI for drug combination feature generation with network-based features
### Author: Pauline Hiort
### Date: 2024-2025

import pandas as pd
import networkx as nx
import os
import numpy as np
import random as rnd
import ray
from tqdm import tqdm
from time import time
import click
import logging

# logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout, level=logging.INFO)


def create_network_from_csv_file(network_file, delim=',', weighted=False):
    """
    create unweighted or weighted network from csv file (first to columns should be node ids 
    and if weighted the third should be the weigths)
    input: 
        network_file (str) - path to csv file
        delim (str) - delimiter of the csv file
        weighted (bool) - set True, if the network is weighted
    output:
        g (networkx graph) - networkx graph
    """

    ### initiate networkx graph, read csv file and format node ids as str
    g = nx.Graph()
    df = pd.read_csv(network_file, sep=delim)
    ### format node ids as str
    df.iloc[:,0:2] = df.iloc[:,0:2].astype(str)
    ### create network
    if weighted:
        ### create weighted network
        g.add_weighted_edges_from(list(df.itertuples(index=False, name=None)))
    else:
        ### create unweighted network
        df = df.iloc[:,:2]
        g.add_edges_from(list(df.itertuples(index=False, name=None)))
    return g


def get_network_from_csv(network_file, output_path, delim=',', weighted=False, only_lcc=False):
    """
    create unweighted or weighted network from csv file, keep only largest connected component if specified
    code adapted from: https://github.com/emreg00/toolbox/blob/master/wrappers.py
    input:
        network_file (str) - path to csv file
        output_path (str) - path to output directory
        delim (str) - delimiter of the csv file
        weighted (bool) - set True, if the network is weighted
        only_lcc (bool) - set True, if only the largest connected component should be kept
    output:
        network (networkx graph) - networkx graph
    """
    ### create network from the given file; first 2 columns will be node ids and if weighted third column will be weights
    network = create_network_from_csv_file(network_file, delim=delim, weighted=weighted)
    logging.info(f'Network shape: {len(network.nodes())} nodes & {len(network.edges())} edges')
    
    ### reduce network to largest connected component and save it in text file with prefix 'lcc_'
    if only_lcc and not network_file.replace('\\', '/').split('/')[-1].startswith('lcc_'):
        logging.info('Shrinking network to its LCC ...')
        components_list = list(
            sorted(nx.connected_components(network), key=len, reverse=True)
        )
        network = network.subgraph(components_list[0])
        logging.info(f'Final shape: {len(network.nodes())} nodes & {len(network.edges())} edges')
        # network_lcc_file = '/'.join(network_file.replace('\\', '/').split('/')[:-1]) + '/lcc_' + network_file.replace('\\', '/').split('/')[-1]
        network_lcc_file = (
            f'{output_path}/lcc_'
            + network_file.replace('\\', '/').split('/')[-1]
        )
        
        ### save lcc to file, if it exists overwrite it
        if os.path.exists(network_lcc_file):
            logging.info(f'File {network_lcc_file} already exits. Overwriting the old file!')

        with open(network_lcc_file, 'w') as f:
            if weighted:
                f.write(f'%s\t%s\t%s\n' % ('proteinA', 'proteinB', 'weight'))
                for u,v,w in network.edges(data=True):
                    f.write(f'%s\t%s\t%s\n' % (u, v, w['weight']))
            else:
                f.write(f'%s\t%s\n' % ('proteinA', 'proteinB'))
                for u,v in network.edges():
                    f.write(f'%s\t%s\n' % (u, v))

    return network


### network feature generation ###

def get_shortest_path_length(G, nodeA, nodeB, weighted=False):
    """
    calculate shortest path length between nodeA and nodeB
    adapted from: https://github.com/emreg00/toolbox/blob/master/wrappers.py
    input:
    """
    ### get shortes path betweeen nodeA and nodeB (weighted or not)
    if weighted: return nx.shortest_path_length(G, nodeA, nodeB, weight='weight') ###!!!
    else: return nx.shortest_path_length(G, nodeA, nodeB)


def get_separation_between_sets(network, nodesA, nodesB, sp_lengths=None, weighted=False):
    """  
    calculate dAB in separation metric
    adapted from: https://github.com/emreg00/toolbox/blob/master/wrappers.py
    input:
        network (networkx graph): networkx graph
        nodesA (list): list of nodes
        nodesB (list): list of nodes
        sp_lengths (dict, optional): precalculated shortest path lengths. Defaults to None.
        weighted (bool, optional): if True, use weighted network. Defaults to False.
    output:
        float: separation for nodes in nodesA and nodesB
    """
    min_sp_distances = []
    nodesA_sp_dist = {}
    nodesB_sp_dist = {}
    ### get shortest path lengths between all nodes in nodesA and nodesB
    for nodeA in nodesA:
        for nodeB in nodesB:
            if sp_lengths is None and network is not None:
                d = get_shortest_path_length(network, nodeA, nodeB, weighted)
            else:
                if str(nodeA) == str(nodeB): d = 0
                else: d = sp_lengths[str(nodeA)][str(nodeB)]
            ### save shortest path distances for nodeA in dict
            nodesA_sp_dist.setdefault(nodeA, []).append(d)
            ### save shortest path distances for nodeB in dict
            nodesB_sp_dist.setdefault(nodeB, []).append(d)
    ### extract distances to closest node in nodesB for all nodes in nodesA
    for nodeA in nodesA:
        sp_distances = nodesA_sp_dist[nodeA]
        min_sp_distances.append(np.min(sp_distances))
    ### extract distances to closest node in nodesA for all nodes in nodesB
    for nodeB in nodesB:
        sp_distances = nodesB_sp_dist[nodeB]
        min_sp_distances.append(np.min(sp_distances))
    return np.mean(min_sp_distances)


def get_separation_within_set(network, nodes, sp_lengths=None, weighted=False):
    '''
    calculate dAA or dBB in separation metric 
    adapted from: https://github.com/emreg00/toolbox/blob/master/wrappers.py
    input:
        network (networkx graph): networkx graph
        nodes (list): list of nodes
        sp_lengths (dict, optional): precalculated shortest path lengths. Defaults to None.
        weighted (bool, optional): if True, use weighted network. Defaults to False.
    output:
        float: separation for nodes in nodes
    '''

    ### if only one node given return separation score 0
    if len(nodes) == 1: return 0
    min_sp_distances = []
    ### get shortest path lengths between all nodes in nodes
    for nodeA in nodes:
        sp_distances = []
        for nodeB in nodes:
            if nodeA == nodeB:
                continue
            if sp_lengths is None and network is not None:
                d = get_shortest_path_length(network, nodeA, nodeB, weighted)
            elif sp_lengths is None and network is None:
                logging.error('No network or shortest path lengths given for separation score computation!')
            else:
                d = sp_lengths[str(nodeA)][str(nodeB)]
            sp_distances.append(d)
        ### extract distance to closest node within the set (A or B)
        min_sp_distances.append(np.min(sp_distances))
    return np.mean(min_sp_distances)


def get_agg_shortest_path(network, nodesA, nodesB, sp_lengths=None, weighted=False):
    '''
    calculate mean, median, min and max shortest path lengths between nodes in nodesA and nodesB
    input:
        network (networkx graph) - networkx graph
        nodesA (list) - list of nodes
        nodesB (list) - list of nodes
        sp_lengths (dict) - precalculated shortest path lengths
        weighted (bool) - set True, if the network is weighted
    output:
        mean, median, min, max (float) - of shortest path lengths
    '''
    ### get shortest path lengths between all nodes in nodesA and nodesB
    distances = []
    for nodeA in nodesA:
        for nodeB in nodesB:
            if sp_lengths is None and network is not None:
                d = get_shortest_path_length(network, nodeA, nodeB, weighted)
            else:
                if str(nodeA) == str(nodeB): d = 0
                else: d = sp_lengths[str(nodeA)][str(nodeB)]
            distances.append(d)
    return np.mean(distances), np.median(distances), np.min(distances), np.max(distances)


def get_seperation_score(network, nodesA, nodesB, dBB=None, sp_lengths=None, weighted=False):
    """
    calculate separation metric score as proposed by Menche et al. 2015 for nodes in network
    code adapted from: https://github.com/emreg00/toolbox/blob/master/wrappers.py
    input:
        network (networkx graph): networkx graph
        nodesA (list): list of nodes
        nodesB (list): list of nodes
        dBB (float, optional): separation score within nodesB. Defaults to None.
        sp_lengths (dict, optional): precalculated shortest path lengths. Defaults to None.
        weighted (bool, optional): if True, use weighted network. Defaults to False.
    output:
        float: separation score for nodes in nodesA and nodesB

    """
    dAA = get_separation_within_set(network, nodesA, sp_lengths, weighted)
    if dBB is None:
        dBB = get_separation_within_set(network, nodesB, sp_lengths, weighted)
    else: dBB = dBB
    dAB = get_separation_between_sets(network, nodesA, nodesB, sp_lengths, weighted)
    d = dAB - (dAA + dBB) / 2.0
    return d


def get_distance_measures(network, nodesA, nodesB, idA, idB, dBB=None, sp_lengths=None, weighted=False):
    """
    calculate separation score, mean, median, min and max shortest path lengths and protein overlap between nodes in nodesA and nodesB
    input:
        network (networkx graph): networkx graph
        nodesA (list): list of nodes
        nodesB (list): list of nodes
        idA (str): id of nodesA
        idB (str): id of nodesB
        dBB (float, optional): separation score within nodesB. Defaults to None.
        sp_lengths (dict, optional): precalculated shortest path lengths. Defaults to None.
        weighted (bool, optional): if True, use weighted network. Defaults to False.
    output:
        idA (str), idB (str), d (float), protein_overlap (int), meanSP (float), medianSP (float), minSP (float), maxSP (float)
    """
    ### compute distance score between nodes in nodesA and nodesB
    ### if nodesA or nodesB not given return NaN
    if len(nodesA) != 0 and len(nodesB) != 0:
        d = get_seperation_score(network, nodesA, nodesB, dBB, sp_lengths, weighted)
        
        ### calculate mean, median, min and max shortest path lengths between nodes in nodesA and nodesB
        agg_sp = get_agg_shortest_path(network, nodesA, nodesB, sp_lengths, weighted)
        ### calculate protein overlap between nodes in nodesA and nodesB
        protein_overlap = len(set(nodesA).intersection(set(nodesB)))

        return (idA, idB, round(d, 4), protein_overlap, 
                round(agg_sp[0], 4), round(agg_sp[1], 4), 
                round(agg_sp[2], 4), round(agg_sp[3], 4)) #TODO
    else: return (idA, idB, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def calculate_drug_drug_distance_scores(network, drug_interactions, drug_targets, sp_lengths=None, weighted=False, progress_bar=True):
    """     
    get distance scores for given network and drug combination pairs
    input:
        network (networkx graph) - networkx graph of PPI network
        drug_interactions (pd.DataFrame) - list of drug combinations
        drug_targets (pd.DataFrame) - list of target nodes connected to each drug ID
        sp_lengths (dict, optional): precalculated shortest path lengths. Defaults to None.
        weighted (bool, optional): if True, use weighted network. Defaults to False.
        progress_bar (bool, optional): if True, show progress bar. Defaults to True.
    output:
        distance_scores (pd.DataFrame) - distance scores for drug-drug pairs
    """
    ### for each drug-drug pair compute distance score
    distance_scores = []
    for i, row in (tqdm(drug_interactions.iterrows(), total=len(drug_interactions), desc='Drug-drug distance scores calculation') if progress_bar else drug_interactions.iterrows()):
        drugA = row.iat[0]
        drugB = row.iat[1]
        targetsA = pd.Series(drug_targets.loc[drugA]['target']).tolist() #drug_targets[drug_targets.iloc[:,0] == drugA].iloc[:,1].tolist()
        targetsB = pd.Series(drug_targets.loc[drugB]['target']).tolist() #drug_targets[drug_targets.iloc[:,0] == drugB].iloc[:,1].tolist()
        ### compute distance score and other distance measures for given drug pair
        d = get_distance_measures(network, targetsA, targetsB, drugA, drugB, sp_lengths=sp_lengths, weighted=weighted)
        distance_scores.append(d)
    return pd.DataFrame(
        distance_scores, columns=['drugA', 'drugB', 's', 'op', 'meanSP', 'medianSP', 'minSP', 'maxSP']
    ).dropna()


def calculate_drug_drug_distance_scores_parallel_chunks(network, drug_interactions, drug_targets, sp_lengths=None, weighted=False, num_cpus=4):
    """
    calculate distance scores for given network and drug combination pairs in parallel
    input:
        network (networkx graph) - networkx graph of PPI network
        drug_interactions (pd.DataFrame) - list of drug combinations
        drug_targets (pd.DataFrame) - list of target nodes connected to each drug ID
        sp_lengths (dict, optional): precalculated shortest path lengths. Defaults to None.
        weighted (bool, optional): if True, use weighted network. Defaults to False.
        num_cpus (int, optional): number of cpus to use for parallel computation. Defaults to 4.
    output:
        distance_scores_res (pd.DataFrame) - distance scores for drug-drug pairs
    """

    ### define size for the dataframe chunks 
    n = int(drug_interactions.shape[0]/num_cpus/2)
    ### split dataframe into equal sized chunks of size n
    drug_interactions_chunks = [drug_interactions[i:i+n].copy() for i in range(0, drug_interactions.shape[0], n)]

    ### ray intitalisation for parallel distance score computation
    distance_scores_res = pd.DataFrame()
    separation_score_chunks = ray.remote(calculate_drug_drug_distance_scores)
    ray.init(include_dashboard=False, num_cpus=num_cpus)
    remote_network = ray.put(network)
    remote_drug_targets = ray.put(drug_targets)
    remote_sp_lengths = ray.put(sp_lengths)
    progress_bar = tqdm(total=len(drug_interactions), desc='Drug-drug distance scores calculation', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    futures = [separation_score_chunks.remote(remote_network, elem, remote_drug_targets, sp_lengths=remote_sp_lengths,
                                              weighted=weighted, progress_bar=False) for elem in drug_interactions_chunks]
    while len(futures):
        [done], futures = ray.wait(futures)
        distance_scores_res = pd.concat([distance_scores_res, ray.get(done)])
        progress_bar.update(len(ray.get(done)))
    progress_bar.close()
    ray.shutdown()
    return (
        pd.DataFrame(distance_scores_res, columns=['drugA', 'drugB', 's', 'op', 'meanSP', 'medianSP', 'minSP', 'maxSP'])
        .dropna()
        .reset_index(drop=True)
    )


def calculate_drug_disease_distance_scores(network, drugs, drug_targets, disease_nodes, disease_id, sp_lengths=None, weighted=False, progress_bar=True):
    """
    calculate disance scores for given network and drug-disease node sets
    input:
        network (networkx graph) - networkx graph of PPI network
        drugs (list) - list of drug IDs
        drug_targets (pd.DataFrame) - list of target nodes connected to each drug ID
        disease_nodes (list) - list of disease nodes
        disease_id (str) - id of disease
        sp_lengths (dict) - precalculated shortest path lengths
        weighted (bool) - set True, if the network is weighted
    output:
        distance_scores (pd.DataFrame) - distance scores for drug-disease pairs

    """
    distance_scores = []
    
    dBB = get_separation_within_set(network, disease_nodes, sp_lengths=sp_lengths, weighted=weighted)
    
    ### for each drug-disease pair compute distance score
    for drug in (tqdm(drugs, total=len(drugs)) if progress_bar else drugs):
        targets = pd.Series(drug_targets.loc[drug]['target']).tolist() 
        ### compute distance score for given drug-disease pair
        d = get_distance_measures(network, targets, disease_nodes, drug, disease_id, dBB, sp_lengths=sp_lengths, weighted=weighted)
        distance_scores.append(d)
    return pd.DataFrame(
        distance_scores, columns=['drug', 'disease', 's', 'op', 'meanSP', 'medianSP', 'minSP', 'maxSP']
    ).dropna()


def calculate_drug_disease_distance_scores_parallel_chunks(network, drugs, drug_targets, disease_nodes, disease_id, sp_lengths=None, weighted=False, num_cpus=4):
    """
    calculate distance scores for given network and drug-disease node sets in parallel
    input:
        network (networkx graph) - networkx graph of PPI network
        drugs (list) - list of drug IDs
        drug_targets (pd.DataFrame) - list of target nodes connected to each drug ID
        disease_nodes (list) - list of disease nodes
        disease_id (str) - id of disease
        sp_lengths (dict) - precalculated shortest path lengths
        weighted (bool) - set True, if the network is weighted
        num_cpus (int) - number of cpus to use for parallel computation
    output:
        distance_scores_res (pd.DataFrame) - distance scores for drug-disease pairs
    """
    ### define size for the dataframe chunks 
    n = int(len(drugs)/num_cpus/2)
    ### split datframe into equal sized chunks of size n
    drug_chunks = [drugs[i:i+n].copy() for i in range(0, len(drugs), n)]

    ### ray intitalisation for parallel distance score computation
    distance_scores_res = pd.DataFrame()
    separation_score_chunks = ray.remote(calculate_drug_disease_distance_scores)
    ray.init(include_dashboard=False, num_cpus=num_cpus)
    remote_network = ray.put(network)
    remote_drug_targets = ray.put(drug_targets)
    remote_disease_nodes = ray.put(disease_nodes)
    remote_sp_lengths = ray.put(sp_lengths)
    progress_bar = tqdm(total=len(drugs), desc='Drug-disease distance scores calculation', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    futures = [separation_score_chunks.remote(remote_network, elem, remote_drug_targets, remote_disease_nodes, disease_id,
                                       sp_lengths=remote_sp_lengths, weighted=weighted, progress_bar=False) for elem in drug_chunks]
    while len(futures):
        progress_bar.refresh()
        [done], futures = ray.wait(futures)
        distance_scores_res = pd.concat([distance_scores_res, ray.get(done)])
        progress_bar.update(len(ray.get(done)))
    progress_bar.close()
    ray.shutdown()
    return (
        pd.DataFrame(distance_scores_res, columns=['drug', 'disease', 's', 'op', 'meanSP', 'medianSP', 'minSP', 'maxSP'])
        .dropna()
        .reset_index(drop=True)
    )



##### z-score computation #####


def calculate_closest_distance(network, target_nodes, disease_nodes, sp_lengths=None, weighted=False):
    """
    calculate the closest distance between drug target nodes and disease nodes
    adapted from: https://github.com/emreg00/toolbox/blob/master/wrappers.py
    input:
        network (networkx graph) - networkx graph
        target_nodes (list) - list of target nodes
        disease_nodes (list) - list of disease nodes
        sp_lengths (dict) - precalculated shortest path lengths
        weighted (bool) - set True, if the network is weighted
    output:
        float: closest distance between drug target nodes and disease nodes
    """
    ### compute closest distance between drug target nodes and disease nodes
    min_distances = []
    for node_y in disease_nodes:
        distances_xy = []
        for node_x in target_nodes:
            if sp_lengths is None and network is None:
                raise ValueError('To calculate the z-scores either a network or a dictionary with the shortest paths has to be given!')
            elif sp_lengths is None and network is not None:
                distances_xy.append(get_shortest_path_length(network, node_x, node_y, weighted=weighted))
            else:
                distances_xy.append(sp_lengths[node_y][node_x])
        min_distances.append(min(distances_xy))
    return np.mean(min_distances)


def get_z_score(network, target_nodes, disease_nodes, drug_id, weighted=False, random_target_nodes=None, 
                random_disease_nodes=None, bins=None, n_random=100, min_bin_size=100, seed=452456, sp_lengths=None):
    """
    calculate z-scores for drug-disease pairs
    code adapted from: https://github.com/emreg00/toolbox/blob/master/wrappers.py
    input:
        network (networkx graph) - networkx graph of PPI network
        target_nodes (list) - list of target nodes
        disease_nodes (list) - list of disease nodes
        drug_id (str) - id of drug
        weighted (bool) - set True, if the network is weighted
        random_target_nodes (list) - list of random target nodes
        random_disease_nodes (list) - list of random disease nodes
        bins (list) - list of bins
        n_random (int) - number of random nodes
        min_bin_size (int) - minimum bin size
        seed (int) - random seed
        sp_lengths (dict) - precalculated shortest path lengths
    output:
        drug_id (str), zTD (float), zDT (float)
    """

    ### if no nodes given return NaN
    if len(target_nodes)==0 or len(disease_nodes)==0: return (drug_id, np.nan)

    ### compute distance between drug target and disease nodes
    dTD = calculate_closest_distance(network, target_nodes, disease_nodes, sp_lengths=sp_lengths, weighted=weighted)
    dDT = calculate_closest_distance(network, disease_nodes, target_nodes, sp_lengths=sp_lengths, weighted=weighted)

    # if node bins not given, bin the nodes by similar degree and min_bin_size
    if bins is None and (random_disease_nodes is None or random_target_nodes is None):
        ### if lengths is given, it will only use nodes with given length!
        bins = get_degree_binning(network, min_bin_size, sp_lengths)
    ### get random drug target nodes from bins
    if random_target_nodes is None:
        random_target_nodes = pick_random_nodes_matching_selected(network, bins, target_nodes, n_random, degree_aware=True, seed=seed)
        #get_random_nodes(drug_targets, network, bins=bins, n_random=n_random, min_bin_size=min_bin_size, seed=seed)
    ### get random disease nodes from bins
    if random_disease_nodes is None:
        random_disease_nodes = pick_random_nodes_matching_selected(network, bins, disease_nodes, n_random, degree_aware=True, seed=seed) 
        #get_random_nodes(disease_nodes, network, bins=bins, n_random=n_random, min_bin_size=min_bin_size, seed=seed)
    random_nodes_list = zip(random_target_nodes, random_disease_nodes)
    random_dTDs = np.empty(len(random_target_nodes))
    random_dDTs = np.empty(len(random_disease_nodes))
    for i, nodes_random in enumerate(random_nodes_list):
        random_target_nodes_set, random_disease_nodes_set = nodes_random
        random_dTDs[i] = calculate_closest_distance(network, random_target_nodes_set, random_disease_nodes_set, sp_lengths=sp_lengths, weighted=weighted)
        random_dDTs[i] = calculate_closest_distance(network, random_disease_nodes_set, random_target_nodes_set, sp_lengths=sp_lengths, weighted=weighted)
    
    ### calc mean and standard deviation of random distances
    TDmu, TDstd = np.mean(random_dTDs), np.std(random_dTDs)
    DTmu, DTstd = np.mean(random_dDTs), np.std(random_dDTs)
    
    ### calc z-scores
    zTD = 0.0 if TDstd == 0 else (dTD - TDmu) / TDstd
    zDT = 0.0 if DTstd == 0 else (dDT - DTmu) / DTstd
    
    return (drug_id, round(zTD, 4), round(zDT, 4))




def get_degree_binning(network, bin_size):
    """
    get degree bins for nodes in network
    adapted from: https://github.com/emreg00/toolbox/blob/master/network_utilities.py
    input:
        network (networkx graph) - networkx graph
        bin_size (int) - size of bins
    output:
        bins (list) - list of bins
    """
    ### sort and bin nodes by degree
    degree_to_nodes = {}
    for node, degree in network.degree(): #.iteritems(): # iterator in networkx 2.0
        degree_to_nodes.setdefault(degree, []).append(node)
    ### get list of degrees
    degrees_list = degree_to_nodes.keys()
    degrees_list = sorted(degrees_list)
    ### bin the degrees by bin_size or nodes in bins
    bins = []
    i = 0
    while i < len(degrees_list):
        ### add nodes to bin until bin_size is reached
        low_degree = degrees_list[i]
        nodes_bin = degree_to_nodes[degrees_list[i]]
        while len(nodes_bin) < bin_size:
            i+=1
            if i==len(degrees_list): break
            nodes_bin.extend(degree_to_nodes[degrees_list[i]])
        if i==len(degrees_list): i-=1
        high_degree = degrees_list[i]
        i+=1
        ### if final bin smaller than bin_sizes add nodes to previous bin
        ### otherwise add new bin
        if len(nodes_bin) < bin_size: ###!!!
            # print(nodes_bin)
            # print('bins', bins)
            low_degree_, high_degree_, nodes_bin_ = bins[-1]
            bins[-1] = (low_degree_, high_degree, nodes_bin_ + nodes_bin)
        else:
            bins.append((low_degree, high_degree, nodes_bin))
    return bins


def pick_random_nodes_matching_selected(network, bins, nodes_selected, n_random, degree_aware=True, connected=False, seed=None):
    """
    get random nodes from bins matching selected nodes
    adapted from: https://github.com/emreg00/toolbox/blob/master/network_utilities.py
    input:
        network (networkx graph) - networkx graph
        bins (list) - list of bins
        nodes_selected (list) - list of selected nodes
        n_random (int) - number of random nodes
        degree_aware (bool) - set True, if degree aware
        connected (bool) - set True, if connected
        seed (int) - random seed
    output:
        values (list) - list of random nodes
    """

    ### set seed
    if seed is not None: rnd.seed(seed)
    ### randomly select nodes n_random times
    values = []
    nodes = network.nodes()
    for _ in range(n_random):
        ### if degree aware select nodes with similar degree
        if degree_aware:
            if connected:
                raise ValueError("This is not implemented! Please either set 'degree_aware=True' or 'connected=True'!")
            ### get degree equivalent nodes for selected nodes
            node_to_equivalent_nodes = get_degree_equivalents(nodes_selected, bins, network)
            nodes_random = set()
            for node, equivalent_nodes in node_to_equivalent_nodes.items():
                chosen = rnd.choice(equivalent_nodes)
                #nodes_random.append(random.choice(equivalent_nodes))
                for _ in range(20):
                    # Try to find a distinct node (at most 20 times)
                    if chosen in nodes_random:
                        chosen = rnd.choice(equivalent_nodes)
                nodes_random.add(chosen)
            nodes_random = list(nodes_random)
        elif connected:
            nodes_random = [rnd.choice(nodes)]
            k = 1
            while k != len(nodes_selected):
                node_random = rnd.choice(nodes_random)
                node_selected = rnd.choice(network.neighbors(node_random))
                if node_selected in nodes_random: continue
                nodes_random.append(node_selected)
                k += 1
        else:
            nodes_random = rnd.sample(nodes, len(nodes_selected))
        values.append(nodes_random)
    return values


def get_degree_equivalents(network, nodes_subset, bins):
    """
    extract degree equivalent nodes for selected nodes
    input:
        network (networkx graph) - networkx graph
        nodes_subset (list) - list of nodes
        bins (list) - list of bins
    output:
        node_to_bin (dict) - dictionary of nodes with similar degree

    """
    ### return list of nodes with similar degree
    node_to_bin = {}
    for node in nodes_subset:
        deg = network.degree(node)
        for l, h, bin_nodes in bins:
            if l <= deg and h >= deg:
                node_to_bin[node] = [i for i in list(bin_nodes) if str(i) != str(node)]
                break
    return node_to_bin


def calculate_z_scores(network, disease_nodes, drug_targets, drugs, sp_lengths=None, weighted=False, nodes_bin_size=100, n_random=100):
    """
    calculate z-scores for all drug-disease pairs
    input:
        network (networkx graph) - networkx graph
        disease_nodes (list) - list of disease nodes
        drug_targets (pd.DataFrame) - list of target nodes connected to each drug ID
        drugs (list) - list of drug IDs
        sp_lengths (dict) - precalculated shortest path lengths
        weighted (bool) - set True, if the network is weighted
        nodes_bin_size (int) - size of bins
        n_random (int) - number of random nodes
    output:
        z_scores (pd.DataFrame) - z-scores for drug-disease pairs
    """

    ### get degree bins
    degree_bins = get_degree_binning(network, bin_size=nodes_bin_size, sp_lengths=sp_lengths)
    ### compute z-score for each drug
    result_set = []
    for drug in tqdm(drugs, total=len(drugs)):
        # targets = drug_targets[drug_targets['drug'] == drug]['target'].astype(str).tolist()
        targets = pd.Series(drug_targets.loc[drug]['target']).tolist()
        if len(targets) != 0:
            z = get_z_score(network, targets, disease_nodes, drug, weighted=weighted, bins=degree_bins, sp_lengths=sp_lengths, n_random=n_random)
            result_set.append(z)
    return pd.DataFrame(result_set, columns=['drug', 'zTD', 'zDT'])


def calculate_z_scores_parallel(network, disease_nodes, drug_targets, drugs, sp_lengths=None, weighted=False, nodes_bin_size=100, n_random=100, num_cpus=4):
    """
    calculate z-scores for all drug-disease pairs in parallel
    input:
        network (networkx graph) - networkx graph
        disease_nodes (list) - list of disease nodes
        drug_targets (pd.DataFrame) - list of target nodes connected to each drug ID
        drugs (list) - list of drug IDs
        sp_lengths (dict) - precalculated shortest path lengths
        weighted (bool) - set True, if the network is weighted
        nodes_bin_size (int) - size of bins
        n_random (int) - number of random nodes
        num_cpus (int) - number of cpus to use for parallel computation
    output:
        z_scores (pd.DataFrame) - z-scores for drug-disease pairs
    """

    ### get degree bins
    degree_bins = get_degree_binning(network, bin_size=nodes_bin_size, sp_lengths=sp_lengths)
    ### compute z-score for each drug
    result_set = []
    ### initialise ray for parallel computation
    z_score = ray.remote(get_z_score)
    ray.shutdown()
    ray.init(include_dashboard=False, num_cpus=num_cpus)
    remote_network = ray.put(network)
    remote_sp_lengths = ray.put(sp_lengths)
    futures = [z_score.remote(remote_network, 
                              pd.Series(drug_targets.loc[drug]['target']).tolist(),
                              disease_nodes, drug, weighted=weighted, bins=degree_bins, sp_lengths=remote_sp_lengths, n_random=n_random) 
                              for drug in drugs]
    progress_bar = tqdm(total=len(futures), desc='Drug-disease z-scores calculation', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    while len(futures):
        [done], futures = ray.wait(futures)
        result_set.append(ray.get(done))
        progress_bar.update()
    progress_bar.close()
    ray.shutdown()
    return pd.DataFrame(result_set, columns=['drug', 'zTD', 'zDT'])


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    '-n',
    '--network_file',
    type=click.Path(exists=True),
    required=True,
    help='Path of the comma separated input file (.csv)'
)
@click.option(
    '-t',
    '--targets_file',
    type=click.Path(exists=True),
    required=True,
    help='Path of the comma separated input file (.csv)'
)
@click.option(
    '-dm',
    '--disease_module_file',
    type=click.Path(exists=True),
    required=True,
    help='Path of the comma separated input file (.csv)'
)
@click.option(
    '-o',
    '--output_path',
    type=click.Path(exists=False),
    required=True,
    help='Path to the folder in which the output files should be placed'
)
@click.option(
    '-w',
    '--weights',
    is_flag=True)
@click.option(
    '-sp',
    '--shortest_paths',
    is_flag=True)
@click.option(
    '-spf',
    '--shortest_paths_file',
    type=click.Path(exists=True))
@click.option(
    '-dcf',
    '--drug_combi_sabs_file',
    type=click.Path(exists=True),
    default=None)
@click.option(
    '-c',
    '--num_cpus',
    default=1,
    type=int,
    help=''
)
@click.option(
    '-mem',
    '--max_mem',
    type=int,
    help=''
)


def main(network_file, targets_file, disease_module_file, output_path, weights, shortest_paths, shortest_paths_file, drug_combi_sabs_file, num_cpus, max_mem):
    """
    drug combination network feature generation CLI
    input:
        network_file (str) - path to the network file
        targets_file (str) - path to the drug target file
        disease_module_file (str) - path to the disease module file
        output_path (str) - path to the output folder
        weights (bool) - set True, if the network is weighted
        shortest_paths (bool) - set True, if shortest paths should be calculated
        shortest_paths_file (str) - path to the shortest paths file
        drug_combi_sabs_file (str) - path to the drug combination file
        num_cpus (int) - number of cpus to use for parallel computation
        max_mem (int) - maximum memory to use for parallel computation

    """

    ### create output folder if not exist
    if not os.path.exists(output_path):
        # logging.info('creating output folder ...')
        os.makedirs(output_path)

    logging.basicConfig(
        handlers=[logging.FileHandler(f'{output_path}/CombDNF.log'),
                  logging.StreamHandler()], 
        level=logging.INFO, 
        format='[%(asctime)s] %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info('Starting CombDNF feature generation ...')
    logging.info('Importing files ...')
    logging.info(f'Importing and creating network from file {network_file} ...')
    ppi_network = get_network_from_csv(network_file, output_path=output_path, delim='\t', weighted=weights, only_lcc=True)
    logging.info(f'Successful import of network.')
    

    ### format drug target data frame: reduce data frame to targets given in network, set drugs as index
    logging.info(f'importing drug target information from file {targets_file} ...')
    drug_targets_df = pd.read_csv(targets_file, sep='\t')
    drug_targets_df.columns = ['drug', 'target']
    drug_targets_df = drug_targets_df.astype(str)
    logging.info(f'successful import of {drug_targets_df.shape[0]} drug target connections.')
    
    logging.info(f'checking for drug targets in network ...')
    drug_targets_df = drug_targets_df[drug_targets_df['target'].astype(str).isin(set(ppi_network.nodes()))]
    drugs_list = list(drug_targets_df.iloc[:, 0].unique())
    drug_targets_df = drug_targets_df.set_index('drug')    
    logging.info(f'{len(drugs_list)} drugs extracted.')
    logging.info(f'{drug_targets_df.shape[0]} drug-target interactions with targets in network.')

    logging.info(f'combining all pairwise drug combination of {len(drugs_list)} drugs ...')
    all_drug_combos_df =  [(A, B) for i, A in enumerate(drugs_list) for B in drugs_list[i + 1:]]
    all_drug_combos_df = pd.DataFrame(all_drug_combos_df, columns=['drugA', 'drugB'])
    ### reduce drug interaction pairs to drug combinations with known targets in the network
    all_drug_combos_df = all_drug_combos_df[(all_drug_combos_df['drugA'].isin(list(drug_targets_df.index)) & all_drug_combos_df['drugB'].isin(list(drug_targets_df.index)))]
    logging.info(f'{len(all_drug_combos_df)} pairwise drug combinations created.')

    logging.info(f'importing disease module information from file {disease_module_file} ...')
    disease_module = pd.read_csv(disease_module_file, sep='\t')
    disease_module = disease_module.astype(str).drop_duplicates()
    logging.info(f'successful import of {disease_module.shape[0]} disease proteins.' )

    ### reduce disease nodes to nodes given in network
    logging.info(f'checking for disease module nodes in network ...')
    disease_nodes = disease_module[disease_module.iloc[:,0].astype(str).isin(set(ppi_network.nodes()))].iloc[:,0].astype(str).values
    # print(disease_nodes)
    logging.info(f'{len(disease_nodes)} disease nodes found in the network.')

    logging.info(f'saving all output files in {output_path}.')


    t=time()
    ###  shortest paths -- takes some time to load for the full network (file size > 500MB)
    if shortest_paths and not shortest_paths_file:
        t=time()
        if weights:
            logging.info(f'computing weighted shortest path lengths ...')
            sp_lengths_dict = dict(nx.shortest_path_length(ppi_network, weight='weight'))
        else:
            logging.info(f'computing unweighted shortest path lengths ...')
            sp_lengths_dict = dict(nx.shortest_path_length(ppi_network))
        sp_lengths_df = pd.DataFrame.from_dict(sp_lengths_dict, orient='index')#.sort_index(axis=0).sort_index(axis=1)
        sp_lengths_df.index = sp_lengths_df.index.astype(str)
        sp_lengths_df.columns = sp_lengths_df.columns.astype(str)
        sp_lengths_df = sp_lengths_df.sort_index(axis=0).sort_index(axis=1)
        
        if weights:
            sp_lengths_df.to_csv(f'{output_path}/CombDNF_weighted_shortest_path_lengths.tsv', sep='\t')
        else: 
            sp_lengths_df.to_csv(f'{output_path}/CombDNF_unweighted_shortest_path_lengths.tsv', sep='\t')
        
        del(sp_lengths_dict)
        logging.info(f'{len(sp_lengths_df)} shortest path lengths computed in {round((time()-t)/60, 3)} minutes.')

    elif shortest_paths_file:
        logging.info(f'importing shortest path lengths from {shortest_paths_file}...')
        #sp_lengths_df = pd.read_csv(shortest_paths_file, index_col=None, sep=',') # TODO
        sp_lengths_df = pd.read_csv(shortest_paths_file, index_col=0, header=0, sep='\t')
        sp_lengths_df.index = sp_lengths_df.index.astype(str)
        sp_lengths_df.columns = sp_lengths_df.columns.astype(str)
        sp_lengths_df = sp_lengths_df.sort_index(axis=0).sort_index(axis=1)
        logging.info(f'successful importing {len(sp_lengths_df)} shortest path lengths in {round((time()-t)/60, 3)} minutes.')
    else:
        sp_lengths_df = None
        logging.info('No shortest path lengths imported or computed. Calculating shortest paths on the go.')
    
    if num_cpus > 1:
        if drug_combi_sabs_file is None:
            logging.info('calculating drug-drug distance scores in parallel ...')
            t=time()
            drug_drug_distance_scores = calculate_drug_drug_distance_scores_parallel_chunks(ppi_network, all_drug_combos_df, drug_targets_df, sp_lengths=sp_lengths_df, weighted=weights, num_cpus=num_cpus)
            logging.info(f'drug-drug distance scores calculated in {round((time()-t)/60, 3)} minutes.')
            drug_drug_distance_scores.sort_values(by=['drugA', 'drugB']).to_csv(f'{output_path}/CombDNF_drug_drug_scores.tsv', sep='\t', index=False)
        
        logging.info('calculating drug-disease distance scores in parallel ...')
        t=time()
        drug_disease_distance_scores = calculate_drug_disease_distance_scores_parallel_chunks(ppi_network, drugs_list, drug_targets_df, disease_nodes, 'disease', sp_lengths=sp_lengths_df, weighted=weights, num_cpus=num_cpus)
        logging.info(f'drug-disease distance scores calculated in {round((time()-t)/60, 3)} minutes.')
        drug_disease_distance_scores.drop(columns=['disease']).to_csv(f'{output_path}/CombDNF_drug_disease_scores.tsv', sep='\t', index=False) #!!! TODO

        logging.info('calculating drug-disease z-scores in parallel ...')
        t=time()
        drug_disease_z_scores = calculate_z_scores_parallel(ppi_network, disease_nodes, drug_targets_df, drugs_list, sp_lengths=sp_lengths_df, weighted=weights, nodes_bin_size=100, n_random=100, num_cpus=num_cpus)
        logging.info(f'drug-disease z-scores calculated in {round((time()-t)/60, 3)} minutes.')
        drug_disease_z_scores.to_csv(f'{output_path}/CombDNF_drug_disease_z_scores.tsv', sep='\t', index=False)
    
    else:
        if drug_combi_sabs_file is None:
            logging.info('calculating drug-drug distance scores ...')
            t=time()
            drug_drug_distance_scores = calculate_drug_drug_distance_scores(ppi_network, all_drug_combos_df, drug_targets_df, sp_lengths=sp_lengths_df, weighted=weights) 
            logging.info(f'drug-drug distance scores calculated in {round((time()-t)/60, 3)} minutes.')
            drug_drug_distance_scores.sort_values(by=['drugA', 'drugB']).to_csv(f'{output_path}/CombDNF_drug_drug_scores.tsv', sep='\t', index=False)

        logging.info('calculating drug-disease distance scores ...')
        t=time()
        drug_disease_distance_scores = calculate_drug_disease_distance_scores(ppi_network, drugs_list, drug_targets_df, disease_nodes, 'disease', sp_lengths=sp_lengths_df, weighted=weights)
        logging.info(f'drug-disease distance scores calculated in {round((time()-t)/60, 3)} minutes.')
        drug_disease_distance_scores.drop(columns=['disease']).to_csv(f'{output_path}/CombDNF_drug_disease_scores.tsv', sep='\t', index=False)

        logging.info('calculating drug-disease z-scores ...')
        t=time()
        drug_disease_z_scores = calculate_z_scores(ppi_network, disease_nodes, drug_targets_df, drugs_list, sp_lengths=sp_lengths_df, weighted=weights, nodes_bin_size=100, n_random=100)
        logging.info(f'drug-disease z-scores calculated in {round((time()-t)/60, 3)} minutes.')
        drug_disease_z_scores.to_csv(f'{output_path}/CombDNF_drug_disease_z_scores.tsv', sep='\t', index=False)


    logging.info(f'assemblying final table ...')
    drugs_list = list(set(drug_drug_distance_scores.iloc[:,0]).union(set(drug_drug_distance_scores.iloc[:,1])))
    all_scores = drug_drug_distance_scores.iloc[:,0:2]
    all_scores['sAB'] = drug_drug_distance_scores.iloc[:,2]
    all_scores['overlapAB'] = drug_drug_distance_scores.iloc[:,3]
    all_scores['mean_spAB'] = drug_drug_distance_scores.iloc[:,4]
    all_scores['median_spAB'] = drug_drug_distance_scores.iloc[:,5]
    all_scores['min_spAB'] = drug_drug_distance_scores.iloc[:,6]
    all_scores['max_spAB'] = drug_drug_distance_scores.iloc[:,7]
    all_scores['zTAD'] = np.nan
    all_scores['zTBD'] = np.nan
    all_scores['zDTA'] = np.nan
    all_scores['zDTB'] = np.nan
    all_scores['sAD'] = np.nan
    all_scores['sBD'] = np.nan
    all_scores['overlapAD'] = np.nan
    all_scores['overlapBD'] = np.nan
    all_scores['mean_spAD'] = np.nan
    all_scores['mean_spBD'] = np.nan
    all_scores['median_spAD'] = np.nan
    all_scores['median_spBD'] = np.nan
    all_scores['min_spAD'] = np.nan
    all_scores['min_spBD'] = np.nan
    all_scores['max_spAD'] = np.nan
    all_scores['max_spBD'] = np.nan
    

    for drug in tqdm(drugs_list, total= len(drugs_list), desc='Assembly of final table'):
        zTD = drug_disease_z_scores.loc[drug_disease_z_scores.drug == drug, 'zTD'].iloc[0].astype(float)
        all_scores.loc[all_scores.drugA == drug, 'zTAD'] = zTD
        all_scores.loc[all_scores.drugB == drug, 'zTBD'] = zTD

        zDT = drug_disease_z_scores.loc[drug_disease_z_scores.drug == drug, 'zDT'].iloc[0].astype(float)
        all_scores.loc[all_scores.drugA == drug, 'zDTA'] = zDT
        all_scores.loc[all_scores.drugB == drug, 'zDTB'] = zDT

        s = drug_disease_distance_scores.loc[drug_disease_distance_scores.drug == drug, 's'].iloc[0].astype(float)
        op = drug_disease_distance_scores.loc[drug_disease_distance_scores.drug == drug, 'op'].iloc[0].astype(float)
        meansp = drug_disease_distance_scores.loc[drug_disease_distance_scores.drug == drug, 'meanSP'].iloc[0].astype(float)
        mediansp = drug_disease_distance_scores.loc[drug_disease_distance_scores.drug == drug, 'medianSP'].iloc[0].astype(float)
        minsp = drug_disease_distance_scores.loc[drug_disease_distance_scores.drug == drug, 'minSP'].iloc[0].astype(float)
        maxsp = drug_disease_distance_scores.loc[drug_disease_distance_scores.drug == drug, 'maxSP'].iloc[0].astype(float)
        all_scores.loc[all_scores.drugA == drug, 'sAD'] = s
        all_scores.loc[all_scores.drugB == drug, 'sBD'] = s
        all_scores.loc[all_scores.drugA == drug, 'overlapAD'] = op
        all_scores.loc[all_scores.drugB == drug, 'overlapBD'] = op
        all_scores.loc[all_scores.drugA == drug, 'mean_spAD'] = meansp
        all_scores.loc[all_scores.drugB == drug, 'mean_spBD'] = meansp
        all_scores.loc[all_scores.drugA == drug, 'median_spAD'] = mediansp
        all_scores.loc[all_scores.drugB == drug, 'medianspBD'] = mediansp
        all_scores.loc[all_scores.drugA == drug, 'min_spAD'] = minsp
        all_scores.loc[all_scores.drugB == drug, 'min_spBD'] = minsp
        all_scores.loc[all_scores.drugA == drug, 'max_spAD'] = maxsp
        all_scores.loc[all_scores.drugB == drug, 'max_spBD'] = maxsp

    
    all_scores.to_csv(f'{output_path}/CombDNF_scores.tsv', sep='\t', index=False)

    logging.info(f'final table assembled in {round((time()-t)/60, 3)} minutes.')
    logging.info('done.')

if __name__ == '__main__':
    main()