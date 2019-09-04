import tensorflow as tf
import difflib
import networkx as nx
from models import KerasTextClassifier

import spacy
nlp_ner = spacy.load("en_core_web_sm")

semEval_model = KerasTextClassifier(input_length=50, n_classes=10, max_words=15000)
semEval_model.load('models/SemEval_relation_model')
label_to_use = list(semEval_model.encoder.classes_)
global graph
graph = tf.get_default_graph()

def connect_graphs(mytriples):
    G = nx.DiGraph()
    for s, p, o in mytriples:
        G.add_edge(s, o, p=p)        
    
    """
    # Get components
    graphs = list(nx.connected_component_subgraphs(G.to_undirected()))
    
    # Get the largest component
    largest_g = max(graphs, key=len)
    largest_graph_center = ''
    largest_graph_center = get_center(nx.center(largest_g))
    
    # for each graph, find the centre node
    smaller_graph_centers = []
    for g in graphs:        
        center = get_center(nx.center(g))
        smaller_graph_centers.append(center)

    for n in smaller_graph_centers:
        if (largest_graph_center is not n):
            G.add_edge(largest_graph_center, n, p='with')
    """
    return G

def rank_by_degree(mytriples): #, limit):
    G = connect_graphs(mytriples)

    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')

    bw_centrality = nx.betweenness_centrality(G.to_undirected(), normalized=False)    
    nx.set_node_attributes(G, bw_centrality, 'betweenness')
    
    # Use this to draw the graph
    #draw_graph_centrality(G, degree_dict)
    #draw_graph_centrality(G, bw_centrality)

    Egos = nx.DiGraph()
    for a, data in sorted(G.nodes(data=True), key=lambda x: x[1]['degree'], reverse=True):
        ego = nx.ego_graph(G, a)
        Egos.add_edges_from(ego.edges(data=True))
        Egos.add_nodes_from(ego.nodes(data=True))
        
    ranked_triples = []
    #[subj, pred_verb, pred_semeval, pred_nyt, obj, subj_degree, obj_degree, subj_betweenness, obj_betweenness]
    for u, v, d in Egos.edges(data=True):
        #ranked_triples.append([u, d['p'], v])
        ranked_triples.append([u, d['p'], '', '', v, Egos.nodes[u]['degree'], Egos.nodes[v]['degree'],
                               Egos.nodes[u]['betweenness'], Egos.nodes[v]['betweenness']])
    return ranked_triples

# Add additional information to the triples for visualisation in the web app: degree, betweenness, SemEval relations, and entity types.
def add_visualisation_information(triples, text):
    print(triples)
    # Set degree centrality and betweenness centrality scores
    TRIPLES = rank_by_degree(triples)

    # Add SemEval relations  
    with graph.as_default():
        SemEval_data_format = []
        for s, v, _, _, o, _, _, _, _ in TRIPLES:
            SemEval_data_format.append('<e1>' + s + '<e1> ' + v + ' <e2>' + o + '<e2>')
        semEval_prediction = semEval_model.predict(SemEval_data_format)

    triples_SemEval_added = []
    for a, pred in zip(TRIPLES, semEval_prediction):
        s, v, _, v_nyt, o, s_d, o_d, s_b, o_b = a
        triples_SemEval_added.append([s, v, label_to_use[pred], v_nyt, o, s_d, o_d, s_b, o_b])

    # Get the named ent tags of the text via Spacy
    # This should ideally be implemented earlier in the pipeline rather than at the end (entities aren't context-sensitive this way).
    doc = nlp_ner(text)
    ents = {ent.text: ent.label_ for ent in doc.ents}

    # Convert the spacy entities (which are based on Ontonotes 5) into the four standard Wiki NER tags (PER, ORG, LOC, MISC)
    def onto2wiki(tag):
        d = {"PERSON": "PER", "NORP": "ORG", "FAC": "LOC", "ORG": "ORG", "GPE": "LOC", "ORDINAL": "O", "PERCENT": "O", "MONEY": "O", "CARDINAL": "O", "TIME": "O", "DATE": "O"}
        return d[tag] if tag in d else "MISC"    

    # Check the tagged spacy ents against a string to determine whether it is an entity.    
    # Performs basic string matching to determine the label.
    def get_label(string):        
        for ent in ents:
            r = difflib.SequenceMatcher(None, string, ent).ratio()
            if r > 0.75:
                return onto2wiki(ents[ent])
        return "O"

    for i, t in enumerate(triples_SemEval_added):
        s = t[0]
        o = t[4]       
        triples_SemEval_added[i].append(get_label(s))
        triples_SemEval_added[i].append(get_label(o))     
        
    return triples_SemEval_added
