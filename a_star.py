import networkx as nx
from networkx.classes.coreviews import AdjacencyView
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

def construct_graph():
    #  - - - - - - - - - - - - - -
    # |(1,3), (2,3), (3,3), (4,3)|
    # |(1,2), (X,X), (3,2), (4,2)|
    # |(1,1), (2,1), (3,1), (4,1)|
    #  - - - - - - - - - - - - - -
    print('')

def main():
    print("Construct graph")
    construct_graph()

if __name__ == "__main__":
    main()