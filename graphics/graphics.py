from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt


def generateGraphLogo():
    # Create an empty graph
    G = nx.Graph()

    # Add nodes
    nodes = ['A', 'B', 'C', 'D', 'E']
    G.add_nodes_from(nodes)

    # Add edges (connections between nodes)
    edges = [('A', 'B'), ('B', 'D'), ('D', 'E'), ('E', 'A'), ('A', 'C')]
    G.add_edges_from(edges)

    # Define the node colors with RGB (0, 91, 58) for green
    node_colors = {'A': 'black', 'B': 'black', 'C': (0/255, 91/255, 58/255), 'D': 'black', 'E': (0/255, 91/255, 58/255)}

    # Draw the graph with larger node size and shorter edges
    pos = nx.spring_layout(G, seed=42, k=0.5)  # Reduce the 'k' value to make edges shorter
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=[node_colors[node] for node in G.nodes()], font_color='white',
            font_weight='bold')

    # Show the plot
    plt.show()


def generateWordClouds(textAxis):
    wordCloudEAP = WordCloud().generate(textAxis[0])  # for EAP
    wordCloudHPL = WordCloud().generate(textAxis[1])  # for HPL
    wordCloudMWS = WordCloud().generate(textAxis[3])  # for MWS

    print(textAxis[0])
    # print(df['author'][0])
    plt.imshow(wordCloudEAP, interpolation='bilinear')
    plt.show()

    print(textAxis[1])
    # print(df['author'][1])
    plt.imshow(wordCloudHPL, interpolation='bilinear')
    plt.show()

    print(textAxis[3])
    # print(df['author'][3])
    plt.imshow(wordCloudMWS, interpolation='bilinear')
    plt.show()


generateGraphLogo()