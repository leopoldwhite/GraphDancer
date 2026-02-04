import os
import sys

# Ensure we can import from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from graph_constants import GRAPH_DEFINITION

def get_examples():
    examples = {}
    
    # MAPLE
    maple_ex = f"""
Definition of the graph: 
{GRAPH_DEFINITION['maple']}
Question: 
When was the paper Strongly Interacting Higgs Sector in the Minimal Standard Model published?
<think>The question is asking some basic information of a node (Strongly Interacting Higgs Sector in the Minimal Standard Model). We need to find the node in the graph.</think><graph>RetrieveNode[Strongly Interacting Higgs Sector in the Minimal Standard Model]</graph>
<information>The ID of this node is 3101448248.</information>
<think>The question is asking the published date of a paper, we need to check the node feature (year) from the graph.</think><graph>NodeFeature[3101448248, year]</graph>
<information>1993</information>
<think>The published date of the paper is 1993.</think><answer>1993</answer>

Definition of the graph: 
{GRAPH_DEFINITION['maple']}
Question: 
How many authors do the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars have?
<think>The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.</think><graph>RetrieveNode[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]</graph>
<information>The ID of this node is 2090642949.</information>
<think>The question is asking the number of authors of a paper, we need to calculate the node's author neighbor degree from the graph.</think><graph>NodeDegree[2090642949, author]</graph>
<information>2</information>
<think>The number of the authors is 2.</think><answer>2</answer>

Definition of the graph: 
{GRAPH_DEFINITION['maple']}
Question: 
What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
<think>The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.</think><graph>RetrieveNode[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]</graph>
<information>The ID of this node is 2090642949.</information>
<think>The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph.</think><graph>NeighbourCheck[2090642949, venue]</graph>
<information>['1980519', '1053242']</information>
<think>The ID of the published venue are 1980519 and 1053242. We need to get their names. I will check the first one.</think><graph>NodeFeature[1980519, name]</graph>
<information>the astrophysical journal</information>
<think>Now I will check the second one.</think><graph>NodeFeature[1053242, name]</graph>
<information>the atmosphere journal</information>
<think>The name of the published venues are the astrophysical journal and the atmosphere journal.</think><answer>the astrophysical journal, the atmosphere journal</answer>
"""
    examples['maple'] = maple_ex

    # AMAZON
    amazon_ex = f"""Definition of the graph: 
{GRAPH_DEFINITION['amazon']}
Question: 
What is the price of the item Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory?
<think>The question is asking some basic information of a item node (Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory). We need to find the node in the graph.</think><graph>RetrieveNode[Purple Sequin Tiny Dancer Tutu Ballet Dance Fairy Princess Costume Accessory]</graph>
<information>The ID of this node is 0000037214.</information>
<think>The question is asking the price of a item, we need to check the node feature (price) from the graph.</think><graph>NodeFeature[0000037214, price]</graph>
<information>6.99</information>
<think>The price of the item is 6.99.</think><answer>6.99</answer>

Definition of the graph: 
{GRAPH_DEFINITION['amazon']}
Question: 
How many also_viewed_item does item The Sherlock Holmes Audio Collection have?
<think>The question is asking information of a item node (The Sherlock Holmes Audio Collection). We need to find the node in the graph.</think><graph>RetrieveNode[The Sherlock Holmes Audio Collection]</graph>
<information>The ID of this node is 0001048236.</information>
<think>The question is asking the number of also_viewed_item of an item, we need to calculate the node's also_viewed_item neighbor degree from the graph.</think><graph>NodeDegree[0001048236, also_viewed_item]</graph>
<information>9</information>
<think>The number of also_viewed_item neighbors is 9.</think><answer>9</answer>

Definition of the graph: 
{GRAPH_DEFINITION['amazon']}
Question: 
What is the brand of item Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle?
<think>The question is asking information of a item node (Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle). We need to find the node in the graph.</think><graph>RetrieveNode[Dr. Suess 19163 Dr. Seuss Puzzle 3 Pack Bundle]</graph>
<information>The ID of this node is 0000191639.</information>
<think>The question is asking the brand of a item, we need to check the node's brand neighbor from the graph.</think><graph>NeighbourCheck[0000191639, brand]</graph>
<information>['brand_5']</information>
<think>The brand of item is ['brand_5']. The ID of this brand node is brand_5. We need to check the brand node's feature (name) from the graph.</think><graph>NodeFeature[brand_5, name]</graph>
<information>Dr. Seuss.</information>
<think>The brand name of the item is Dr. Seuss.</think><answer>Dr. Seuss</answer>
"""
    examples['amazon'] = amazon_ex

    # BIOMEDICAL
    biomedical_ex = f"""Definition of the graph: 
{GRAPH_DEFINITION['biomedical']}
Question: 
What compounds can be used to treat Crohn's disease? Please answer the compound names rather than IDs.
<think>The question is related to a disease node (Crohn's disease). We need to find the node in the graph.</think><graph>RetrieveNode[Crohn's disease]</graph>
<information>The ID of this node is DOID:8778.</information>
<think>The question is asking the compounds which can be used to treat a disease, we need to check the node's 'Compound-treats-Disease' neighbor from the graph.</think><graph>NeighbourCheck[DOID:8778, Compound-treats-Disease]</graph>
<information>['DB01014', 'DB00244', 'DB00795']</information>
<think>The IDs of the compounds are 'DB01014', 'DB00244', 'DB00795'. I need to get their names. Checking the first one.</think><graph>NodeFeature[DB01014, name]</graph>
<information>Balsalazide</information>
<think>Checking the second one.</think><graph>NodeFeature[DB00244, name]</graph>
<information>Mesalazine</information>
<think>Checking the third one.</think><graph>NodeFeature[DB00795, name]</graph>
<information>Sulfasalazine</information>
<think>The name of compounds are Balsalazide, Mesalazine, Sulfasalazine.</think><answer>Balsalazide, Mesalazine, Sulfasalazine</answer>

Definition of the graph: 
{GRAPH_DEFINITION['biomedical']}
Question: 
What is the inchikey of Caffeine?
<think>The question is related to a compound node (Caffeine). We need to find the node in the graph.</think><graph>RetrieveNode[Caffeine]</graph>
<information>The ID of this node is DB00201.</information>
<think>The question is asking the inchikey feature of a node, we need to check the node's 'inchikey' feature from the graph.</think><graph>NodeFeature[DB00201, inchikey]</graph>
<information>InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N</information>
<think>The inchikey of the node is InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N.</think><answer>InChIKey=RYYVLZVUVIJVGH-UHFFFAOYSA-N</answer>

Definition of the graph: 
{GRAPH_DEFINITION['biomedical']}
Question: 
How many side effects does Caffeine have?
<think>The question is related to a compound node (Caffeine). We need to find the node in the graph.</think><graph>RetrieveNode[Caffeine]</graph>
<information>The ID of this node is DB00201.</information>
<think>The question is asking the number of side effects a compound has, we need to calculate the number of the node's 'Compound-causes-Side Effect' neighbors from the graph.</think><graph>NodeDegree[DB00201, Compound-causes-Side Effect]</graph>
<information>58</information>
<think>The number of 'Compound-causes-Side Effect' neighbors are 58.</think><answer>58</answer>
"""
    examples['biomedical'] = biomedical_ex

    # LEGAL
    legal_ex = f"""Definition of the graph: 
{GRAPH_DEFINITION['legal']}
Question: 
Who are the judges that involved in the opinion_cluster: <p>Constitutional law ...</p>?
<think>The question is related to a opinion_cluster. We need to find the node in the graph.</think><graph>RetrieveNode[<p>Constitutional law ...</p>]</graph>
<information>The ID of this node is opc-6381448.</information>
<think>The question is asking the judges of this opinion_cluster node, we need to check the node's 'judges' feature from the graph.</think><graph>NodeFeature[opc-6381448, judges]</graph>
<information>Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart</information>
<think>The judges of this opinion cluster is Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart</think><answer>Brown, Elkin, Fell, Mestrezat, Moschzisker, Potter, Stewart</answer>

Definition of the graph: 
{GRAPH_DEFINITION['legal']}
Question: 
How many opinions are contained in this opinion_cluster: <p>A general statement ...</p>?
<think>The question is related to a opinion_cluster. We need to find the node in the graph.</think><graph>RetrieveNode[<p>A general statement ...</p>]</graph>
<information>The ID of this node is opc-94592.</information>
<think>The question is asking the number of the opinion neighbors of this node, we need to calculate the node's "opinion" neighbor degree from the graph.</think><graph>NodeDegree[opc-94592, opinion]</graph>
<information>1</information>
<think>The number of the opinion neighbors is 1</think><answer>1</answer>

Definition of the graph: 
{GRAPH_DEFINITION['legal']}
Question: 
What is the docket pacer case ID for this opinion cluster: In May, 1873, a brewer ...?
<think>The question is related to a opinion cluster. We need to find the node in the graph.</think><graph>RetrieveNode[In May, 1873, a brewer ...]</graph>
<information>The ID of this node is opc-8599951.</information>
<think>The question is asking the judges of this opinion cluster node, we need to check the node's 'judges' feature from the graph.</think><graph>NodeFeature[opc-8599951, judges]</graph>
<information>Eichaedson</information>
<think>The judges of this opinion cluster is Eichaedson</think><answer>Eichaedson</answer>
"""
    examples['legal'] = legal_ex

    # GOODREADS
    goodreads_ex = f"""Definition of the graph: 
{GRAPH_DEFINITION['goodreads']}
Question: 
Who is the author of the book 'The Great Gatsby'?
<think>The question is asking for the author of a specific book (The Great Gatsby). We need to find the book node in the graph.</think><graph>RetrieveNode[The Great Gatsby]</graph>
<information>The ID of this book node is 21366268.</information>
<think>The question is asking for the author of the book. We need to check the book node's author neighbor from the graph.</think><graph>NeighbourCheck[21366268, author]</graph>
<information>['3190']</information>
<think>The ID of the author is 3190. We need to check the name.</think><graph>NodeFeature[3190, name]</graph>
<information>F. Scott Fitzgerald</information>
<think>The author of 'The Great Gatsby' is F. Scott Fitzgerald.</think><answer>F. Scott Fitzgerald</answer>

Definition of the graph: 
{GRAPH_DEFINITION['goodreads']}
Question: 
What is the genre of the book '1984'?
<think>The question is asking for the genre of a specific book (1984). We need to find the book node in the graph.</think><graph>RetrieveNode[1984]</graph>
<information>The ID of this book node is 3475269.</information>
<think>The question is asking for the genre of the book. We need to check the book node's genre feature from the graph.</think><graph>NodeFeature[3475269, genres]</graph>
<information>currently-reading, classics, fiction</information>
<think>The genres of the book '1984' are currently-reading, classics, fiction.</think><answer>currently-reading, classics, fiction</answer>

Definition of the graph: 
{GRAPH_DEFINITION['goodreads']}
Question: 
How many series included the book 'Harry Potter and the Sorcerer's Stone'?
<think>We need to find the book node for 'Harry Potter and the Sorcerer's Stone' to determine its series.</think><graph>RetrieveNode[Harry Potter and the Sorcerer's Stone]</graph>
<information>The ID of this book node is 27421523.</information>
<think>Now that we have the book node, we need to calculate the node's series neighbor degree from the graph.</think><graph>NodeDegree[27421523, series]</graph>
<information>1</information>
<think>The number of series neighbors is 1.</think><answer>1</answer>
"""
    examples['goodreads'] = goodreads_ex

    # DBLP
    dblp_ex = f"""Definition of the graph: 
{GRAPH_DEFINITION['dblp']}
Question: 
What organization is researcher Greg Daville from?
<think>The question is asking about the organization of a researcher named Greg Daville. We need to find the node corresponding to Greg Daville in the graph.</think><graph>RetrieveNode[Greg Daville]</graph>
<information>The ID of this retrieval target node is 53f460a7dabfaee4dc83702a.</information>
<think>The question is asking for the organization of a researcher, so we need to check the node's organization feature from the graph.</think><graph>NodeFeature[53f460a7dabfaee4dc83702a, organization]</graph>
<information>Hove East Sussex, United Kingdom</information>
<think>The organization of researcher Greg Daville is Hove East Sussex, United Kingdom.</think><answer>Hove East Sussex, United Kingdom</answer>

Definition of the graph: 
{GRAPH_DEFINITION['dblp']}
Question: 
How many papers are written by author Nicholas Lydon?
<think>The question is asking for the number of written papers of a specific author (Nicholas Lydon). We need to find the author node in the graph.</think><graph>RetrieveNode[Nicholas Lydon]</graph>
<information>The ID of this retrieval target node is 53f438c3dabfaedf43596117.</information>
<think>The question is asking for the number of papers written by Nicholas Lydon. We need to calculate the "paper" neighbor degree of this node.</think><graph>NodeDegree[53f438c3dabfaedf43596117, paper]</graph>
<information>2</information>
<think>The number of the paper neighbors is 2.</think><answer>2</answer>

Definition of the graph: 
{GRAPH_DEFINITION['dblp']}
Question: 
What was the publish venue of the paper Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars?
<think>The question is asking information of a node (Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars). We need to find the node in the graph.</think><graph>RetrieveNode[Mass Accretion Rates in Self-Regulated Disks of T Tauri Stars]</graph>
<information>The ID of this node is 2090642949.</information>
<think>The question is asking the published venue of a paper, we need to check the node's venue neighbor from the graph.</think><graph>NeighbourCheck[2090642949, venue]</graph>
<information>['1980519', '1053242']</information>
<think>The ID of the published venue are 1980519 and 1053242. We need to get their names. Checking first one.</think><graph>NodeFeature[1980519, name]</graph>
<information>the astrophysical journal</information>
<think>Checking second one.</think><graph>NodeFeature[1053242, name]</graph>
<information>the atmosphere journal</information>
<think>The name of the published venues are the astrophysical journal and the atmosphere journal</think><answer>the astrophysical journal, the atmosphere journal</answer>
"""
    examples['dblp'] = dblp_ex
    
    return examples

EXAMPLES = get_examples()
