
GRAPH_DEFINITION = {
    'maple': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.\nPaper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.',
    'biomedical': 'There are eleven types of nodes in the graph: Anatomy, Biological Process, Cellular Component, Compound, Disease, Gene, Molecular Function, Pathway, Pharmacologic Class, Side Effect, Symptom.\nEach node has name feature.\nThere are these types of edges: Anatomy-downregulates-Gene, Anatomy-expresses-Gene, Anatomy-upregulates-Gene, Compound-binds-Gene, Compound-causes-Side Effect, Compound-downregulates-Gene, Compound-palliates-Disease, Compound-resembles-Compound, Compound-treats-Disease, Compound-upregulates-Gene, Disease-associates-Gene, Disease-downregulates-Gene, Disease-localizes-Anatomy, Disease-presents-Symptom, Disease-resembles-Disease, Disease-upregulates-Gene, Gene-covaries-Gene, Gene-interacts-Gene, Gene-participates-Biological Process, Gene-participates-Cellular Component, Gene-participates-Molecular Function, Gene-participates-Pathway, Gene-regulates-Gene, Pharmacologic Class-includes-Compound.',
    'legal': 'There are four types of nodes in the graph: opinion, opinion_cluster, docket, and court.\nOpinion nodes have features: plain_text. Opinion_cluster nodes have features: syllabus, judges, case_name, attorneys. Docket nodes have features: pacer_case_id, case_name. Court nodes have features: full_name, start_date, end_date, citation_string.\nOpinion nodes are linked to their reference nodes and cited_by nodes, as well as their opinion_cluster nodes. Opinion_cluster nodes are linked to opinion nodes and docket nodes. Docket nodes are linked to opinion_cluster nodes and court nodes. Court nodes are linked to docket nodes.',
    'amazon': 'There are two types of nodes in the graph: item and brand.\nItem nodes have features: title, description, price, img, category. Brand nodes have features: name.\nItem nodes are linked to their brand nodes, also_viewed_item nodes, buy_after_viewing_item nodes, also_bought_item nodes, bought_together_item nodes. Brand nodes are linked to their item nodes.',
    'goodreads': 'There are four types of nodes in the graph: book, author, publisher, and series.\nBook nodes have features: country_code, language_code, is_ebook, title, description, format, num_pages, publication_year, url, popular_shelves, and genres. Author nodes have features: name. Publisher nodes have features: name. Series nodes have features: title and description.\nBook nodes are linked to their author nodes, publisher nodes, series nodes and similar_books nodes. Author nodes are linked to their book nodes. Publisher nodes are linked to their book nodes. Series nodes are linked to their book nodes.',
    'dblp': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, keywords, lang, and year. Author nodes have features: name and organization. Venue nodes have features: name.\nPaper nodes are linked to their author nodes, venue nodes, reference nodes (the papers this paper cite) and cited_by nodes (other papers which cite this paper). Author nodes are linked to their paper nodes. Venue nodes are linked to their paper nodes.',
    'maple/Biology': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.\nPaper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.',
    'maple/Chemistry': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.\nPaper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.',
    'maple/Materials_Science': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.\nPaper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.',
    'maple/Medicine': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.\nPaper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.',
    'maple/Physics': 'There are three types of nodes in the graph: paper, author and venue.\nPaper nodes have features: title, abstract, year and label. Author nodes have features: name. Venue nodes have features: name.\nPaper nodes are linked to author nodes, venue nodes, reference nodes and cited_by nodes. Author nodes are linked to paper nodes. Venue nodes are linked to paper nodes.'
}

NODE_TEXT_KEYS = {
    'maple': {'paper': ['title'], 'author': ['name'], 'venue': ['name']},
    'amazon': {'item': ['title'], 'brand': ['name']},
    'biomedical': {'Anatomy': ['name'], 'Biological_Process':['name'], 'Cellular_Component':['name'], 'Compound':['name'], 'Disease':['name'], 'Gene':['name'], 'Molecular_Function':['name'], 'Pathway':['name'], 'Pharmacologic_Class':['name'], 'Side_Effect':['name'], 'Symptom':['name']},
    'legal': {'opinion': ['plain_text'], 'opinion_cluster': ['syllabus'], 'docket': ['pacer_case_id', 'case_name'], 'court': ['full_name']},
    'goodreads': {'book': ['title'], 'author': ['name'], 'publisher': ['name'], 'series': ['title']},
    'dblp': {'paper': ['title'], 'author': ['name', 'organization'], 'venue': ['name']},
    'maple/Biology': {'paper': ['title'], 'author': ['name'], 'venue': ['name']},
    'maple/Chemistry': {'paper': ['title'], 'author': ['name'], 'venue': ['name']},
    'maple/Materials_Science': {'paper': ['title'], 'author': ['name'], 'venue': ['name']},
    'maple/Medicine': {'paper': ['title'], 'author': ['name'], 'venue': ['name']},
    'maple/Physics': {'paper': ['title'], 'author': ['name'], 'venue': ['name']}
}

GraphAgent_INSTRUCTION = """Solve a question answering task by repeating bundled steps that contain reasoning (<think>...</think>) followed by exactly one graph interaction (<graph>...</graph>). After each <graph> call, the environment returns feedback inside <information>...</information>. You may take as many steps as necessary.
Output protocol (bundled steps):
- Intermediate step (must include BOTH in a single output, in this order):
  <think>...</think><graph>Function[...]</graph>
  (Then the environment responds with <information>...</information>.)
- Final step (no more graph calls):
  <think>...</think><answer>...</answer>

Rules:
1) You MUST conduct reasoning inside <think>...</think> before every graph call and after every <information> you receive.
2) Inside <graph>...</graph>, issue EXACTLY ONE function from the list below per step. Do NOT include any other text in <graph>.
3) Do NOT fabricate <information>; it is ONLY produced by the environment immediately after your <graph> step.
4) Keep thoughts concise and ONLY inside <think>. Do NOT put a graph call inside <think>, and do NOT put thoughts inside <graph>.
5) The final output MUST contain ONLY one <answer>...</answer> block with the requested node main features (e.g., names), not node IDs.

Available graph functions (use EXACT names/signatures INSIDE <graph>...):
- RetrieveNode[keyword]              # retrieves the related node from the graph according to the query
- NodeFeature[Node, feature]          # returns detailed attribute information of Node for the given "feature" key
- NodeDegree[Node, neighbor_type]     # returns the number of "neighbor_type" neighbors of Node
- NeighbourCheck[Node, neighbor_type] # lists the "neighbor_type" neighbors of Node and returns them

Here are some examples (each intermediate step bundles <think>+<graph>, and the environment replies with <information>; the last step uses <think>+<answer>):
{examples}
(END OF EXAMPLES)

Definition of the graph:
{graph_definition}
Question:
{question}
{scratchpad}

"""

