#ifdef GRAPH_H
// #include "graph.h"

template<typename captype, typename tcaptype, typename flowtype>
Graph<captype, tcaptype, flowtype>::Graph(int hintNbNodes, int hintNbArcs)
	:nodes(), arcs(), flow(0), activeBegin(0), activeEnd(0), orphans(), time(0),
	TERMINAL(0), ORPHAN(0)
{
	nodes.reserve(hintNbNodes);
	arcs.reserve(hintNbArcs);
}

template<typename captype, typename tcaptype, typename flowtype>
Graph<captype, tcaptype, flowtype>::~Graph()
{}

/// Add node the graph, return the node id
template<typename captype, typename tcaptype, typename flowtype>
typename Graph<captype, tcaptype, flowtype>::node_id
Graph<captype, tcaptype, flowtype>::add_node()
{
	node n = { -1, 0, 0, 0, 0, SOURCE, 0 };
	node_id i = static_cast<node_id>(nodes.size());
	nodes.push_back(n);
	return i;
}

/// Add two edges between 'i' and 'j' with weights 'capij' and 'capji'
template<typename captype, typename tcaptype, typename flowtype>
void Graph<captype, tcaptype, flowtype>::add_edge(node_id i, node_id j, captype capij, captype capji)
{
	assert(0 <= i && i < (int)nodes.size());
	assert(0 <= j && j < (int)nodes.size());
	assert(i != j);
	assert(capij >= 0);
	assert(capji >= 0);

	arc_id ij = static_cast<arc_id>(arcs.size()), ji = ij + 1;

	arc aij = { j, nodes[i].first, ji, capij };  // head next sister, cap
	arc aji = { i, nodes[j].first, ij, capji };	

	nodes[i].first = ij;
	nodes[j].first = ji;
	arcs.push_back(aij);
	arcs.push_back(aji);
}

/// Add edge with infinite capacity from node 'i' to 'j'
template<typename captype, typename tcaptype, typename flowtype>
void Graph<captype, tcaptype, flowtype>::add_edge_infty(node_id i, node_id j)
{
	add_edge(i, j, std::numeric_limits<captype>::max(), 0);
}

/// Add new edges 'source(s)->i' and 'i->sink(t)' with corresponding weights.
/// weights can be negative
/// ���capacity
template<typename captype, typename tcaptype, typename flowtype>
void Graph<captype, tcaptype, flowtype>::add_tweights(node_id i, tcaptype capS, tcaptype capT)
{
	assert(0 <= i && i < (int)nodes.size());
	tcaptype delta = nodes[i].cap;	// residual
	if (delta > 0) capS += delta;	// delta>0�������source->nodei��capcity
	else		   capT -= delta;

	flow += (capS < capT) ? capS : capT;	// flow����capS capT��С���Ǹ�
	nodes[i].cap = capS - capT;
}

/// After the maxflow is computed, this function returns to which segment the
/// node 'i' belongs (SOURCE or SINK).
template<typename captype, typename tcaptype, typename flowtype>
type Graph<captype, tcaptype, flowtype>::termtype
Graph<captype, tcaptype, flowtype>::what_segment(node_id i, termtype def) const
{
	return (nodes[i].parent ? nodes[i].term : def);
}

#endif