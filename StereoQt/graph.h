#ifndef GRAPH_H
#define GRAPH_H

#include <string.h>
#include <assert.h>
#include <vector>
#include <queue>
#include <limits>

/// captype: type of edge capacities (excluding t-links)
/// tcaptype: type of t-links (edges between nodes and terminals)
/// flowtype: type of total flow
template <typename captype, typename tcaptype, typename flowtype> class Graph
{
public:
	typedef enum { SOURCE = 0, SINK = 1 } termtype;
	typedef int node_id;
	typedef int arc_id;

	Graph(int hintNbNodes = 0, int hintNbArcs = 0);
	virtual ~Graph();

	node_id add_node();
	void add_edge(node_id i, node_id j, captype capij, captype capji);
	void add_edge_infty(node_id i, node_id j);
	void add_tweights(node_id i, tcaptype capS, tcaptype capT);

	flowtype maxflow();
	termtype what_segment(node_id i, termtype defaultSegm = SOURCE) const;

private:
	struct node;
	struct arc;

	struct node {
		arc_id first;	///< �Ӹõ�����ĵ�һ����
		arc* parent;	///< initial path to root (a terminal node) if in tree
		node* next;		///< ָ����һ��active node��ָ�루��������һ��node��ָ���Լ���
		int ts;			///< timestamp showing when DIST was computed
		int dist;		///< distance to the terminal
		termtype term;	///< source or sink (only if parent!=NULL)
		tcaptype cap;	///< capacity of arc source->node(>0) or node->sink(<0)
	};

	struct arc {
		node_id head;	///< �ñ�ָ���node
		arc_id next;	///< ������ͬһnode����һ����
		arc_id sister;	///< reverse arc
		captype cap;	///< residual capacity
	};

	std::vector<node> nodes;
	std::vector<arc> arcs;

	flowtype flow;		///< total flow
	node *activeBegin, *activeEnd;		///< list of active nodes
	std::queue<node*>	orphans;
	int time;			///< monotonically increasing global counter

						// special constants for node.parent
	arc* TERMINAL;	///< ָ��terminal�ı�
	arc* ORPHAN;	///< ָ��orpan�ı�

					// procesing active list
	void set_active(node* i);
	node* next_active();

	// processing orphans
	void set_orphan(node* i);
	void process_orphan(node* i);
	void adopt_orphans();

	void maxflow_init();
	int dist_to_root(node* j);
	arc* grow_tree(node* i);
	captype find_bottleneck(arc* midarc);
	void push_flow(arc* midarc, captype f);
	void augment(arc* middle_arc);
};

// Necessary for templates: provide full implementation
#include "graph.cpp"
#include "maxflow.cpp"


#endif
