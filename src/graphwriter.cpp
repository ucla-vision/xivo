#include "graphwriter.h"


namespace xivo {


bool GraphWriter::HasNode(int node) {
  return (nodes_.count(node) && edges_.count(node));
}

bool GraphWriter::HasEdge(int node1, int node2) {
  CHECK(HasNode(node1)) << "GraphWriter: adding edge for missing node " << node1;
  CHECK(HasNode(node2)) << "GraphWriter: adding edge for missing node " << node2;

  std::unordered_set<int> node1_adj = edges_.at(node1);
  std::unordered_set<int> node2_adj = edges_.at(node2);
  bool in_node1_list = node1_adj.count(node2);
  bool in_node2_list = node2_adj.count(node1);
  return (in_node1_list || in_node2_list);
}

void GraphWriter::AddNode(int node) {
  if (!HasNode(node)) {
    nodes_.insert(node);
    edges_.insert({node, {}});
  }
}

void GraphWriter::AddEdge(int node1, int node2) {
  if (!HasEdge(node1, node2) && (node1 != node2)) {
    edges_.at(node1).insert(node2);
  }
}

void GraphWriter::CollectGraph(GraphBase* graphobj) {

  std::vector<GroupPtr> groups = graphobj->GetGroups();

  // Add all groups
  for (auto g: groups) {
    AddNode(g->id());
  }

  // Add all edges
  for (auto g1: groups) {
    std::vector<FeaturePtr> visible_features = graphobj->GetFeaturesOf(g1);
    for (auto f: visible_features) {
      std::vector<GroupPtr> other_groups = graphobj->GetGroupsOf(f);
      for (auto g2: other_groups) {
        AddEdge(g1->id(), g2->id());
      }
    }
  }
}

void GraphWriter::WriteDot(std::string filename) {

  std::ofstream output;
  output.open(filename);

  output << "graph D { " << "\n" << "\n";

  for (auto e: edges_) {
    int node1 = e.first;
    for (auto node2: e.second) {
      output << "\t" << node1 << " -- " << node2 << ";\n";
    }
  }

  output << "}";
  output.close();
}


}