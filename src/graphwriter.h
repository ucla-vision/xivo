#pragma once

#include <string>
#include <unordered_set>
#include <unordered_map>

#include "feature.h"
#include "group.h"
#include "core.h"
#include "graphbase.h"


namespace xivo {

/** Writes the `Graph` and the `Mapper` into a `.dot` file that can be
 * visualized. */
class GraphWriter {

public:
  GraphWriter() {};

  void CollectGraph(GraphBase* graphobj);

  void WriteDot(std::string filename);

private:
  void AddNode(int node);
  void AddEdge(int node1, int node2);
  bool HasNode(int node);
  bool HasEdge(int node1, int node2);
  std::unordered_set<int> nodes_;
  std::unordered_map<int, std::unordered_set<int>> edges_;

};


}