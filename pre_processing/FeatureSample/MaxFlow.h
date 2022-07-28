#include <vector>
using std::vector;

void MaxFlow(int n_vert, const vector<std::pair<int, int>> &edge, const vector<double> &cap, int s, int t, vector<double>& output_flow);
void MinCutfromMaxFlow(int n_vert, const vector<std::pair<int, int>> &edge, const vector<double> &cap, int s, int t, vector<int> &part1, vector<int> &part2);
void MinCut(int n_vert, const vector<std::pair<int, int>> &edge, const vector<double> &cap, vector<int> &part1, vector<int> &part2);