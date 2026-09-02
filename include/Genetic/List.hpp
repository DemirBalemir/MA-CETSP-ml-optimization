/**
 * List.hpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#ifndef CETSP_LIST_HPP
#define CETSP_LIST_HPP

#include "Node.hpp"
#include <vector>
#include <iostream>
#include <numeric>
#include <climits>

class List {
public:
    int _size;
    Node* _head;
    double value;
    double distance;    // distance to population
    double fitness;     // value and min distance
    int birth_iter = -1;
    int death_iter = -1;
    int instance_index = -1;
    double post_vnd_fitness_at_birth = -1;
    double final_fitness = -1;
    std::vector<std::pair<double, double>> pre_vnd_coords;
    bool was_inserted = false;
    bool censored = false;
    bool already_logged = false;  // true after censored log at training time — skip death log
    double pre_vnd_value = -1;
    double post_vnd_value = -1;
    double post_greed_value = -1;  // cost after 1st greedy, BEFORE LKH (cheap-stage probe)
    double post_lkh_value = -1;    // cost after LKH (expensive stage)
    // ---- LINEAGE (provenance) probe ----------------------------------------
    // Everything below is known BEFORE VND runs, so it is a legitimate filter
    // feature. Parents are population members that have already been through
    // VND, so their cost and fitness are meaningful; the two edit distances are
    // already computed in Population::nextPopulation and were previously
    // discarded. This is the one feature family the geometry set cannot express.
    double parent1_value = -1;     // post-VND cost of parent A
    double parent2_value = -1;     // post-VND cost of parent B
    double parent1_fitness = -1;   // value-rank + beta*distance-rank of parent A
    double parent2_fitness = -1;   // ditto, parent B
    double parent1_dist = -1;      // edit distance offspring <-> parent A (pre-VND)
    double parent2_dist = -1;      // edit distance offspring <-> parent B (pre-VND)
    int    mutated = 0;            // 1 if the patience-driven mutation fired
    double cox_lp = 0.0;
    bool has_cox_lp = false;



public:
    List();
    List(const List& s);
    ~List();
    List& operator=(List& s);
    void add(Node* node);
    void add(Node* node, Node* pos);
    void remove(Node *pos);
    void reverse();
    void reverse(Node* begin, Node* end);
    Node* head();
    int size();
    void print();
    double getValue();
    double getDistance();
    double getFitness();
    void setValue(double value);
    void setSize(int size);
    void setHead(Node* p);
    void setDistance(double distance);
    void setFitness(double fitness);
    void evaluate();
};

#endif // CETSP_LIST_HPP
