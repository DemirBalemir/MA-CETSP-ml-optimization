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
    double pre_vnd_value = -1;
    double post_vnd_value = -1;
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
