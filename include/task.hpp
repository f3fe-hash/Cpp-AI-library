#ifndef __TASK_HPP__
#define __TASK_HPP__

#include <thread>
#include <vector>
#include <iostream>

typedef struct
{
    double* input;
    double* weights;
    double bias;
    double* output;
    int num_weights;
    double (*activation)(double*);
} NNArgs;

typedef struct
{
    NNArgs* args;
    std::thread thread;
} NNTask;

class NNTaskManager
{
public:
    NNTaskManager();
    ~NNTaskManager();

    NNTask* new_task(NNArgs* args);

    double finish_task(NNTask* task);
};

void nn_calc(NNArgs* args);

#endif