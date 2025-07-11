#ifndef __TASK_HPP__
#define __TASK_HPP__

#include <pthread.h>
#include <vector>

typedef struct
{
    double* input;
    double* weights;
    double bias;
    int num_weights;
    double (*activation)(double*);
} NNArgs;

typedef struct
{
    NNArgs* args;
    pthread_t thread;
} NNTask;

class NNTaskManager
{
public:
    NNTaskManager();
    ~NNTaskManager();

    NNTask* new_task(NNArgs* args);

    double finish_task(NNTask* task);
};

void* nn_calc(void* args);

#endif