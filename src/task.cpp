#include "task.hpp"

NNTaskManager::NNTaskManager()
{}

NNTaskManager::~NNTaskManager()
{}

NNTask* NNTaskManager::new_task(NNArgs* args)
{
    NNTask* task = new NNTask();
    pthread_create(&task->thread, NULL, nn_calc, (void *)args);

    return task;
}

double NNTaskManager::finish_task(NNTask* task)
{
    void* output;
    pthread_join(task->thread, &output);
    return *(double *)output;
}

void* nn_calc(void* args)
{
    NNArgs* nnargs = (NNArgs *)args;

    double sum = nnargs->bias;
    for (int i = 0; i < nnargs->num_weights; i++)
        sum += nnargs->weights[i] * nnargs->input[i];

    double out = nnargs->activation(sum);
    return (void *)&out;
}