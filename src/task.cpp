#include "task.hpp"

NNTaskManager::NNTaskManager()
{}

NNTaskManager::~NNTaskManager()
{}

NNTask* NNTaskManager::new_task(NNArgs* args)
{
    NNTask* task = new NNTask();
    task->args = args;
    task->thread = std::thread(nn_calc, task->args);

    return task;
}

double NNTaskManager::finish_task(NNTask* task)
{
    if (task->thread.joinable()) task->thread.join();

    double output = *task->args->output;

    return output;
}

void nn_calc(NNArgs* args)
{
    double sum = args->bias;
    for (int i = 0; i < args->num_weights; i++)
        sum +=  args->weights[i] * args->input[i];

    args->output = new double;
    *args->output = args->activation(&sum);
}