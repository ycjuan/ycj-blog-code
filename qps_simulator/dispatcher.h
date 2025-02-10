#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <map>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <iostream>

#include "retriever.h"

using namespace std;

struct Packet // A data structure to hold a request and its response
{
    Request request;
    Response response;
    bool ready = false;
};

class Dispatcher
{
public:
    Dispatcher(const Retriever &retriever) : retriever_(retriever) {}

    Response retrieve(const Request &request)
    {
        Packet packet;
        packet.request = request;
        
        // --------------------
        // put the request into the queue
        string groupName = request.groupName;
        {
            // obtain the queue lock
            lock_guard<mutex> lock(mtx_queue_);
            // if the group name is unseen, create a new queue
            if (requestQueueMap_.find(groupName) == requestQueueMap_.end())
            {
                requestQueueMap_[groupName] = queue<Packet *>();
            }
            // push the packet into the queue
            requestQueueMap_.at(groupName).push(&packet);
        }

        // --------------------
        // wait for the response
        shared_lock<shared_mutex> lock(mtx_request_);
        cv_.wait(lock, [&packet]
                 { return packet.ready; });

        return packet.response;
    }

    void processRequests()
    {
        while (true)
        {
            // --------------------
            // check if the dispatcher should stop
            if (shouldStop_)
            {
                break;
            }

            // --------------------
            // get a list of packets from the queue
            vector<Packet *> p_packets;
            {
                //this_thread::sleep_for(chrono::microseconds(10)); // tried this, but seems not necessary

                // obtain the queue lock
                lock_guard<mutex> lock(mtx_queue_);

                // find which queue to process
                queue<Packet *> *p_queue = findQueue();

                // if no queue is found, continue
                if (p_queue == nullptr)
                {
                    continue;
                }

                // get packets from the queue
                while (!p_queue->empty())
                {
                    p_packets.push_back(p_queue->front());
                    p_queue->pop();
                    if (p_packets.size() >= retriever_.getMaxBatchSize())
                    {
                        break;
                    }
                }
            }

            // --------------------
            // retrieve the responses
            int batchSize = p_packets.size();
            vector<Request> requests;
            for (Packet *p_packet : p_packets)
            {
                requests.push_back(p_packet->request);
            }
            Response response = retriever_.retrieve(requests);

            // --------------------
            // update the reponses
            auto now = chrono::high_resolution_clock::now();
            for (Packet *p_packet : p_packets)
            {
                p_packet->response.latencyMs = chrono::duration<float, milli>(now - p_packet->request.creationTimepoint).count();
                p_packet->response.batchSize = batchSize;
                p_packet->ready = true;
            }

            // --------------------
            // notify the waiting threads
            cv_.notify_all();
        }
    }

    void stop() { shouldStop_ = true; }

private:

    const Retriever &retriever_;

    // ----------------------
    // each group has it's own queue
    map<string, queue<Packet *>> requestQueueMap_;

    // ----------------------
    // synchronization variables
    condition_variable_any cv_;
    mutex mtx_queue_;
    shared_mutex mtx_request_;
    bool shouldStop_ = false;

    // Note: the logic below presents a very simple "greedy" approach to find the queue with the smallest timeout.
    // A more sophisticated approach could be implemented to get better QPS
    queue<Packet *> *findQueue()
    {
        queue<Packet *> *p_minQueue = nullptr;
        for (auto &pair : requestQueueMap_)
        {
            queue<Packet *> &queue = pair.second;
            if (queue.empty())
            {
                // if queue is empty, skip it
                continue;
            }
            else if (p_minQueue == nullptr)
            {
                // if this is the first non-empty queue, set it as the min queue
                p_minQueue = &queue;
            }
            else if (p_minQueue->front()->request.timeoutTimepoint > queue.front()->request.timeoutTimepoint)
            {
                // if this queue has a smaller timeout, set it as the min queue
                p_minQueue = &queue;
            }
        }
        return p_minQueue;
    }
};

#endif