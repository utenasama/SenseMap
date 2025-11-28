#define TEST_NAME "util/threading"

#define BOOST_TEST_DYN_LINK  // this is optional
// #define BOOST_TEST_MODULE ThreadTest   // specify the name of your test module

#include "util/threading.h"
#include "util/logging.h"
#include "util/testing.h"

using namespace sensemap;

BOOST_AUTO_TEST_CASE(TestThreadWait) {
    class TestThread : public Thread {
        void Run() { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
    };

    TestThread thread;
    BOOST_CHECK(!thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Start();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadPause) {
    class TestThread : public Thread {
        void Run() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            BlockIfPaused();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    };

    TestThread thread;
    BOOST_CHECK(!thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Start();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Resume();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadStop) {
    class TestThread : public Thread {
        void Run() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (IsStopped()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return;
            }
        }
    };

    TestThread thread;
    BOOST_CHECK(!thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Start();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadPauseStop) {
    class TestThread : public Thread {
        void Run() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            BlockIfPaused();
            if (IsStopped()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                return;
            }
        }
    };

    TestThread thread;
    BOOST_CHECK(!thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Start();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(thread.IsFinished());
}

BOOST_AUTO_TEST_CASE(TestThreadRestart) {
    class TestThread : public Thread {
        void Run() { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
    };

    TestThread thread;
    BOOST_CHECK(!thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    for (size_t i = 0; i < 2; ++i) {
        thread.Start();
        BOOST_CHECK(thread.IsStarted());
        BOOST_CHECK(!thread.IsStopped());
        BOOST_CHECK(!thread.IsPaused());
        BOOST_CHECK(thread.IsRunning());
        BOOST_CHECK(!thread.IsFinished());

        thread.Wait();
        BOOST_CHECK(thread.IsStarted());
        BOOST_CHECK(!thread.IsStopped());
        BOOST_CHECK(!thread.IsPaused());
        BOOST_CHECK(!thread.IsRunning());
        BOOST_CHECK(thread.IsFinished());
    }
}

BOOST_AUTO_TEST_CASE(TestThreadValidSetup) {
    class TestThread : public Thread {
        void Run() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            SignalValidSetup();
        }
    };

    TestThread thread;
    BOOST_CHECK(!thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Start();

    BOOST_CHECK(thread.CheckValidSetup());
    BOOST_CHECK(thread.CheckValidSetup());

    thread.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(thread.IsFinished());
    BOOST_CHECK(thread.CheckValidSetup());
}

BOOST_AUTO_TEST_CASE(TestThreadInvalidSetup) {
    class TestThread : public Thread {
        void Run() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            SignalInvalidSetup();
        }
    };

    TestThread thread;
    BOOST_CHECK(!thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(!thread.IsFinished());

    thread.Start();

    BOOST_CHECK(!thread.CheckValidSetup());
    BOOST_CHECK(!thread.CheckValidSetup());

    thread.Wait();
    BOOST_CHECK(thread.IsStarted());
    BOOST_CHECK(!thread.IsStopped());
    BOOST_CHECK(!thread.IsPaused());
    BOOST_CHECK(!thread.IsRunning());
    BOOST_CHECK(thread.IsFinished());
    BOOST_CHECK(!thread.CheckValidSetup());
}

BOOST_AUTO_TEST_CASE(TestCallback) {
    class TestThread : public Thread {
    public:
        enum Callbacks {
            CALLBACK1,
            CALLBACK2,
        };

        TestThread() {
            RegisterCallback(CALLBACK1);
            RegisterCallback(CALLBACK2);
        }

    private:
        void Run() {
            Callback(CALLBACK1);
            Callback(CALLBACK2);
        }
    };

    bool called_back1 = false;
    std::function<void()> CallbackFunc1 = [&called_back1]() { called_back1 = true; };

    bool called_back2 = false;
    std::function<void()> CallbackFunc2 = [&called_back2]() { called_back2 = true; };

    bool called_back3 = false;
    std::function<void()> CallbackFunc3 = [&called_back3]() { called_back3 = true; };

    TestThread thread;
    thread.AddCallback(TestThread::CALLBACK1, CallbackFunc1);
    thread.Start();
    thread.Wait();
    BOOST_CHECK(called_back1);
    BOOST_CHECK(!called_back2);
    BOOST_CHECK(!called_back3);

    called_back1 = false;
    called_back2 = false;
    thread.AddCallback(TestThread::CALLBACK2, CallbackFunc2);
    thread.Start();
    thread.Wait();
    BOOST_CHECK(called_back1);
    BOOST_CHECK(called_back2);
    BOOST_CHECK(!called_back3);

    called_back1 = false;
    called_back2 = false;
    called_back3 = false;
    thread.AddCallback(TestThread::CALLBACK1, CallbackFunc3);
    thread.Start();
    thread.Wait();
    BOOST_CHECK(called_back1);
    BOOST_CHECK(called_back2);
    BOOST_CHECK(called_back3);
}

BOOST_AUTO_TEST_CASE(TestDefaultCallback) {
    class TestThread : public Thread {
    private:
        void Run() { std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
    };

    bool called_back1 = false;
    std::function<void()> CallbackFunc1 = [&called_back1]() { called_back1 = true; };

    bool called_back2 = false;
    std::function<void()> CallbackFunc2 = [&called_back2]() { called_back2 = true; };

    TestThread thread;
    thread.AddCallback(TestThread::STARTED_CALLBACK, CallbackFunc1);
    thread.AddCallback(TestThread::FINISHED_CALLBACK, CallbackFunc2);
    thread.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    BOOST_CHECK(called_back1);
    BOOST_CHECK(!called_back2);
    thread.Wait();
    BOOST_CHECK(called_back1);
    BOOST_CHECK(called_back2);
}

BOOST_AUTO_TEST_CASE(TestThreadTimer) {
    class TestThread : public Thread {
        void Run() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            BlockIfPaused();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    };

    TestThread thread;
    thread.Start();
    thread.Wait();
    const auto elapsed_seconds1 = thread.GetTimer().ElapsedSeconds();
    BOOST_CHECK_GT(elapsed_seconds1, 0.35);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    BOOST_CHECK_EQUAL(thread.GetTimer().ElapsedSeconds(), elapsed_seconds1);

    thread.Start();
    BOOST_CHECK_LT(thread.GetTimer().ElapsedSeconds(), elapsed_seconds1);

    thread.Pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const auto elapsed_seconds2 = thread.GetTimer().ElapsedSeconds();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    BOOST_CHECK_EQUAL(thread.GetTimer().ElapsedSeconds(), elapsed_seconds2);

    thread.Resume();
    thread.Wait();
    BOOST_CHECK_GT(thread.GetTimer().ElapsedSeconds(), elapsed_seconds2);
    BOOST_CHECK_GT(thread.GetTimer().ElapsedSeconds(), 0.35);
}

BOOST_AUTO_TEST_CASE(TestJobQueueSingleProducerSingleConsumer) {
    JobQueue<int> job_queue;

    std::thread producer_thread([&job_queue]() {
        for (int i = 0; i < 10; ++i) {
            CHECK(job_queue.Push(i));
        }
    });

    std::thread consumer_thread([&job_queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK_EQ(job_queue.Size(), 10);
        for (int i = 0; i < 10; ++i) {
            const auto job = job_queue.Pop();
            CHECK(job.IsValid());
            CHECK_EQ(job.Data(), i);
        }
    });

    producer_thread.join();
    consumer_thread.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueSingleProducerSingleConsumerMaxNumJobs) {
    JobQueue<int> job_queue(2);

    std::thread producer_thread([&job_queue]() {
        for (int i = 0; i < 10; ++i) {
            CHECK(job_queue.Push(i));
        }
    });

    std::thread consumer_thread([&job_queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK_EQ(job_queue.Size(), 2);
        for (int i = 0; i < 10; ++i) {
            const auto job = job_queue.Pop();
            CHECK(job.IsValid());
            CHECK_EQ(job.Data(), i);
        }
    });

    producer_thread.join();
    consumer_thread.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueMultipleProducerSingleConsumer) {
    JobQueue<int> job_queue(1);

    std::thread producer_thread1([&job_queue]() {
        for (int i = 0; i < 10; ++i) {
            CHECK(job_queue.Push(i));
        }
    });

    std::thread producer_thread2([&job_queue]() {
        for (int i = 0; i < 10; ++i) {
            CHECK(job_queue.Push(i));
        }
    });

    std::thread consumer_thread([&job_queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK_EQ(job_queue.Size(), 1);
        for (int i = 0; i < 20; ++i) {
            const auto job = job_queue.Pop();
            CHECK(job.IsValid());
            CHECK_LT(job.Data(), 10);
        }
    });

    producer_thread1.join();
    producer_thread2.join();
    consumer_thread.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueSingleProducerMultipleConsumer) {
    JobQueue<int> job_queue(1);

    std::thread producer_thread([&job_queue]() {
        for (int i = 0; i < 20; ++i) {
            CHECK(job_queue.Push(i));
        }
    });

    std::thread consumer_thread1([&job_queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK_LE(job_queue.Size(), 1);
        for (int i = 0; i < 10; ++i) {
            const auto job = job_queue.Pop();
            CHECK(job.IsValid());
            CHECK_LT(job.Data(), 20);
        }
    });

    std::thread consumer_thread2([&job_queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK_LE(job_queue.Size(), 1);
        for (int i = 0; i < 10; ++i) {
            const auto job = job_queue.Pop();
            CHECK(job.IsValid());
            CHECK_LT(job.Data(), 20);
        }
    });

    producer_thread.join();
    consumer_thread1.join();
    consumer_thread2.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueMultipleProducerMultipleConsumer) {
    JobQueue<int> job_queue(1);

    std::thread producer_thread1([&job_queue]() {
        for (int i = 0; i < 10; ++i) {
            CHECK(job_queue.Push(i));
        }
    });

    std::thread producer_thread2([&job_queue]() {
        for (int i = 0; i < 10; ++i) {
            CHECK(job_queue.Push(i));
        }
    });

    std::thread consumer_thread1([&job_queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK_LE(job_queue.Size(), 1);
        for (int i = 0; i < 10; ++i) {
            const auto job = job_queue.Pop();
            CHECK(job.IsValid());
            CHECK_LT(job.Data(), 10);
        }
    });

    std::thread consumer_thread2([&job_queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK_LE(job_queue.Size(), 1);
        for (int i = 0; i < 10; ++i) {
            const auto job = job_queue.Pop();
            CHECK(job.IsValid());
            CHECK_LT(job.Data(), 10);
        }
    });

    producer_thread1.join();
    producer_thread2.join();
    consumer_thread1.join();
    consumer_thread2.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueWait) {
    JobQueue<int> job_queue;

    std::thread producer_thread([&job_queue]() {
        for (int i = 0; i < 10; ++i) {
            CHECK(job_queue.Push(i));
        }
    });

    std::thread consumer_thread([&job_queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK_EQ(job_queue.Size(), 10);
        for (int i = 0; i < 10; ++i) {
            const auto job = job_queue.Pop();
            CHECK(job.IsValid());
            CHECK_EQ(job.Data(), i);
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    job_queue.Wait();

    BOOST_CHECK_EQUAL(job_queue.Size(), 0);
    BOOST_CHECK(job_queue.Push(0));
    BOOST_CHECK(job_queue.Pop().IsValid());

    producer_thread.join();
    consumer_thread.join();
}

BOOST_AUTO_TEST_CASE(TestJobQueueStopProducer) {
    JobQueue<int> job_queue(1);

    std::thread producer_thread([&job_queue]() {
        CHECK(job_queue.Push(0));
        CHECK(!job_queue.Push(0));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    BOOST_CHECK_EQUAL(job_queue.Size(), 1);

    job_queue.Stop();
    producer_thread.join();

    BOOST_CHECK(!job_queue.Push(0));
    BOOST_CHECK(!job_queue.Pop().IsValid());
}

BOOST_AUTO_TEST_CASE(TestJobQueueStopConsumer) {
    JobQueue<int> job_queue(1);

    BOOST_CHECK(job_queue.Push(0));

    std::thread consumer_thread([&job_queue]() {
        const auto job = job_queue.Pop();
        CHECK(job.IsValid());
        CHECK_EQ(job.Data(), 0);
        CHECK(!job_queue.Pop().IsValid());
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    BOOST_CHECK_EQUAL(job_queue.Size(), 0);

    job_queue.Stop();
    consumer_thread.join();

    BOOST_CHECK(!job_queue.Push(0));
    BOOST_CHECK(!job_queue.Pop().IsValid());
}

BOOST_AUTO_TEST_CASE(TestJobQueueClear) {
    JobQueue<int> job_queue(1);

    BOOST_CHECK(job_queue.Push(0));
    BOOST_CHECK_EQUAL(job_queue.Size(), 1);

    job_queue.Clear();
    BOOST_CHECK_EQUAL(job_queue.Size(), 0);
}
