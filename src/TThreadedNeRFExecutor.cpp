#include "TThreadedNeRFExecutor.h"

TThreadedNeRFExecutor :: TThreadedNeRFExecutor()
{
	Initialize();
}

TThreadedNeRFExecutor :: ~TThreadedNeRFExecutor()
{
	Finalize();
}

void TThreadedNeRFExecutor :: SetExecutor(std::unique_ptr<NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall>> &executor)
{
	//if (Executor != nullptr)
	//	throw std::runtime_error("TNeRFRenderWidget :: SetExecutor not intended to be called twice");
	
	//Cтопит поток, дожидается остановки, меняет экзекьютор потом снова запускает	
	Finalize();
	Executor = std::move(executor);
	Start();
}

void TThreadedNeRFExecutor :: Initialize()
{
	Stopped = false;
	IsFinished = true;
	this->moveToThread(&Thread);
	//Timer.moveToThread(&Thread);

	// Соединяем сигнал started потока, со слотом process "рабочего" класса, т.е. начинается выполнение нужной работы.
	connect(&Thread, SIGNAL(started()), this, SLOT(Process()));
	//connect(&Timer, SIGNAL(timeout()), this, SLOT(OnTimerTimeout()));
}

void TThreadedNeRFExecutor :: Finalize()
{
	Stop();
	//Дождаться окончания рендеринга
	while (!IsFinished)
	{
		QThread::msleep(10);
	};
}
	
void TThreadedNeRFExecutor :: Start()		//Запуск потока
{
	Stopped = false;
	IsFinished = false;
	Thread.start();
}

void TThreadedNeRFExecutor :: Stop()		//Завершение работы потока
{
	Stopped = true;
}

void TThreadedNeRFExecutor :: RenderView(
	const torch::Tensor render_pose,	//rays?  	std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() };			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
	int w,
	int h,
	const torch::Tensor k,
	const RenderParams &rparams
){
	RenderPose = render_pose;
	W = w;
	H = h;
	K = k;
	RParams = rparams;
	NeedUpdate = true;
}

//SLOTS
void TThreadedNeRFExecutor :: Process()	///Действия, выполняемые потоком
{
	////Запуск таймера
	//Timer.start(rate);

	while (!Stopped)
	{
		if (NeedUpdate)
		{
			torch::NoGradGuard no_grad;
			
			int render_factor = 13;
			NeedUpdate = false;
			while (NeedUpdate == false && render_factor >= 1)
			{
				NeedUpdate = false;
				RParams.RenderFactor = (render_factor > 1) ? render_factor : 0;
				emit UpdateResult(Executor->RenderView(RenderPose, W, H, K, RParams));
				(render_factor > 5) ? render_factor -= 4 : render_factor -= 2;
			}
		}
		Thread.msleep(1);
	};

	//Timer.stop();
	Thread.quit();
	IsFinished = true;
	emit Finished();
}