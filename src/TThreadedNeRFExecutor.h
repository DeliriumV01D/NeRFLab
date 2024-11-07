#pragma once

#include "NeRFExecutor.h"

#include <QThread.h>

class TThreadedNeRFExecutor : public QObject {
	Q_OBJECT
public:
	using TExecutor = NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall, NeRFRenderer<CuHashEmbedder, CuSHEncoder, NeRFSmall>,
		CuHashEmbedder, LeRF, LeRFRenderer>;
protected:
	bool Stopped = false;
	bool IsFinished = true;
	bool NeedUpdate = false;

	torch::Tensor RenderPose;
	int W;
	int H;
	torch::Tensor K;
	NeRFRenderParams RParams;

	QThread Thread;
	//QMutex Mutex;

public:
	//!!!Возможно здесь следует расположить интерфейсный базовый класс для NeRFExecutor
	//std::unique_ptr<NeRFExecutor <CuHashEmbedder, CuSHEncoder, NeRFSmall, NeRFRenderer<CuHashEmbedder, CuSHEncoder, NeRFSmall>>> Executor = nullptr;
	std::unique_ptr<TExecutor> Executor = nullptr;

	TThreadedNeRFExecutor();
	virtual ~TThreadedNeRFExecutor();
	
	void RenderView(
		const torch::Tensor render_pose,	//rays?  	std::pair<torch::Tensor, torch::Tensor> rays = { torch::Tensor(), torch::Tensor() };			///array of shape[2, batch_size, 3].Ray origin and direction for each example in batch.
		int w,
		int h,
		const torch::Tensor k,
		const NeRFRenderParams &rparams
	);
	void SetExecutor(std::unique_ptr<TExecutor> &executor);
	void SetLeRFPrompts(const std::string &lerf_positives, const std::vector<std::string> &lerf_negatives);
	std::tuple<torch::Tensor, torch::Tensor> GetLeRFPrompts();
	void Initialize();
	void Finalize();
	void Start();		///Запуск потока
	void Stop();		///Завершение работы потока
public slots:
	void Process();	///Действия, выполняемые потоком
signals:
	void Finished();
	void Error(QString err);
	void UpdateResult(std::tuple <NeRFRenderResult, LeRFRenderResult> render_result);
};